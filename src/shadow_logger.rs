use anyhow::{anyhow, Result};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

pub struct ShadowLogger {
    pub processed_bubbles: HashSet<String>,
    cursor_storage_dir: PathBuf,
}

impl ShadowLogger {
    pub fn new() -> Self {
        let cursor_storage_dir = Self::get_cursor_config_dir();
        Self {
            processed_bubbles: HashSet::new(),
            cursor_storage_dir,
        }
    }

    fn get_cursor_config_dir() -> PathBuf {
        if let Ok(dir) = env::var("CURSOR_STORAGE_DIR") {
            return PathBuf::from(dir);
        }

        let home = env::var("HOME").expect("HOME not set");
        let default_path = if cfg!(target_os = "macos") {
            PathBuf::from(format!(
                "{}/Library/Application Support/Cursor/User/workspaceStorage",
                home
            ))
        } else {
            // Linux / default
            PathBuf::from(format!("{}/.config/Cursor/User/workspaceStorage", home))
        };

        default_path
    }

    pub fn extract_new_memories(&mut self) -> Result<Vec<String>> {
        let mut new_memories = Vec::new();
        let dbs = self.get_workspace_dbs()?;

        debug!("Found {} workspace DBs", dbs.len());

        // Use a temp directory for snapshots
        let temp_dir = tempfile::tempdir()?;

        for db_path in dbs {
            let workspace_name = self.resolve_workspace_name(&db_path).unwrap_or_default();

            // Snapshot
            let snapshot_path = match self.snapshot_database(&db_path, temp_dir.path()) {
                Ok(p) => p,
                Err(e) => {
                    warn!("Failed to snapshot {:?}: {}", db_path, e);
                    continue;
                }
            };

            // Extract
            match self.process_database(&snapshot_path, &workspace_name) {
                Ok(mems) => new_memories.extend(mems),
                Err(e) => warn!("Error processing DB {:?}: {}", snapshot_path, e),
            }
        }

        Ok(new_memories)
    }

    fn get_workspace_dbs(&self) -> Result<Vec<PathBuf>> {
        if !self.cursor_storage_dir.exists() {
            return Ok(vec![]);
        }

        let mut dbs = Vec::new();
        for entry in fs::read_dir(&self.cursor_storage_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let db_path = path.join("state.vscdb");
                if db_path.exists() {
                    dbs.push(db_path);
                }
            }
        }
        Ok(dbs)
    }

    fn resolve_workspace_name(&self, db_path: &Path) -> Option<String> {
        let parent = db_path.parent()?;
        let ws_json = parent.join("workspace.json");
        if ws_json.exists() {
            if let Ok(content) = fs::read_to_string(ws_json) {
                if let Ok(json) = serde_json::from_str::<Value>(&content) {
                    if let Some(folder) = json.get("folder").and_then(|v| v.as_str()) {
                        // Decode URI component if needed, simplistic approach for now
                        let name = folder.split('/').last().unwrap_or("Unknown");
                        return Some(format!("[Project: {}] ", name));
                    }
                }
            }
        }
        None
    }

    fn snapshot_database(&self, source: &Path, temp_dir: &Path) -> Result<PathBuf> {
        let file_name = source
            .file_name()
            .ok_or_else(|| anyhow!("Invalid source path"))?;
        let target = temp_dir.join(file_name);

        fs::copy(source, &target)?;

        // Try copying WAL/SHM if they exist
        let _ = fs::copy(
            format!("{}-wal", source.display()),
            format!("{}-wal", target.display()),
        );
        let _ = fs::copy(
            format!("{}-shm", source.display()),
            format!("{}-shm", target.display()),
        );

        Ok(target)
    }

    fn process_database(&mut self, db_path: &Path, context: &str) -> Result<Vec<String>> {
        let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut memories = Vec::new();

        // 1. Sidebar Chats
        memories.extend(self.extract_sidebar_chats(&conn, context)?);

        // 2. Composer Chats
        memories.extend(self.extract_composer_chats(&conn, context)?);

        Ok(memories)
    }

    fn extract_sidebar_chats(&mut self, conn: &Connection, context: &str) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            "SELECT value FROM ItemTable WHERE key = 'workbench.panel.aichat.view.aichat.chatdata'",
        )?;
        let mut rows = stmt.query([])?;

        let mut extracted = Vec::new();

        while let Some(row) = rows.next()? {
            let json_str: String = row.get(0)?;
            if let Ok(data) = serde_json::from_str::<Value>(&json_str) {
                if let Some(tabs) = data.get("tabs").and_then(|t| t.as_array()) {
                    for tab in tabs {
                        if let Some(bubbles) = tab.get("bubbles").and_then(|b| b.as_array()) {
                            for bubble in bubbles {
                                if let Some(text) = bubble
                                    .get("text")
                                    .or_else(|| bubble.get("rawText"))
                                    .and_then(|t| t.as_str())
                                {
                                    if text.trim().is_empty() {
                                        continue;
                                    }

                                    let bubble_id = bubble
                                        .get("id")
                                        .and_then(|i| i.as_str())
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| {
                                            format!("{:x}", md5::compute(text.as_bytes()))
                                        });

                                    if self.processed_bubbles.contains(&bubble_id) {
                                        continue;
                                    }

                                    let role = match bubble.get("type").and_then(|t| t.as_str()) {
                                        Some("user") => "User",
                                        Some("ai") => "AI",
                                        _ => "Unknown",
                                    };

                                    let memory = format!("{}{}: {}", context, role, text.trim());
                                    extracted.push(memory);
                                    self.processed_bubbles.insert(bubble_id);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(extracted)
    }

    fn extract_composer_chats(&mut self, conn: &Connection, context: &str) -> Result<Vec<String>> {
        // Check if table exists
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='cursorDiskKV'")?;
        if !stmt.exists([])? {
            return Ok(vec![]);
        }

        let mut stmt =
            conn.prepare("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'")?;
        let mut rows = stmt.query([])?;
        let mut extracted = Vec::new();

        // Collect keys to query later to avoid nested query issues if any
        let mut bubble_ids_to_fetch = Vec::new();

        while let Some(row) = rows.next()? {
            let val_str: String = row.get(1)?;
            if let Ok(data) = serde_json::from_str::<Value>(&val_str) {
                if let Some(headers) = data
                    .get("fullConversationHeadersOnly")
                    .and_then(|h| h.as_array())
                {
                    for header in headers {
                        if let Some(bid) = header.get("bubbleId").and_then(|s| s.as_str()) {
                            if !self.processed_bubbles.contains(bid) {
                                bubble_ids_to_fetch.push(bid.to_string());
                            }
                        }
                    }
                }
            }
        }

        // Now fetch bubbles
        for bid in bubble_ids_to_fetch {
            let mut b_stmt = conn.prepare("SELECT value FROM cursorDiskKV WHERE key = ?")?;
            let mut b_rows = b_stmt.query([&bid])?;
            if let Some(row) = b_rows.next()? {
                let val_str: String = row.get(0)?;
                if let Ok(b_data) = serde_json::from_str::<Value>(&val_str) {
                    let text = b_data
                        .get("text")
                        .or_else(|| b_data.get("rawText"))
                        .and_then(|t| t.as_str());
                    if let Some(t) = text {
                        if !t.trim().is_empty() {
                            let role_type =
                                b_data.get("type").and_then(|v| v.as_i64()).unwrap_or(0);
                            let role = if role_type == 1 { "User" } else { "AI" };

                            let memory = format!("{}{}: {}", context, role, t.trim());
                            extracted.push(memory);
                            self.processed_bubbles.insert(bid);
                        }
                    }
                }
            }
        }

        Ok(extracted)
    }
}
