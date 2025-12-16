use crate::memory_system::MemorySystem;
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
// CHANGE: Use Tokio Mutex to match mcp_server
use std::fs::File;
use tokio::sync::Mutex;

pub fn spawn_shadow_watcher(
    // CHANGE: Accept the same type that mcp_server creates
    memory_system: Arc<Mutex<MemorySystem>>,
) {
    thread::spawn(move || {
        eprintln!("Shadow Brain: Watcher thread started.");

        // We need a runtime handle to call async methods from this sync thread
        let rt = tokio::runtime::Handle::current();

        let mut processed_bubbles = HashSet::new();
        let storage_dir = get_cursor_storage_dir();

        if !storage_dir.exists() {
            eprintln!(
                "Shadow Brain: Cursor storage directory not found at {:?}",
                storage_dir
            );
            return;
        }

        eprintln!("Shadow Brain: Watching {:?}", storage_dir);

        let (tx, rx) = channel();
        let mut watcher = match RecommendedWatcher::new(tx, Config::default()) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Shadow Brain: Failed to create watcher: {}", e);
                return;
            }
        };

        if let Err(e) = watcher.watch(&storage_dir, RecursiveMode::Recursive) {
            eprintln!("Shadow Brain: Failed to watch directory: {}", e);
            return;
        }

        let debounce_time = Duration::from_secs(5);
        let mut last_scan = Instant::now();

        // Initial scan (using block_on to bridge async lock)
        rt.block_on(async {
            scan_and_ingest(&storage_dir, &memory_system, &mut processed_bubbles).await;
        });

        loop {
            match rx.recv() {
                Ok(Ok(event)) => {
                    let relevant = event
                        .paths
                        .iter()
                        .any(|p| p.file_name().map(|n| n == "state.vscdb").unwrap_or(false));

                    if relevant {
                        if last_scan.elapsed() > debounce_time {
                            eprintln!("Shadow Brain: Detected change, scanning...");
                            thread::sleep(Duration::from_millis(500));
                            // Use block_on to call the async scan function
                            rt.block_on(async {
                                scan_and_ingest(
                                    &storage_dir,
                                    &memory_system,
                                    &mut processed_bubbles,
                                )
                                .await;
                            });
                            last_scan = Instant::now();
                        }
                    }
                }
                Ok(Err(e)) => eprintln!("Shadow Brain: Watch error: {}", e),
                Err(_) => break,
            }
        }
    });
}

// ... get_cursor_storage_dir remains the same ...
fn get_cursor_storage_dir() -> PathBuf {
    if let Ok(env_dir) = std::env::var("CURSOR_STORAGE_DIR") {
        return PathBuf::from(env_dir);
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());

    #[cfg(target_os = "linux")]
    {
        Path::new(&home).join(".config/Cursor/User/workspaceStorage")
    }
    #[cfg(target_os = "macos")]
    {
        Path::new(&home).join("Library/Application Support/Cursor/User/workspaceStorage")
    }
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| ".".to_string());
        Path::new(&appdata).join("Cursor/User/workspaceStorage")
    }
}

// CHANGE: Make this async to handle the lock
async fn scan_and_ingest(
    root: &Path,
    memory_system: &Arc<Mutex<MemorySystem>>,
    processed_bubbles: &mut HashSet<String>,
) {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut new_memories = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let db_path = path.join("state.vscdb");
            if db_path.exists() {
                let project_name = resolve_workspace_name(&path);
                if let Some(mems) = extract_from_db(&db_path, &project_name, processed_bubbles) {
                    new_memories.extend(mems);
                }
            }
        }
    }

    if !new_memories.is_empty() {
        eprintln!(
            "Shadow Brain: Ingesting {} new memories...",
            new_memories.len()
        );
        // CHANGE: async lock
        let mut ms = memory_system.lock().await;
        for mem in new_memories {
            if let Err(e) = ms.ingest(&mem) {
                eprintln!("Shadow Brain: Ingestion failed: {}", e);
            }
        }
    }
}

fn resolve_workspace_name(path: &Path) -> String {
    let json_path = path.join("workspace.json");
    if json_path.exists() {
        if let Ok(file) = File::open(json_path) {
            if let Ok(json) = serde_json::from_reader::<_, serde_json::Value>(file) {
                if let Some(folder) = json.get("folder").and_then(|v| v.as_str()) {
                    let decoded = urlencoding::decode(folder).unwrap_or_default();
                    if let Some(name) = Path::new(decoded.as_ref()).file_name() {
                        return format!("[Project: {}] ", name.to_string_lossy());
                    }
                }
            }
        }
    }
    String::new()
}

fn extract_from_db(
    db_path: &Path,
    project_context: &str,
    processed_bubbles: &mut HashSet<String>,
) -> Option<Vec<String>> {
    // Snapshot to temp file to avoid locking
    let temp_dir = std::env::temp_dir();
    let temp_db = temp_dir.join(format!("shadow_{}.db", uuid::Uuid::new_v4()));

    std::fs::copy(db_path, &temp_db).ok()?;
    // Try to copy WAL/SHM if they exist
    let _ = std::fs::copy(
        db_path.with_extension("vscdb-wal"),
        temp_db.with_extension("db-wal"),
    );
    let _ = std::fs::copy(
        db_path.with_extension("vscdb-shm"),
        temp_db.with_extension("db-shm"),
    );

    let conn = Connection::open_with_flags(
        &temp_db,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .ok()?;

    let mut memories = Vec::new();

    // 1. Sidebar Chats (ItemTable)
    {
        // Rusqlite prepare returns Result
        if let Ok(mut stmt) = conn.prepare(
            "SELECT value FROM ItemTable WHERE key = 'workbench.panel.aichat.view.aichat.chatdata'",
        ) {
            if let Ok(mut rows) = stmt.query([]) {
                while let Ok(Some(row)) = rows.next() {
                    let json_str: String = row.get(0).unwrap_or_default();
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&json_str) {
                        if let Some(tabs) = data.get("tabs").and_then(|v| v.as_array()) {
                            for tab in tabs {
                                if let Some(bubbles) = tab.get("bubbles").and_then(|v| v.as_array())
                                {
                                    for bubble in bubbles {
                                        if let Some(text) = bubble
                                            .get("text")
                                            .or(bubble.get("rawText"))
                                            .and_then(|v| v.as_str())
                                        {
                                            let id = bubble
                                                .get("id")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string())
                                                .unwrap_or_else(|| {
                                                    format!("{:x}", md5::compute(text.as_bytes()))
                                                });

                                            if !processed_bubbles.contains(&id) {
                                                let type_val = bubble
                                                    .get("type")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("unknown");
                                                let role =
                                                    if type_val == "user" { "User" } else { "AI" };
                                                memories.push(format!(
                                                    "{}{}: {}",
                                                    project_context, role, text
                                                ));
                                                processed_bubbles.insert(id);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Composer Chats (cursorDiskKV)
    {
        // Check if table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='cursorDiskKV'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0)
            > 0;

        if table_exists {
            if let Ok(mut stmt) =
                conn.prepare("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'")
            {
                if let Ok(mut rows) = stmt.query([]) {
                    while let Ok(Some(row)) = rows.next() {
                        let val_str: String = row.get(1).unwrap_or_default();
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&val_str) {
                            if let Some(headers) = data
                                .get("fullConversationHeadersOnly")
                                .and_then(|v| v.as_array())
                            {
                                for header in headers {
                                    if let Some(bubble_id) =
                                        header.get("bubbleId").and_then(|v| v.as_str())
                                    {
                                        if processed_bubbles.contains(bubble_id) {
                                            continue;
                                        }

                                        // Need to fetch the bubble content from DB
                                        // Nested query inside loop is bad but simple for now
                                        if let Ok(mut bubble_stmt) = conn
                                            .prepare("SELECT value FROM cursorDiskKV WHERE key = ?")
                                        {
                                            if let Ok(bubble_val) = bubble_stmt
                                                .query_row([bubble_id], |r| r.get::<_, String>(0))
                                            {
                                                if let Ok(b_data) =
                                                    serde_json::from_str::<serde_json::Value>(
                                                        &bubble_val,
                                                    )
                                                {
                                                    if let Some(text) = b_data
                                                        .get("text")
                                                        .or(b_data.get("rawText"))
                                                        .and_then(|v| v.as_str())
                                                    {
                                                        let role = if b_data
                                                            .get("type")
                                                            .and_then(|v| v.as_i64())
                                                            .unwrap_or(0)
                                                            == 1
                                                        {
                                                            "User"
                                                        } else {
                                                            "AI"
                                                        };
                                                        memories.push(format!(
                                                            "{}{}: {}",
                                                            project_context, role, text
                                                        ));
                                                        processed_bubbles
                                                            .insert(bubble_id.to_string());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Cleanup temp file
    let _ = std::fs::remove_file(&temp_db);
    let _ = std::fs::remove_file(temp_db.with_extension("db-wal"));
    let _ = std::fs::remove_file(temp_db.with_extension("db-shm"));

    if memories.is_empty() {
        None
    } else {
        Some(memories)
    }
}
