use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;

use anyhow::Error;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::SplatMemoryConfig;
use crate::embeddings::EmbeddingModel;
use crate::indexing::fingerprint::{fingerprint_from_splat, wasserstein_distance};
use crate::indexing::TantivyIndex;
use crate::indexing::TopologicalFingerprint;
use crate::llm::ollama::OllamaClient;
use crate::memory_system::MemorySystem;
use crate::retrieval::{recall_episode, subconscious_priming, HybridRetriever};
use crate::storage::{InMemoryBlobStore, OpaqueSplatRef, TopologicalMemoryStore};
use crate::tivm::SplatRagConfig;
use crate::types::{SplatId, SplatInput, SplatMeta};

pub type AppResult<T> = std::result::Result<T, AppError>;

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .route("/perceive", post(perceive))
        .route("/search_topological", post(search_topological))
        .route("/store_eposodic", post(store_eposodic))
        .route("/priming_hint", post(priming_hint))
        .route("/recall_episode", post(recall_episode_handler))
        .route("/chat", post(chat_handler))
        .route("/reflex", post(reflex_search)) // The Product API
        .route("/ingest", post(ingest_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
}

async fn auth_middleware(
    State(_state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    Ok(next.run(req).await)
}

#[derive(Clone)]
pub struct AppState {
    config: SplatMemoryConfig,
    rag_config: SplatRagConfig,
    api_key: Option<String>,
    store: Arc<Mutex<TopologicalMemoryStore<InMemoryBlobStore>>>,
    pub memory_system: Arc<TokioMutex<MemorySystem>>, // New Memory System
    embedding_model: Arc<EmbeddingModel>,
    tantivy_index: Arc<TantivyIndex>,
    temp_cache: Arc<Mutex<HashMap<String, CachedFingerprint>>>,
    temp_counter: Arc<AtomicU64>,
    metrics: Arc<AppMetrics>,
    llm_client: Arc<OllamaClient>, // Added LLM Client
}

impl AppState {
    pub fn new(
        config: SplatMemoryConfig,
        rag_config: SplatRagConfig,
        api_key: Option<String>,
        store: TopologicalMemoryStore<InMemoryBlobStore>,
        memory_system: MemorySystem,
        embedding_model: Arc<EmbeddingModel>,
        tantivy_index: Arc<TantivyIndex>,
    ) -> Self {
        Self {
            config,
            rag_config,
            api_key,
            store: Arc::new(Mutex::new(store)),
            memory_system: Arc::new(TokioMutex::new(memory_system)),
            embedding_model,
            tantivy_index,
            temp_cache: Arc::new(Mutex::new(HashMap::new())),
            temp_counter: Arc::new(AtomicU64::new(1)),
            metrics: Arc::new(AppMetrics::default()),
            llm_client: Arc::new(OllamaClient::new(Some("gemma3:4b-it-qat".to_string()))),
        }
    }

    pub fn store(&self) -> Arc<Mutex<TopologicalMemoryStore<InMemoryBlobStore>>> {
        self.store.clone()
    }

    pub fn next_temp_id(&self) -> String {
        let id = self.temp_counter.fetch_add(1, Ordering::Relaxed);
        format!("temp_fingerprint_{:016x}", id)
    }

    fn cached_fingerprint(&self, id: &str) -> AppResult<CachedFingerprint> {
        let cache = self
            .temp_cache
            .lock()
            .map_err(|_| AppError::internal("temp cache poisoned"))?;
        cache
            .get(id)
            .cloned()
            .ok_or_else(|| AppError::cache_miss(id.to_string()))
    }
}

#[derive(Debug, Default)]
pub struct AppMetrics {
    perceive_calls: AtomicU64,
    search_calls: AtomicU64,
    store_calls: AtomicU64,
    priming_calls: AtomicU64,
    recall_calls: AtomicU64,
    // Latency tracking (in microseconds)
    perceive_latency_us: AtomicU64,
    search_latency_us: AtomicU64,
    store_latency_us: AtomicU64,
    priming_latency_us: AtomicU64,
    recall_latency_us: AtomicU64,
    // Operation counts for latency averaging
    perceive_latency_count: AtomicU64,
    search_latency_count: AtomicU64,
    store_latency_count: AtomicU64,
    priming_latency_count: AtomicU64,
    recall_latency_count: AtomicU64,
}

impl AppMetrics {
    fn record_perceive(&self) {
        self.perceive_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn record_perceive_latency(&self, latency_us: u64) {
        self.perceive_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.perceive_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_search(&self) {
        self.search_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn record_search_latency(&self, latency_us: u64) {
        self.search_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.search_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_store(&self) {
        self.store_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn record_store_latency(&self, latency_us: u64) {
        self.store_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.store_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_priming(&self) {
        self.priming_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn record_priming_latency(&self, latency_us: u64) {
        self.priming_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.priming_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_recall(&self) {
        self.recall_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn record_recall_latency(&self, latency_us: u64) {
        self.recall_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.recall_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            perceive_calls: self.perceive_calls.load(Ordering::Relaxed),
            search_calls: self.search_calls.load(Ordering::Relaxed),
            store_calls: self.store_calls.load(Ordering::Relaxed),
            priming_calls: self.priming_calls.load(Ordering::Relaxed),
            recall_calls: self.recall_calls.load(Ordering::Relaxed),
            perceive_latency_us: self.perceive_latency_us.load(Ordering::Relaxed),
            search_latency_us: self.search_latency_us.load(Ordering::Relaxed),
            store_latency_us: self.store_latency_us.load(Ordering::Relaxed),
            priming_latency_us: self.priming_latency_us.load(Ordering::Relaxed),
            recall_latency_us: self.recall_latency_us.load(Ordering::Relaxed),
            perceive_latency_count: self.perceive_latency_count.load(Ordering::Relaxed),
            search_latency_count: self.search_latency_count.load(Ordering::Relaxed),
            store_latency_count: self.store_latency_count.load(Ordering::Relaxed),
            priming_latency_count: self.priming_latency_count.load(Ordering::Relaxed),
            recall_latency_count: self.recall_latency_count.load(Ordering::Relaxed),
            // Average latencies computed later in compute_averages()
            avg_perceive_latency_ms: None,
            avg_search_latency_ms: None,
            avg_store_latency_ms: None,
            avg_priming_latency_ms: None,
            avg_recall_latency_ms: None,
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct MetricsSnapshot {
    perceive_calls: u64,
    search_calls: u64,
    store_calls: u64,
    priming_calls: u64,
    recall_calls: u64,
    // Latency metrics (microseconds)
    perceive_latency_us: u64,
    search_latency_us: u64,
    store_latency_us: u64,
    priming_latency_us: u64,
    recall_latency_us: u64,
    // Latency counts for averaging
    perceive_latency_count: u64,
    search_latency_count: u64,
    store_latency_count: u64,
    priming_latency_count: u64,
    recall_latency_count: u64,
    // Computed average latencies (milliseconds)
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_perceive_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_search_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_store_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_priming_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_recall_latency_ms: Option<f64>,
}

impl MetricsSnapshot {
    fn compute_averages(mut self) -> Self {
        self.avg_perceive_latency_ms = if self.perceive_latency_count > 0 {
            Some(self.perceive_latency_us as f64 / self.perceive_latency_count as f64 / 1000.0)
        } else {
            None
        };

        self.avg_search_latency_ms = if self.search_latency_count > 0 {
            Some(self.search_latency_us as f64 / self.search_latency_count as f64 / 1000.0)
        } else {
            None
        };

        self.avg_store_latency_ms = if self.store_latency_count > 0 {
            Some(self.store_latency_us as f64 / self.store_latency_count as f64 / 1000.0)
        } else {
            None
        };

        self.avg_priming_latency_ms = if self.priming_latency_count > 0 {
            Some(self.priming_latency_us as f64 / self.priming_latency_count as f64 / 1000.0)
        } else {
            None
        };

        self.avg_recall_latency_ms = if self.recall_latency_count > 0 {
            Some(self.recall_latency_us as f64 / self.recall_latency_count as f64 / 1000.0)
        } else {
            None
        };

        self
    }
}

#[derive(Debug, Clone)]
struct CachedFingerprint {
    splat: SplatInput,
    fingerprint: TopologicalFingerprint,
    embedding: Vec<f32>,
    blob: Option<OpaqueSplatRef>,
}

#[derive(Debug, Deserialize)]
struct PerceiveRequest {
    splat: SplatInput,
    #[serde(default)]
    blob_handle: Option<String>,
}

#[derive(Debug, Serialize)]
struct PerceiveResponse {
    fingerprint_id: String,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum SearchMode {
    Priming,
    Recall,
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Priming
    }
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    fingerprint_id: Option<String>,
    query_text: Option<String>, // Added for Hybrid Search
    k: usize,
    #[serde(default)]
    mode: SearchMode,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
struct SearchHit {
    splat_id: SplatId,
    distance: f32,
    radiance: Option<f32>, // Added for Radiance visibility
    caption: String,
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StoreRequest {
    fingerprint_id: String,
    #[serde(default)]
    agent_notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct StoreResponse {
    splat_id: SplatId,
    status: &'static str,
}

#[derive(Debug, Deserialize)]
struct PrimingRequest {
    fingerprint_id: String,
    k: usize,
}

#[derive(Debug, Serialize)]
struct PrimingResponse {
    hints: Vec<SearchHit>,
}

#[derive(Debug, Deserialize)]
struct RecallEpisodeRequest {
    fingerprint_id: String,
    steps: usize,
}

#[derive(Debug, Serialize)]
struct RecallEpisodeResponse {
    steps: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
struct MetricsResponse {
    perceive_calls: u64,
    search_calls: u64,
    store_calls: u64,
    priming_calls: u64,
    recall_calls: u64,
    cached_fingerprints: usize,
    stored_memories: usize,
    // Average latencies in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_perceive_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_search_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_store_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_priming_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_recall_latency_ms: Option<f64>,
}

// New Chat Types
#[derive(Debug, Deserialize)]
struct ChatRequest {
    query: String,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    response: String,
}

#[derive(Debug, Deserialize)]
struct ReflexRequest {
    query: String,
    #[serde(default)]
    mode: String,
}

#[derive(Debug, Serialize)]
struct ReflexResponse {
    results: Vec<SearchHit>,
    meta: ReflexMeta,
}

#[derive(Debug, Serialize)]
struct ReflexMeta {
    weight: f32,
    std_dev: f32,
}

#[derive(Debug)]
pub enum AppError {
    CacheMiss(String),
    BadRequest(String),
    Internal(String),
}

impl AppError {
    fn cache_miss(id: String) -> Self {
        Self::CacheMiss(id)
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

impl From<Error> for AppError {
    fn from(err: Error) -> Self {
        AppError::Internal(err.to_string())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::CacheMiss(id) => (
                StatusCode::NOT_FOUND,
                format!("unknown fingerprint_id: {}", id),
            ),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, Json(json!({ "error": message }))).into_response()
    }
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

async fn metrics(State(state): State<AppState>) -> AppResult<Json<MetricsResponse>> {
    let counters = state.metrics.snapshot().compute_averages();
    let cached_fingerprints = state
        .temp_cache
        .lock()
        .map_err(|_| AppError::internal("temp cache poisoned"))?
        .len();
    let stored_memories = state
        .store
        .lock()
        .map_err(|_| AppError::internal("memory store poisoned"))?
        .len();

    Ok(Json(MetricsResponse {
        perceive_calls: counters.perceive_calls,
        search_calls: counters.search_calls,
        store_calls: counters.store_calls,
        priming_calls: counters.priming_calls,
        recall_calls: counters.recall_calls,
        cached_fingerprints,
        stored_memories,
        avg_perceive_latency_ms: counters.avg_perceive_latency_ms,
        avg_search_latency_ms: counters.avg_search_latency_ms,
        avg_store_latency_ms: counters.avg_store_latency_ms,
        avg_priming_latency_ms: counters.avg_priming_latency_ms,
        avg_recall_latency_ms: counters.avg_recall_latency_ms,
    }))
}

async fn perceive(
    State(state): State<AppState>,
    Json(payload): Json<PerceiveRequest>,
) -> AppResult<Json<PerceiveResponse>> {
    let mut splat = payload.splat;
    if splat.meta.timestamp.is_none() {
        splat.meta.timestamp = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        );
    }

    let fingerprint = fingerprint_from_splat(&splat, &state.rag_config);
    let embedding = fingerprint.to_vector();
    let blob = payload.blob_handle.map(OpaqueSplatRef::External);

    let cache_entry = CachedFingerprint {
        splat,
        fingerprint,
        embedding,
        blob,
    };

    let fingerprint_id = state.next_temp_id();

    let mut cache = state
        .temp_cache
        .lock()
        .map_err(|_| AppError::internal("temp cache poisoned"))?;
    cache.insert(fingerprint_id.clone(), cache_entry);

    state.metrics.record_perceive();

    Ok(Json(PerceiveResponse { fingerprint_id }))
}

async fn search_topological(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> AppResult<Json<SearchResponse>> {
    if payload.k == 0 {
        return Err(AppError::bad_request("k must be greater than 0"));
    }

    let store = state
        .store
        .lock()
        .map_err(|_| AppError::internal("memory store poisoned"))?;

    // Hybrid Search Path
    if let Some(query) = payload.query_text {
        let retriever = HybridRetriever::new(
            &state.tantivy_index,
            &store,
            &state.embedding_model,
            &state.config,
        );

        let scored_memories = retriever.search(&query, payload.k);
        let results = scored_memories
            .into_iter()
            .map(|m| {
                let record = store.get(m.id);
                let (caption, tags) = if let Some(rec) = record {
                    generate_caption(m.id, &rec.meta, SearchMode::Recall)
                } else {
                    ("Unknown Memory".to_string(), vec![])
                };

                SearchHit {
                    splat_id: m.id,
                    distance: m.score, // In Hybrid mode, this is Score, not Distance
                    radiance: Some(m.radiance),
                    caption,
                    tags,
                }
            })
            .collect();

        state.metrics.record_search();
        return Ok(Json(SearchResponse { results }));
    }

    // Legacy Topological Search Path
    if let Some(fid) = payload.fingerprint_id {
        let cache_entry = state.cached_fingerprint(&fid)?;
        let mut hits = store.search_embeddings(&cache_entry.embedding, payload.k)?;
        let mode = payload.mode;
        let mut results = Vec::with_capacity(hits.len());

        for (splat_id, ann_distance) in hits.drain(..) {
            if let Some(record) = store.get(splat_id) {
                let distance = match mode {
                    SearchMode::Priming => ann_distance,
                    SearchMode::Recall => {
                        wasserstein_distance(&cache_entry.fingerprint, &record.fingerprint)
                    }
                };
                let (caption, mut tags) = generate_caption(splat_id, &record.meta, mode);
                if matches!(mode, SearchMode::Recall) {
                    tags.push("recall".into());
                }
                results.push(SearchHit {
                    splat_id,
                    distance,
                    radiance: None,
                    caption,
                    tags,
                });
            }
        }

        if matches!(mode, SearchMode::Recall) {
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        state.metrics.record_search();
        return Ok(Json(SearchResponse { results }));
    }

    Err(AppError::bad_request(
        "Either query_text or fingerprint_id required",
    ))
}

async fn store_eposodic(
    State(state): State<AppState>,
    Json(payload): Json<StoreRequest>,
) -> AppResult<Json<StoreResponse>> {
    let mut cache = state
        .temp_cache
        .lock()
        .map_err(|_| AppError::internal("temp cache poisoned"))?;
    let mut cache_entry = cache
        .remove(&payload.fingerprint_id)
        .ok_or_else(|| AppError::cache_miss(payload.fingerprint_id.clone()))?;
    drop(cache);

    if let Some(notes) = payload.agent_notes.as_ref().and_then(|n| {
        let trimmed = n.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }) {
        cache_entry
            .splat
            .meta
            .labels
            .push(format!("agent_note:{}", notes));
    }

    let blob = cache_entry
        .blob
        .take()
        .unwrap_or_else(|| OpaqueSplatRef::External("memory_palace://ephemeral".into()));

    let mut store = state
        .store
        .lock()
        .map_err(|_| AppError::internal("memory store poisoned"))?;

    // 2. Prepare Text Content (from labels)
    // We extract text from labels for now as a proxy for memory content
    let text_content = cache_entry.splat.meta.labels.join(" ");

    // 1. Store in Topological Store (Vector + Splat)
    let splat_id = store.add_splat(
        &cache_entry.splat,
        blob,
        text_content.clone(),
        cache_entry.embedding.clone(),
    )?;

    // 3. Index in Tantivy (The Grip)
    if !text_content.is_empty() {
        state
            .tantivy_index
            .add_document(splat_id, &text_content, &cache_entry.splat.meta.labels)
            .map_err(|e| AppError::internal(format!("Tantivy error: {}", e)))?;
    }

    state.metrics.record_store();

    Ok(Json(StoreResponse {
        splat_id,
        status: "stored",
    }))
}

async fn priming_hint(
    State(state): State<AppState>,
    Json(payload): Json<PrimingRequest>,
) -> AppResult<Json<PrimingResponse>> {
    if payload.k == 0 {
        return Err(AppError::bad_request("k must be greater than 0"));
    }

    let cache_entry = state.cached_fingerprint(&payload.fingerprint_id)?;
    let store = state
        .store
        .lock()
        .map_err(|_| AppError::internal("memory store poisoned"))?;

    let contexts = subconscious_priming(&store, &cache_entry.splat, &state.rag_config, payload.k)?;
    let hints = contexts
        .into_iter()
        .map(|ctx| {
            let (caption, mut tags) =
                generate_caption(ctx.splat_id, &ctx.meta, SearchMode::Priming);
            if tags.is_empty() {
                tags.push("priming".into());
            }
            SearchHit {
                splat_id: ctx.splat_id,
                distance: ctx.distance,
                radiance: None,
                caption,
                tags,
            }
        })
        .collect();

    state.metrics.record_priming();

    Ok(Json(PrimingResponse { hints }))
}

async fn recall_episode_handler(
    State(state): State<AppState>,
    Json(payload): Json<RecallEpisodeRequest>,
) -> AppResult<Json<RecallEpisodeResponse>> {
    if payload.steps == 0 {
        return Err(AppError::bad_request("steps must be greater than 0"));
    }

    let cache_entry = state.cached_fingerprint(&payload.fingerprint_id)?;
    let store = state
        .store
        .lock()
        .map_err(|_| AppError::internal("memory store poisoned"))?;

    let steps = recall_episode(
        &cache_entry.splat,
        payload.steps,
        &store,
        &state.rag_config,
        |result| {
            store
                .get(result.splat_id)
                .map(|record| record.splat.clone())
        },
    )?
    .into_iter()
    .map(|step| {
        let (caption, mut tags) = generate_caption(step.splat_id, &step.meta, SearchMode::Recall);
        tags.push("recall".into());
        SearchHit {
            splat_id: step.splat_id,
            distance: step.distance,
            radiance: None,
            caption,
            tags,
        }
    })
    .collect();

    state.metrics.record_recall();

    Ok(Json(RecallEpisodeResponse { steps }))
}

// --- LLM Integration ---

async fn chat_handler(
    State(state): State<AppState>,
    Json(payload): Json<ChatRequest>,
) -> AppResult<Json<ChatResponse>> {
    // 1. Retrieve Holographic Context via Hybrid Search
    let context_str = {
        let store = state
            .store
            .lock()
            .map_err(|_| AppError::internal("Memory store poisoned"))?;
        let retriever = HybridRetriever::new(
            &state.tantivy_index,
            &store,
            &state.embedding_model,
            &state.config,
        );

        let results = retriever.search(&payload.query, 5);

        let mut ctx = String::new();
        for res in results {
            if let Some(record) = store.get(res.id) {
                ctx.push_str(&format!(
                    "- MEMORY (Score: {:.2}, Radiance: {:.2}): {}\n",
                    res.score,
                    res.radiance,
                    record.meta.labels.join(" ")
                ));
            }
        }

        if ctx.is_empty() {
            ctx = "No relevant memories found.".to_string();
        }
        ctx
    };

    let system_prompt = "You are a helpful AI assistant connected to a SplatRag memory system. 
    Use the provided holographic memories to answer. 
    If integrity is low (<90%), warn the user.";

    // 2. Get LLM Response with Sentiment
    let response_obj = state
        .llm_client
        .chat_with_sentiment(system_prompt, &payload.query, &context_str)
        .await
        .map_err(|e: anyhow::Error| AppError::internal(format!("LLM Error: {}", e)))?;

    // 3. Store Interaction back into Memory (Self-Reflection Loop)
    // We use the LLM's own valence to color this new memory.
    // Note: store_eposodic uses splat input. Here we have text.
    // Ideally we would construct a splat from text and store it.
    // For now, we just log the chat.

    Ok(Json(ChatResponse {
        response: response_obj.response,
    }))
}

async fn reflex_search(
    State(state): State<AppState>,
    Json(payload): Json<ReflexRequest>,
) -> AppResult<Json<ReflexResponse>> {
    // Embed
    let mut embedding = state
        .embedding_model
        .embed(&payload.query)
        .map_err(|e: anyhow::Error| AppError::internal(e.to_string()))?;

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }

    let store = state
        .store
        .lock()
        .map_err(|_| AppError::internal("memory store poisoned"))?;

    let k = 50;
    let hits = store.search_embeddings(&embedding, k)?;

    // Calculate stats: distance = 1 - score
    let scores: Vec<f32> = hits.iter().map(|(_, d)| (1.0f32 - d).max(0.0f32)).collect();
    let stats = crate::ranking::calculate_adaptive_weight(&scores);

    // Format results
    let results = hits
        .into_iter()
        .take(10)
        .map(|(id, dist)| {
            let record = store.get(id);
            let (caption, tags) = if let Some(rec) = record {
                generate_caption(id, &rec.meta, SearchMode::Recall)
            } else {
                ("Unknown".to_string(), vec![])
            };

            SearchHit {
                splat_id: id,
                distance: dist,
                radiance: None,
                caption,
                tags,
            }
        })
        .collect();

    Ok(Json(ReflexResponse {
        results,
        meta: ReflexMeta {
            weight: stats.weight,
            std_dev: stats.std_dev,
        },
    }))
}

fn generate_caption(
    splat_id: SplatId,
    meta: &SplatMeta,
    mode: SearchMode,
) -> (String, Vec<String>) {
    let caption = if let Some(label) = meta.labels.first() {
        format!("{} match around '{}'", mode_label(mode), label)
    } else {
        format!("{} match for splat {}", mode_label(mode), splat_id)
    };

    let mut tags = meta.labels.clone();
    if tags.is_empty() {
        tags.push("untagged".into());
    }

    (caption, tags)
}

fn mode_label(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Priming => "Priming",
        SearchMode::Recall => "Recall",
    }
}

#[derive(Deserialize)]
struct IngestRequest {
    text: String,
    #[allow(dead_code)]
    metadata: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
struct IngestResponse {
    id: u64,
    status: String,
}

async fn ingest_handler(
    State(state): State<AppState>,
    Json(payload): Json<IngestRequest>,
) -> AppResult<Json<IngestResponse>> {
    // 1. Embed
    let embedding = state
        .embedding_model
        .embed(&payload.text)
        .map_err(|e: anyhow::Error| AppError::internal(e.to_string()))?;

    // 2. Create dummy SplatInput
    let pos = if embedding.len() >= 3 {
        [embedding[0], embedding[1], embedding[2]]
    } else {
        [0.0; 3]
    };

    let splat = SplatInput {
        static_points: vec![pos],
        covariances: vec![[0.01; 9]],
        motion_velocities: None,
        meta: SplatMeta {
            timestamp: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            ),
            labels: vec![format!("text:{}", payload.text)],
            emotional_state: None,
            fitness_metadata: None,
        },
        normals: None,
        idiv: None,
        ide: None,
        sss_params: None,
        sh_occlusion: None,
    };

    // 3. Store
    let mut store = state
        .store
        .lock()
        .map_err(|_| AppError::internal("store lock poisoned"))?;

    let id = store
        .add_splat(
            &splat,
            OpaqueSplatRef::External("ingested".to_string()),
            payload.text.clone(),
            embedding,
        )
        .map_err(|e| AppError::internal(e.to_string()))?;

    Ok(Json(IngestResponse {
        id,
        status: "success".to_string(),
    }))
}
