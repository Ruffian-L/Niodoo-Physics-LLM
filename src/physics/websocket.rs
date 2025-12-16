use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct PhysicsUpdate {
    pub step: usize,
    pub positions: Vec<Vec<f32>>, // [N, 3] from Candle
    pub colors: Vec<Vec<f32>>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ControlCommand {
    SetAttractor { id: Option<String>, pos: [f32; 3] },
    SetGravity { strength: f32 },
    Reset,
}

#[derive(Clone)]
struct ServerState {
    tx_update: broadcast::Sender<PhysicsUpdate>,
    tx_control: mpsc::Sender<ControlCommand>,
}

pub async fn start_physics_server(
    port: u16,
    tx_control: mpsc::Sender<ControlCommand>,
) -> broadcast::Sender<PhysicsUpdate> {
    let (tx_update, _rx) = broadcast::channel(100);

    let state = ServerState {
        tx_update: tx_update.clone(),
        tx_control,
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!("Physics WebSocket Server listening on {}", addr);

    // Spawn server in background
    tokio::spawn(async move {
        // Updated axum 0.8 / latest listener style
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    tx_update
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<ServerState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: ServerState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx_update = state.tx_update.subscribe();

    // Spawn broadcast -> ws task
    let mut send_task = tokio::spawn(async move {
        while let Ok(update) = rx_update.recv().await {
            let json = serde_json::to_string(&update).unwrap_or_default();
            if sender.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming control commands
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            if let Ok(cmd) = serde_json::from_str::<ControlCommand>(&text) {
                let _ = state.tx_control.send(cmd).await;
            }
        }
    }

    send_task.abort();
}
