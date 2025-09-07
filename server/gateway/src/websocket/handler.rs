use crate::{
    types::{ConnectionId, PlayerId, TableId},
    websocket::{message::WsMessage, server::GameSessionManagerHandle},
};
use actix_ws::AggregatedMessage;
use futures_util::StreamExt as _;
use std::{
    pin::pin,
    time::{Duration, Instant},
};
use tokio::{sync::mpsc, time::interval};

/// How often heartbeat pings are sent.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
/// How long before lack of client response causes a timeout.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

/// Handles WebSocket communication for a game table.
///
/// # Arguments
///
/// * `player_id` - The player ID.
/// * `table_id` - The table ID.
/// * `game_server` - The game server.
/// * `session` - The WebSocket session.
/// * `stream` - The WebSocket message stream.
pub async fn game_ws(
    player_id: PlayerId,
    table_id: TableId,
    game_server: GameSessionManagerHandle,
    mut session: actix_ws::Session,
    stream: actix_ws::MessageStream,
) {
    let mut last_heartbeat = Instant::now();
    let mut interval = interval(HEARTBEAT_INTERVAL);

    let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
    let conn_id = game_server.connect(conn_tx, &table_id, &player_id).await;
    let mut stream = pin!(stream.aggregate_continuations());

    let close_reason = loop {
        let tick = pin!(interval.tick());
        let messages = pin!(stream.next());
        let msg_rx = pin!(conn_rx.recv());

        tokio::select! {
            // Heartbeat internal tick.
            _ = tick => {
                if Instant::now().duration_since(last_heartbeat) > CLIENT_TIMEOUT {
                    log::info!("Client has not sent heartbeat in over {:?} seconds. Closing connection.", CLIENT_TIMEOUT);
                    break None;
                }
                let _ = session.ping(b"").await;
            }

            // Message from client.
            msg = messages => {
                match msg {
                    Some(Ok(AggregatedMessage::Ping(bytes))) => {
                        last_heartbeat = Instant::now();
                        session.pong(&bytes).await.unwrap();
                    },
                    Some(Ok(AggregatedMessage::Pong(_))) => {
                        last_heartbeat = Instant::now();
                    },
                    Some(Ok(AggregatedMessage::Text(text))) => {
                        let message: WsMessage = serde_json::from_str(&text).unwrap();
                        process_message(&game_server, &mut session, message, &conn_id, &player_id, &table_id).await;
                    },
                    Some(Ok(AggregatedMessage::Binary(_))) => {
                        log::warn!("Unexpected binary message received from client.");
                    },
                    Some(Ok(AggregatedMessage::Close(reason))) => {
                        break reason;
                    },

                    // Client WebSocket stream error.
                    Some(Err(err)) => {
                        log::error!("{}", err);
                        break None;
                    },

                    // Client WebSocket stream ended.
                    None => {
                        break None;
                    }
                }
            }

            // Message from server.
            Some(msg) = msg_rx => {
                match msg {
                    WsMessage::Connected(msg) => {
                        session.text(serde_json::json!({ "Connected": msg }).to_string()).await.unwrap();
                    },
                    WsMessage::Disconnected(msg) => {
                        session.text(serde_json::json!({ "Disconnected": msg }).to_string()).await.unwrap();
                    },
                    WsMessage::Step(_) => log::error!("Unexpected message received from server."),
                    WsMessage::GameState(state_response_message) => {
                        let json = serde_json::json!(state_response_message);
                        session.text(json.to_string()).await.unwrap();
                    },
                    WsMessage::InternalServerError => {
                        session.text(serde_json::json!({ "error": "Internal Server Error" }).to_string()).await.unwrap();
                    },
                };
            }
        }
    };

    game_server.disconnect(conn_id).await;
    let _ = session.close(close_reason).await;
}

async fn process_message(
    game_server_handle: &GameSessionManagerHandle,
    session: &mut actix_ws::Session,
    message: WsMessage,
    conn_id: &ConnectionId,
    player_id: &PlayerId,
    table_id: &TableId,
) {
    match message {
        WsMessage::Step(action) => {
            let new_game_state = game_server_handle.step(table_id, conn_id, &action).await;
            match new_game_state {
                Ok(new_game_state) => {
                    game_server_handle
                        .broadcast_message(conn_id, WsMessage::GameState(new_game_state))
                        .await;
                }
                Err(_) => {
                    game_server_handle
                        .broadcast_message(conn_id, WsMessage::InternalServerError)
                        .await;
                }
            }
        }
        _ => {
            log::warn!("Unexpected message type received.");
            session
                .text(
                    serde_json::json!({ "error": "Unexpected message type received." }).to_string(),
                )
                .await
                .unwrap();
        }
    }
}
