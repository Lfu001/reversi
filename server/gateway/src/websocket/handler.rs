use crate::{
    types::{ConnectionId, TableId},
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
const CLIENT_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long before lack of client authentication causes a timeout.
const CLIENT_UNAUTHENTICATED_TIMEOUT: Duration = Duration::from_secs(5);

/// Handles WebSocket communication for a game table.
///
/// # Arguments
///
/// * `table_id` - The table ID.
/// * `game_server` - The game server.
/// * `session` - The WebSocket session.
/// * `stream` - The WebSocket message stream.
pub async fn game_ws(
    table_id: TableId,
    game_server: GameSessionManagerHandle,
    mut session: actix_ws::Session,
    stream: actix_ws::MessageStream,
) {
    let mut last_heartbeat = Instant::now();
    let connected_at = Instant::now();
    let mut interval = interval(HEARTBEAT_INTERVAL);

    let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
    let conn_id = game_server.connect(conn_tx).await;
    let mut stream = pin!(stream.aggregate_continuations());

    let close_reason = loop {
        let tick = pin!(interval.tick());
        let messages = pin!(stream.next());
        let msg_rx = pin!(conn_rx.recv());

        tokio::select! {
            // Heartbeat internal tick.
            _ = tick => {
                if Instant::now().duration_since(connected_at) > CLIENT_UNAUTHENTICATED_TIMEOUT && !game_server.is_authenticated(&conn_id).await {
                    // Disconnect immediately if the connection is not authenticated yet.
                    log::info!("Client is not authenticated. Closing connection.");
                    game_server.disconnect(&conn_id, false).await;
                    break None;
                }
                if Instant::now().duration_since(last_heartbeat) > CLIENT_HEARTBEAT_TIMEOUT {
                    log::info!("Client has not sent heartbeat in over {:?} seconds. Closing connection.", CLIENT_HEARTBEAT_TIMEOUT);
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
                        process_message(&game_server, &mut session, message, &conn_id, &table_id).await;
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
                    WsMessage::Players(msg) => {
                        session.text(serde_json::json!({ "Players": msg }).to_string()).await.unwrap();
                    },
                    WsMessage::GameState(state_response_message) => {
                        let json = serde_json::json!({"GameState": state_response_message});
                        session.text(json.to_string()).await.unwrap();
                    },
                    WsMessage::SuggestionResponse(suggestion_response_message) => {
                        let json = serde_json::json!({"SuggestionResponse": suggestion_response_message});
                        session.text(json.to_string()).await.unwrap();
                    },
                    WsMessage::InternalServerError(err) => {
                        session.text(serde_json::json!({ "InternalServerError": err }).to_string()).await.unwrap();
                    },
                    _ => {
                        log::error!("Unexpected message received from server.");
                    }
                };
            }
        }
    };

    game_server.disconnect(&conn_id, true).await;
    let _ = session.close(close_reason).await;
}

/// Process a text message from a client.
///
/// This function will handle text messages from a client by performing the required action
/// and sending a response back to the client.
///
/// # Arguments
///
/// * `game_server_handle`: A reference to the game server manager handle.
/// * `session`: A reference to the actix websocket session.
/// * `message`: The message received from the client.
/// * `conn_id`: The connection id of the client that sent the message.
/// * `table_id`: The table id associated with the client that sent the message.
async fn process_message(
    game_server_handle: &GameSessionManagerHandle,
    session: &mut actix_ws::Session,
    message: WsMessage,
    conn_id: &ConnectionId,
    table_id: &TableId,
) {
    if !matches!(message, WsMessage::Authenticate(_))
        && !game_server_handle.is_authenticated(conn_id).await
    {
        // Disconnect immediately if the message is not authentication request and the connection is not authenticated yet.
        log::info!("Client is not authenticated. Closing connection.");
        game_server_handle.disconnect(conn_id, false).await;
        return;
    }

    match message {
        WsMessage::Step(action) => {
            let new_game_state = game_server_handle.step(conn_id, table_id, &action).await;
            match new_game_state {
                Ok(new_game_state) => {
                    game_server_handle
                        .broadcast_message(conn_id, WsMessage::GameState(new_game_state))
                        .await;
                }
                Err(err) => {
                    log::error!("Failed to step game state: {}", err);
                    game_server_handle
                        .broadcast_message(conn_id, WsMessage::InternalServerError(err))
                        .await;
                }
            }
        }
        WsMessage::Authenticate(token) => match game_server_handle.authenticate(&token).await {
            Ok(player_id) => {
                game_server_handle.join(conn_id, table_id, &player_id).await;
            }
            Err(_) => {
                // Disconnect immediately if authentication fails.
                log::info!("Authentication failed. Closing connection.");
                game_server_handle.disconnect(conn_id, false).await;
            }
        },
        WsMessage::Start => {
            let initial_game_state = game_server_handle.fetch_initial_state(table_id).await;
            match initial_game_state {
                Ok(initial_game_state) => {
                    game_server_handle
                        .broadcast_message(conn_id, WsMessage::GameState(initial_game_state))
                        .await;
                }
                Err(err) => {
                    log::error!("Failed to fetch initial game state: {}", err);
                    game_server_handle
                        .broadcast_message(conn_id, WsMessage::InternalServerError(err))
                        .await;
                }
            }
        }
        WsMessage::SuggestionRequest(req) => {
            let message = match game_server_handle
                .suggest_placement(conn_id, &req, table_id)
                .await
            {
                Ok(response) => WsMessage::SuggestionResponse(response),
                Err(err) => {
                    log::error!("Failed to suggest placement: {}", err);
                    WsMessage::InternalServerError(err)
                }
            };
            game_server_handle.send_message(conn_id, message).await;
        }
        _ => {
            log::error!("Unexpected message type received.");
            session
                .text(
                    serde_json::json!({ "error": "Unexpected message type received." }).to_string(),
                )
                .await
                .unwrap();
        }
    }
}
