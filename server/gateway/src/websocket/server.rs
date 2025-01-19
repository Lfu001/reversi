use super::message::{WsMessage, WsMessageType};
use std::{
    collections::{HashMap, HashSet},
    io,
};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

pub type ConnectionId = Uuid;
pub type TableId = String;

/// A command received by the game session manager.
pub enum Command {
    /// Establish a new connection.
    Connect {
        connection_tx: mpsc::UnboundedSender<WsMessage>,
        response_tx: oneshot::Sender<ConnectionId>,
        table_id: TableId,
    },
    /// Disconnect a connection.
    Disconnect(ConnectionId),
    /// Send a message.
    Message {
        message: WsMessage,
        connection_id: ConnectionId,
        response_tx: oneshot::Sender<()>,
    },
    /// Step game state.
    Step {
        table_id: TableId,
        connection_id: ConnectionId,
        data: String,
        response_tx: oneshot::Sender<String>,
    },
}

/// Manages game sessions, which include multiple WebSocket connections,
/// and handles communication between them.
pub struct GameSessionManager {
    /// A map of WebSocket connections to their senders.
    sessions: HashMap<ConnectionId, mpsc::UnboundedSender<WsMessage>>,
    /// A map of tables to their connections.
    tables: HashMap<TableId, HashSet<ConnectionId>>,
    /// A command receiver.
    command_rx: mpsc::UnboundedReceiver<Command>,
}

impl GameSessionManager {
    /// Creates a new game session manager and its handle.
    pub fn new() -> (Self, GameSessionManagerHandle) {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        (
            Self {
                sessions: HashMap::new(),
                tables: HashMap::new(),
                command_rx,
            },
            GameSessionManagerHandle { command_tx },
        )
    }

    /// Register new session and assign connection ID to this session.
    async fn connect(
        &mut self,
        tx: mpsc::UnboundedSender<WsMessage>,
        table_id: &TableId,
    ) -> ConnectionId {
        let id = Uuid::new_v4();
        self.sessions.insert(id, tx);
        self.tables
            .entry(table_id.to_owned())
            .or_default()
            .insert(id);
        id
    }

    /// Unregister connection and broadcast disconnection message.
    async fn disconnect(&mut self, conn_id: ConnectionId) {
        // Remove sender.
        if self.sessions.remove(&conn_id).is_some() {
            for conn_ids in self.tables.values_mut() {
                conn_ids.remove(&conn_id);
            }
        }

        // Broadcast disconnection message.
        todo!("set appropriate message");
        self.broadcast_message(
            conn_id,
            WsMessage::new(WsMessageType::GameState, String::from("")),
        )
        .await;
    }

    /// Broadcast a message to all clients in the table except the given connection ID.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the sender.
    /// * `message` - The message to broadcast.
    async fn broadcast_message(&self, conn_id: ConnectionId, message: WsMessage) {
        // Find table where the connection ID participates in.
        if let Some((_, conn_ids)) = self
            .tables
            .iter()
            .find(|(_, participants)| participants.contains(&conn_id))
        {
            for conn in conn_ids {
                if let Some(tx) = self.sessions.get(conn) {
                    let _ = tx.send(message.clone());
                }
            }
        }
    }

    /// Step game state and return the new one.
    async fn step(&mut self, table_id: &str, conn_id: &ConnectionId, data: &str) -> String {
        // Fetch current game state from Redis.
        todo!();

        // Call game API.
        // TODO: use environment variable for API URL
        let res = ureq::post("http://localhost:8000/step")
            .send_form(&[("table", table_id), ("data", data)]);
        if res.is_err() {
            log::error!("{}", res.unwrap_err());
        }
        let res = res.unwrap();
        let new_game_state = res.into_string().unwrap();

        // Update game state in Redis.

        // Return game state
        new_game_state
    }

    /// Start the game session manager.
    pub async fn run(mut self) -> io::Result<()> {
        while let Some(command) = self.command_rx.recv().await {
            match command {
                Command::Connect {
                    connection_tx,
                    response_tx,
                    table_id,
                } => {
                    let conn_id = self.connect(connection_tx, &table_id).await;
                    todo!("set appropriate message");
                    self.broadcast_message(
                        conn_id,
                        WsMessage::new(WsMessageType::GameState, String::from("")),
                    )
                    .await;
                    let _ = response_tx.send(conn_id);
                }
                Command::Disconnect(conn_id) => self.disconnect(conn_id).await,
                Command::Message {
                    message,
                    connection_id,
                    response_tx,
                } => {
                    self.broadcast_message(connection_id, message).await;
                    let _ = response_tx.send(());
                }
                Command::Step {
                    table_id,
                    connection_id,
                    data,
                    response_tx,
                } => {
                    let new_game_state = self.step(&table_id, &connection_id, &data).await;
                    let _ = response_tx.send(new_game_state);
                }
            }
        }

        Ok(())
    }
}

/// Handle and command sender for the game session manager.
#[derive(Clone)]
pub struct GameSessionManagerHandle {
    /// The command sender.
    command_tx: mpsc::UnboundedSender<Command>,
}

impl GameSessionManagerHandle {
    /// Register client message sender and generate a connection ID.
    pub async fn connect(
        &self,
        conn_tx: mpsc::UnboundedSender<WsMessage>,
        table_id: &str,
    ) -> ConnectionId {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Connect {
                connection_tx: conn_tx,
                response_tx: res_tx,
                table_id: table_id.to_owned(),
            })
            .unwrap();

        res_rx.await.unwrap()
    }

    /// Unregister message sender and broadcast disconnection message to current table.
    pub async fn disconnect(&self, conn_id: ConnectionId) {
        self.command_tx.send(Command::Disconnect(conn_id)).unwrap();
    }

    /// Broadcast a message to all clients in the table.
    pub async fn broadcast_message(&self, conn_id: ConnectionId, message: WsMessage) {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Message {
                message,
                connection_id: conn_id,
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap();
    }

    /// Step game state.
    pub async fn step(&self, table_id: &str, conn_id: ConnectionId, data: &str) -> String {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Step {
                table_id: table_id.to_owned(),
                connection_id: conn_id,
                data: data.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }
}
