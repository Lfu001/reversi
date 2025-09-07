use super::message::WsMessage;
use crate::{
    services::redis_client::RedisClient,
    types::{ConnectionId, PlayerId, TableId, TableState},
};
use common::{Action, DiskColor, PutConfig, StateResponseMessage, StepRequestMessage};
use std::{
    collections::{HashMap, HashSet},
    env, io,
    sync::Arc,
};
use tokio::sync::{mpsc, oneshot, Mutex};

/// A command received by the game session manager.
enum Command {
    /// Establish a new connection.
    Connect {
        connection_tx: mpsc::UnboundedSender<WsMessage>,
        response_tx: oneshot::Sender<ConnectionId>,
        table_id: TableId,
        player_id: PlayerId,
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
        player_id: PlayerId,
        action: Action,
        response_tx: oneshot::Sender<Result<StateResponseMessage, String>>,
    },
}

/// Manages game sessions, which include multiple WebSocket connections,
/// and handles communication between them.
pub struct GameSessionManager {
    /// A map of WebSocket connections to their senders.
    sessions: HashMap<ConnectionId, mpsc::UnboundedSender<WsMessage>>,
    /// A map of tables to their connections.
    tables: HashMap<TableId, HashSet<ConnectionId>>,
    /// A map of connection ID to Player ID.
    connection_player_map: HashMap<ConnectionId, PlayerId>,
    /// A command receiver.
    command_rx: mpsc::UnboundedReceiver<Command>,
    /// A Redis client.
    redis_client: Arc<Mutex<dyn RedisClient + Send>>,
}

impl GameSessionManager {
    /// Creates a new game session manager and its handle.
    ///
    /// # Arguments
    ///
    /// * `redis_client` - A Redis client instance.
    pub fn new(redis_client: impl RedisClient + 'static) -> (Self, GameSessionManagerHandle) {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        (
            Self {
                sessions: HashMap::new(),
                tables: HashMap::new(),
                connection_player_map: HashMap::new(),
                command_rx,
                redis_client: Arc::new(Mutex::new(redis_client)),
            },
            GameSessionManagerHandle { command_tx },
        )
    }

    /// Register new session and assign connection ID to this session.
    ///
    /// # Arguments
    ///
    /// * `tx` - A message sender.
    /// * `table_id` - An ID of the table to join.
    /// * `player_id` - A player ID.
    async fn connect(
        &mut self,
        tx: mpsc::UnboundedSender<WsMessage>,
        table_id: &TableId,
        player_id: &PlayerId,
    ) -> ConnectionId {
        let id = ConnectionId::new();
        self.sessions.insert(id.clone(), tx);
        self.tables
            .entry(table_id.to_owned())
            .or_default()
            .insert(id.clone());
        self.connection_player_map
            .insert(id.to_owned(), player_id.to_owned());
        id
    }

    /// Unregister session.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - A connection ID to unregister.
    async fn disconnect(&mut self, conn_id: &ConnectionId) {
        // Remove sender.
        if self.sessions.remove(conn_id).is_some() {
            for conn_ids in self.tables.values_mut() {
                conn_ids.remove(conn_id);
            }
        }
        self.connection_player_map.remove(conn_id);
    }

    /// Broadcast a message to all clients in the table except the given connection ID.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the sender.
    /// * `message` - The message to broadcast.
    async fn broadcast_message(&self, conn_id: &ConnectionId, message: WsMessage) {
        // Find table where the connection ID participates in.
        if let Some((_, conn_ids)) = self
            .tables
            .iter()
            .find(|(_, participants)| participants.contains(conn_id))
        {
            for conn in conn_ids {
                if let Some(tx) = self.sessions.get(conn) {
                    let _ = tx.send(message.clone());
                }
            }
        }
    }

    /// Step game state and return the new one.
    ///
    /// # Arguments
    ///
    /// * `table_id` - An ID of the table to step.
    /// * `player_id` - A player ID of the sender.
    /// * `action` - An action to step.
    async fn step(
        &mut self,
        table_id: &TableId,
        player_id: &PlayerId,
        action: &Action,
    ) -> Result<StateResponseMessage, String> {
        // Fetch current game state from Redis.
        let table_state = self.fetch_game_state(table_id).await;

        // Override the disk color of the player who sent the action with the one stored in Redis.
        let roles = self
            .redis_client
            .lock()
            .await
            .json_get(table_id, ".roles")
            .await
            .unwrap();
        let roles: HashMap<PlayerId, DiskColor> = serde_json::from_value(roles).unwrap();
        let color = roles.get(player_id).unwrap();
        let action = match action {
            Action::PutDisk(config) => Action::PutDisk(PutConfig::new(*color, *config.position())),
            Action::PassTurn(_) => Action::PassTurn(*color),
        };

        // Call game API and get the new game state.
        let param = StepRequestMessage::new(table_state.table().clone(), action.to_owned());
        let url =
            env::var("STEP_GAME_API_URL").unwrap_or(String::from("http://localhost:8080/step"));
        let res = match ureq::post(&url).send_json(&param) {
            Ok(res) => res,
            Err(ureq::Error::Status(_, response)) => {
                return Err(response.into_string().unwrap_or(String::from("")));
            }
            Err(ureq::Error::Transport(transport)) => {
                return Err(String::from(transport.message().unwrap_or("")));
            }
        };
        let new_game_state = res.into_string().unwrap();
        let new_game_state: StateResponseMessage = serde_json::from_str(&new_game_state).unwrap();

        // Update game state in Redis.
        let table_state = TableState::new(
            new_game_state.table().clone(),
            roles,
            new_game_state.puttable_positions().clone(),
            *new_game_state.judge_result(),
        );
        self.update_game_state(table_id, &table_state).await;

        Ok(new_game_state)
    }

    /// Fetch the current game state from Redis.
    ///
    /// # Arguments
    ///
    /// * `table_id` - An ID of the table to fetch.
    async fn fetch_game_state(&self, table_id: &TableId) -> TableState {
        let json = self
            .redis_client
            .lock()
            .await
            .json_get(table_id, ".")
            .await
            .unwrap();
        serde_json::from_value(json).unwrap()
    }

    /// Update the current game state in Redis.
    ///
    /// # Arguments
    ///
    /// * `table_id` - An ID of the table to update.
    /// * `table_state` - A new game state.
    async fn update_game_state(&self, table_id: &TableId, table_state: &TableState) {
        self.redis_client
            .lock()
            .await
            .json_set(table_id, ".", &serde_json::json!(table_state))
            .await
            .unwrap();
    }

    /// Fetch the player name from Redis.
    ///
    /// # Arguments
    ///
    /// * `player_id` - An ID of the player to fetch
    async fn fetch_player_name(&self, player_id: &PlayerId) -> String {
        self.redis_client.lock().await.get(player_id).await.unwrap()
    }

    /// Start the game session manager.
    pub async fn run(mut self) -> io::Result<()> {
        while let Some(command) = self.command_rx.recv().await {
            match command {
                Command::Connect {
                    connection_tx,
                    response_tx,
                    table_id,
                    player_id,
                } => {
                    let conn_id = self.connect(connection_tx, &table_id, &player_id).await;

                    // Broadcast to other players in the table that a new player has joined.
                    let player_name = self.fetch_player_name(&player_id).await;
                    let message = WsMessage::Connected(player_name);
                    self.broadcast_message(&conn_id, message).await;

                    let _ = response_tx.send(conn_id);
                }
                Command::Disconnect(conn_id) => {
                    self.disconnect(&conn_id).await;

                    // Broadcast to other players in the table that a player has left.
                    let player_id = self.connection_player_map.get(&conn_id).unwrap();
                    let player_name = self.fetch_player_name(player_id).await;
                    let message = WsMessage::Disconnected(format!("{} has left", player_name));
                    self.broadcast_message(&conn_id, message).await;
                }
                Command::Message {
                    message,
                    connection_id,
                    response_tx,
                } => {
                    self.broadcast_message(&connection_id, message).await;
                    let _ = response_tx.send(());
                }
                Command::Step {
                    table_id,
                    player_id,
                    action,
                    response_tx,
                } => {
                    let new_game_state = self.step(&table_id, &player_id, &action).await;
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
    ///
    /// # Arguments
    ///
    /// * `conn_tx` - A message sender.
    /// * `table_id` - An ID of the table to join.
    /// * `player_id` - A player ID.
    pub async fn connect(
        &self,
        conn_tx: mpsc::UnboundedSender<WsMessage>,
        table_id: &TableId,
        player_id: &PlayerId,
    ) -> ConnectionId {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Connect {
                connection_tx: conn_tx,
                response_tx: res_tx,
                table_id: table_id.to_owned(),
                player_id: player_id.to_owned(),
            })
            .unwrap();

        res_rx.await.unwrap()
    }

    /// Unregister message sender and broadcast disconnection message to current table.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - A connection ID to unregister
    pub async fn disconnect(&self, conn_id: ConnectionId) {
        self.command_tx.send(Command::Disconnect(conn_id)).unwrap();
    }

    /// Broadcast a message to all clients in the table.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the sender.
    /// * `message` - The message to broadcast
    pub async fn broadcast_message(&self, conn_id: &ConnectionId, message: WsMessage) {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Message {
                message,
                connection_id: conn_id.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap();
    }

    /// Step game state.
    pub async fn step(
        &self,
        table_id: &TableId,
        player_id: &PlayerId,
        action: &Action,
    ) -> Result<StateResponseMessage, String> {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Step {
                table_id: table_id.to_owned(),
                player_id: player_id.to_owned(),
                action: action.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }
}
