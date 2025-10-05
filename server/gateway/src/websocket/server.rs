use crate::{
    authentication::validate_jwt,
    services::redis_client::RedisClient,
    types::{ConnectionId, PlayerId, PlayerProfile, TableId, TableState},
    websocket::{
        message::WsMessage,
        suggestion::{InvocationRequest, ModelResponse, SuggestionRequest, SuggestionResponse},
    },
};
use common::{Action, DiskColor, PutConfig, StateResponseMessage, StepRequestMessage};
use fastrand;
use jwt_simple::prelude::HS256Key;
use std::{collections::HashMap, env, io, sync::Arc};
use tokio::sync::{mpsc, oneshot, Mutex};
use url::Url;

/// A command received by the game session manager.
enum Command {
    /// Establish a new connection.
    Connect {
        connection_tx: mpsc::UnboundedSender<WsMessage>,
        response_tx: oneshot::Sender<ConnectionId>,
    },
    /// Disconnect a connection.
    Disconnect {
        connection_id: ConnectionId,
        notify_table: bool,
    },
    /// Authenticate a player.
    Authenticate {
        token: String,
        response_tx: oneshot::Sender<Result<PlayerId, String>>,
    },
    /// Check if a connection is authenticated.
    CheckAuthenticated {
        connection_id: ConnectionId,
        response_tx: oneshot::Sender<bool>,
    },
    /// Join a table.
    Join {
        connection_id: ConnectionId,
        table_id: TableId,
        player_id: PlayerId,
        response_tx: oneshot::Sender<()>,
    },
    /// Broadcast a message to all connections in a table.
    BroadcastMessage {
        message: WsMessage,
        connection_id: ConnectionId,
        response_tx: oneshot::Sender<()>,
    },
    /// Send a message to a specific connection.
    SendMessage {
        message: WsMessage,
        connection_id: ConnectionId,
        response_tx: oneshot::Sender<()>,
    },
    /// Fetch the initial game state.
    FetchInitialState {
        table_id: TableId,
        response_tx: oneshot::Sender<Result<StateResponseMessage, String>>,
    },
    /// Step game state.
    Step {
        connection_id: ConnectionId,
        table_id: TableId,
        action: Action,
        response_tx: oneshot::Sender<Result<StateResponseMessage, String>>,
    },
    /// Request a placement suggestion from a model.
    SuggestPlacement {
        connection_id: ConnectionId,
        request: SuggestionRequest,
        table_id: TableId,
        response_tx: oneshot::Sender<Result<SuggestionResponse, String>>,
    },
}

/// Manages game sessions, which include multiple WebSocket connections,
/// and handles communication between them.
pub struct GameSessionManager {
    /// A map of WebSocket connections to their senders.
    sessions: HashMap<ConnectionId, mpsc::UnboundedSender<WsMessage>>,
    /// A map of tables to their connections.
    tables: HashMap<TableId, Vec<ConnectionId>>,
    /// A map of connection ID to Player ID. Only authenticated connections are stored.
    connection_player_map: HashMap<ConnectionId, PlayerId>,
    /// A command receiver.
    command_rx: mpsc::UnboundedReceiver<Command>,
    /// A Redis client.
    redis_client: Arc<Mutex<dyn RedisClient + Send>>,
    /// A JWT key.
    jwt_key: HS256Key,
}

impl GameSessionManager {
    /// Creates a new game session manager and its handle.
    ///
    /// # Arguments
    ///
    /// * `redis_client` - A Redis client instance.
    /// * `jwt_key` - A JWT key.
    pub fn new(
        redis_client: impl RedisClient + 'static,
        jwt_key: HS256Key,
    ) -> (Self, GameSessionManagerHandle) {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        (
            Self {
                sessions: HashMap::new(),
                tables: HashMap::new(),
                connection_player_map: HashMap::new(),
                command_rx,
                redis_client: Arc::new(Mutex::new(redis_client)),
                jwt_key,
            },
            GameSessionManagerHandle { command_tx },
        )
    }

    /// Register new session and assign connection ID to this session.
    ///
    /// # Arguments
    ///
    /// * `tx` - A message sender.
    async fn connect(&mut self, tx: mpsc::UnboundedSender<WsMessage>) -> ConnectionId {
        let id = ConnectionId::new();
        self.sessions.insert(id.clone(), tx);
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
                conn_ids.retain(|id| id != conn_id);
            }
        }
        self.connection_player_map.remove(conn_id);
    }

    /// Authenticate a player.
    ///
    /// # Arguments
    ///
    /// * `token` - The JWT token.
    async fn authenticate(&self, token: &str) -> Result<PlayerId, String> {
        let player_id = match validate_jwt(&self.jwt_key, token) {
            Ok(player_id) => player_id,
            Err(err) => {
                return Err(err.to_string());
            }
        };
        Ok(player_id)
    }

    /// Check if a connection is authenticated.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID.
    async fn is_authenticated(&self, conn_id: &ConnectionId) -> bool {
        self.connection_player_map.contains_key(conn_id)
    }

    /// Join a table.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the player who is joining.
    /// * `table_id` - The table ID of the table to join.
    /// * `player_id` - The player ID of the player who is joining.
    async fn join(&mut self, conn_id: &ConnectionId, table_id: &TableId, player_id: &PlayerId) {
        self.tables
            .entry(table_id.to_owned())
            .or_default()
            .push(conn_id.clone());
        self.connection_player_map
            .insert(conn_id.to_owned(), player_id.to_owned());
    }

    /// Broadcast a message to all clients in the table.
    ///
    /// # Arguments
    ///
    /// * `table_id` - The table ID of the sender.
    /// * `message` - The message to broadcast.
    async fn broadcast_message(&self, table_id: &TableId, message: WsMessage) {
        match self.tables.get(table_id) {
            Some(conn_ids) => {
                for conn in conn_ids {
                    self.send_message(conn, message.clone()).await;
                }
            }
            None => {
                log::warn!(
                    "Tried to broadcast message but the table {:?} does not exist.",
                    table_id
                );
            }
        }
    }

    /// Send a message to a specific connection.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the receiver.
    /// * `message` - The message to send.
    async fn send_message(&self, conn_id: &ConnectionId, message: WsMessage) {
        if let Some(tx) = self.sessions.get(conn_id) {
            let _ = tx.send(message);
        }
    }

    /// Fetch the initial game state.
    async fn fetch_initial_state(
        &mut self,
        table_id: &TableId,
    ) -> Result<StateResponseMessage, String> {
        // Fetch the initial game state from the game API.
        let url = env::var("NEW_GAME_API_URL").unwrap_or("http://localhost:8080/new".to_string());
        let res = match ureq::get(&url).call() {
            Ok(res) => res,
            Err(ureq::Error::Status(_, response)) => {
                return Err(response.into_string().unwrap_or(String::from("")));
            }
            Err(ureq::Error::Transport(transport)) => {
                return Err(String::from(transport.message().unwrap_or("")));
            }
        };
        let new_game_state = res.into_string().unwrap();
        let mut new_game_state: StateResponseMessage =
            serde_json::from_str(&new_game_state).unwrap();

        // Assign disk colors to players.
        let connection_ids = self.tables.get(table_id).unwrap();
        let player_ids = connection_ids
            .iter()
            .map(|conn| self.connection_player_map.get(conn).unwrap().to_owned())
            .collect::<Vec<PlayerId>>();
        let mut rng = fastrand::Rng::new();
        let mut colors = vec![DiskColor::Dark, DiskColor::Light];
        rng.shuffle(&mut colors);
        new_game_state.set_player_colors(colors.clone());
        let roles = HashMap::from_iter(player_ids.into_iter().zip(colors));

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

    /// Step game state and return the new one.
    ///
    /// # Arguments
    ///
    /// * `connection_id` - An ID of the connection to step.
    /// * `table_id` - An ID of the table to step.
    /// * `action` - An action to step.
    async fn step(
        &mut self,
        connection_id: &ConnectionId,
        table_id: &TableId,
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
        let player_id = self.connection_player_map.get(connection_id).unwrap();
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

    /// Request a placement suggestion from a model.
    ///
    /// This function will send the current game state to a model, and receive a list of suggested positions.
    ///
    /// # Arguments
    ///
    /// * `connection_id` - The connection ID of the player who is requesting the suggestion.
    /// * `request` - A suggestion request which contains the request ID and the model to use.
    /// * `table_id` - The table ID to get the game state from.
    async fn suggest_placement(
        &self,
        connection_id: &ConnectionId,
        request: &SuggestionRequest,
        table_id: &TableId,
    ) -> Result<SuggestionResponse, String> {
        let game_state = self.fetch_game_state(table_id).await;

        // Get player ID from connection
        let player_id = self
            .connection_player_map
            .get(connection_id)
            .ok_or_else(|| "Player not authenticated".to_string())?;

        // Get player's color from game state
        let player_color = game_state
            .roles()
            .iter()
            .find(|(id, _)| *id == player_id)
            .map(|(_, color)| *color)
            .ok_or_else(|| "Player not found in this game".to_string())?;

        // Check if it's the player's turn
        if game_state.table().turn() != player_color {
            return Err("It's not your turn".to_string());
        }

        let base_url =
            env::var("SUGGESTION_API_URL").unwrap_or("http://proxy:80/models".to_string());
        let mut url = Url::parse(&base_url).map_err(|e| e.to_string())?;
        url.path_segments_mut()
            .map_err(|_| "Cannot be a base URL".to_string())?
            .push(&request.model)
            .push("suggest");

        let data = InvocationRequest {
            board: game_state.table().board().clone(),
            turn: game_state.table().turn(),
            puttable_positions: game_state.puttable_positions().clone(),
        };
        let res = match ureq::post(url.as_str()).send_json(&data) {
            Ok(res) => res,
            Err(ureq::Error::Status(_, response)) => {
                return Err(response.into_string().unwrap_or(String::from("")));
            }
            Err(ureq::Error::Transport(transport)) => {
                return Err(String::from(transport.message().unwrap_or("")));
            }
        };

        let model_response: Result<ModelResponse, _> = res.into_json();
        match model_response {
            Ok(model_response) => Ok(SuggestionResponse {
                request_id: request.request_id.clone(),
                positions: model_response.positions,
            }),
            Err(e) => Err(format!("Failed to parse model response: {}", e)),
        }
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

    /// Read the player profiles from Redis.
    ///
    /// # Arguments
    ///
    /// * `table_id` - An ID of the table to read.
    async fn read_player_profiles(&self, table_id: &TableId) -> Vec<PlayerProfile> {
        let table = self.tables.get(table_id).unwrap();
        let player_ids = table
            .iter()
            .map(|conn| self.connection_player_map.get(conn).unwrap().to_owned())
            .collect::<Vec<PlayerId>>();
        let mut player_profiles = Vec::new();
        for player_id in player_ids {
            let json = self
                .redis_client
                .lock()
                .await
                .json_get(&player_id, ".")
                .await
                .unwrap();
            player_profiles.push(serde_json::from_value(json).unwrap());
        }
        player_profiles
    }

    /// Start the game session manager.
    pub async fn run(mut self) -> io::Result<()> {
        while let Some(command) = self.command_rx.recv().await {
            match command {
                Command::Connect {
                    connection_tx,
                    response_tx,
                } => {
                    let conn_id = self.connect(connection_tx).await;
                    let _ = response_tx.send(conn_id);
                }
                Command::Disconnect {
                    connection_id,
                    notify_table,
                } => {
                    if notify_table {
                        // Broadcast to players in the table that the player has left.
                        let table_id = self
                            .tables
                            .iter()
                            .find(|(_, participants)| participants.contains(&connection_id))
                            .map(|(table_id, _)| table_id.clone());

                        self.disconnect(&connection_id).await;

                        match table_id {
                            Some(table_id) => {
                                let player_profiles = self.read_player_profiles(&table_id).await;
                                let message = WsMessage::Players(player_profiles);
                                self.broadcast_message(&table_id, message).await;
                            }
                            None => {
                                log::warn!(
                                    "Failed to broadcast message: connection is not in any table."
                                );
                            }
                        }
                    } else {
                        self.disconnect(&connection_id).await;
                    }
                }
                Command::Authenticate { token, response_tx } => {
                    let player_id = self.authenticate(&token).await;
                    let _ = response_tx.send(player_id);
                }
                Command::CheckAuthenticated {
                    connection_id,
                    response_tx,
                } => {
                    let is_authenticated = self.is_authenticated(&connection_id).await;
                    let _ = response_tx.send(is_authenticated);
                }
                Command::Join {
                    connection_id,
                    table_id,
                    player_id,
                    response_tx,
                } => {
                    self.join(&connection_id, &table_id, &player_id).await;

                    // Broadcast to players in the table that a new player has joined.
                    let player_profiles = self.read_player_profiles(&table_id).await;
                    let message = WsMessage::Players(player_profiles);
                    self.broadcast_message(&table_id, message).await;

                    let _ = response_tx.send(());
                }
                Command::BroadcastMessage {
                    message,
                    connection_id,
                    response_tx,
                } => {
                    let table = self
                        .tables
                        .iter()
                        .find(|(_, participants)| participants.contains(&connection_id));
                    match table {
                        Some((table_id, _)) => {
                            self.broadcast_message(table_id, message).await;
                        }
                        None => {
                            log::warn!(
                                "Failed to broadcast message: connection is not in any table."
                            );
                        }
                    }
                    let _ = response_tx.send(());
                }
                Command::SendMessage {
                    message,
                    connection_id: conn_id,
                    response_tx,
                } => {
                    self.send_message(&conn_id, message).await;
                    let _ = response_tx.send(());
                }
                Command::FetchInitialState {
                    table_id,
                    response_tx,
                } => {
                    let game_state = self.fetch_initial_state(&table_id).await;
                    let _ = response_tx.send(game_state);
                }
                Command::Step {
                    connection_id,
                    table_id,
                    action,
                    response_tx,
                } => {
                    let new_game_state = self.step(&connection_id, &table_id, &action).await;
                    let _ = response_tx.send(new_game_state);
                }
                Command::SuggestPlacement {
                    connection_id,
                    request,
                    table_id,
                    response_tx,
                } => {
                    let result = self
                        .suggest_placement(&connection_id, &request, &table_id)
                        .await;
                    let _ = response_tx.send(result);
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
    pub async fn connect(&self, conn_tx: mpsc::UnboundedSender<WsMessage>) -> ConnectionId {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Connect {
                connection_tx: conn_tx,
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }

    /// Unregister message sender and optionally broadcast disconnection message to current table.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - A connection ID to unregister
    /// * `notify_table` - Whether to notify the table of the player's disconnection
    pub async fn disconnect(&self, conn_id: &ConnectionId, notify_table: bool) {
        self.command_tx
            .send(Command::Disconnect {
                connection_id: conn_id.to_owned(),
                notify_table,
            })
            .unwrap();
    }

    /// Authenticate a player.
    ///
    /// # Arguments
    ///
    /// * `token` - A JWT token.
    ///
    /// # Returns
    ///
    /// A player ID if authentication is successful, or an error message otherwise.
    pub async fn authenticate(&self, token: &str) -> Result<PlayerId, String> {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Authenticate {
                token: token.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }

    /// Checks if a connection is authenticated.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID to check.
    ///
    /// # Returns
    ///
    /// `true` if the connection is authenticated, `false` otherwise.
    pub async fn is_authenticated(&self, conn_id: &ConnectionId) -> bool {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::CheckAuthenticated {
                connection_id: conn_id.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }

    /// Join a table.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the sender.
    /// * `table_id` - The table ID.
    /// * `player_id` - The player ID.
    pub async fn join(&self, conn_id: &ConnectionId, table_id: &TableId, player_id: &PlayerId) {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Join {
                connection_id: conn_id.to_owned(),
                table_id: table_id.to_owned(),
                player_id: player_id.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap();
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
            .send(Command::BroadcastMessage {
                message,
                connection_id: conn_id.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap();
    }

    /// Send a message to a specific connection.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the receiver.
    /// * `message` - The message to send.
    pub async fn send_message(&self, conn_id: &ConnectionId, message: WsMessage) {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::SendMessage {
                connection_id: conn_id.to_owned(),
                message,
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap();
    }

    /// Fetch the initial game state.
    ///
    /// # Arguments
    ///
    /// * `table_id` - An ID of the table to fetch.
    pub async fn fetch_initial_state(
        &self,
        table_id: &TableId,
    ) -> Result<StateResponseMessage, String> {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::FetchInitialState {
                table_id: table_id.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }

    /// Step game state.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the sender.
    /// * `table_id` - The table ID.
    /// * `action` - The action to step.
    ///
    /// # Returns
    ///
    /// The new game state if stepping is successful, an error message otherwise.
    pub async fn step(
        &self,
        conn_id: &ConnectionId,
        table_id: &TableId,
        action: &Action,
    ) -> Result<StateResponseMessage, String> {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::Step {
                connection_id: conn_id.to_owned(),
                table_id: table_id.to_owned(),
                action: action.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }

    /// Request a placement suggestion from a model.
    ///
    /// # Arguments
    ///
    /// * `conn_id` - The connection ID of the player making the request.
    /// * `request` - The suggestion request.
    /// * `table_id` - The table ID.
    pub async fn suggest_placement(
        &self,
        conn_id: &ConnectionId,
        request: &SuggestionRequest,
        table_id: &TableId,
    ) -> Result<SuggestionResponse, String> {
        let (res_tx, res_rx) = oneshot::channel();

        self.command_tx
            .send(Command::SuggestPlacement {
                connection_id: conn_id.to_owned(),
                request: request.clone(),
                table_id: table_id.to_owned(),
                response_tx: res_tx,
            })
            .unwrap();

        res_rx.await.unwrap()
    }
}
