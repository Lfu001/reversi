use common::{Action, StateResponseMessage};
use serde::{Deserialize, Serialize};

/// A message that is sent over the WebSocket connection.
#[derive(Serialize, Deserialize, Clone)]
pub enum WsMessage {
    /// **(Client <= Server)** The server sends the list of players in the table.
    Players(Vec<String>),
    /// **(Client => Server)** The player requests authentication.
    Authenticate(String),
    /// **(Client => Server)** The player is ready to start the game.
    Start,
    /// **(Client => Server)** The player makes some reversi action.
    Step(Action),
    /// **(Client <= Server)** The game state is updated.
    GameState(StateResponseMessage),
    /// **(Client <= Server)** Internal server error occurred.
    InternalServerError(String),
}
