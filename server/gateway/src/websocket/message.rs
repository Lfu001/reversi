use common::{Action, StateResponseMessage};
use serde::{Deserialize, Serialize};

/// A message that is sent over the WebSocket connection.
#[derive(Serialize, Deserialize, Clone)]
pub enum WsMessage {
    /// **(Client <= Server)** Another player joins the table.
    Connected(String),
    /// **(Client <= Server)** Another player leaves the table.
    Disconnected(String),
    /// **(Client => Server)** The player makes some reversi action.
    Step(Action),
    /// **(Client <= Server)** The game state is updated.
    GameState(StateResponseMessage),
    /// **(Client <= Server)** Internal server error occurred.
    InternalServerError,
}
