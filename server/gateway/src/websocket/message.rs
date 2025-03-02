use common::{Action, StateResponseMessage};
use serde::{Deserialize, Serialize};

/// A message that is sent over the WebSocket connection.
#[derive(Serialize, Deserialize, Clone)]
pub enum WsMessage {
    /// A message sent to a client when another player joins the table.
    Connected(String),
    /// A message sent to a client when another player leaves the table.
    Disconnected(String),
    /// A message sent from a client to the server to make a move.
    Step(Action),
    /// A message sent from the server to all clients in the table
    /// to update their game state.
    GameState(StateResponseMessage),
    /// A message sent from the server to a client to signal an internal
    /// server error.
    InternalServerError,
}
