use crate::{
    types::PlayerProfile,
    websocket::suggestion::{SuggestionRequest, SuggestionResponse},
};
use common::{Action, StateResponseMessage};
use serde::{Deserialize, Serialize};

/// A message that is sent over the WebSocket connection.
#[derive(Serialize, Deserialize, Clone)]
pub enum WsMessage {
    /// **(Client <= Server)** The server sends the list of players in the table.
    Players(Vec<PlayerProfile>),
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
    /// **(Client => Server)** The player requests a suggestion.
    SuggestionRequest(SuggestionRequest),
    /// **(Client <= Server)** The server sends a suggestion.
    SuggestionResponse(SuggestionResponse),
}
