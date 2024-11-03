use crate::game_logic::{action::Action, state::Table};
use serde::{Deserialize, Serialize};

/// A request message to the game endpoint.
#[derive(Serialize, Deserialize)]
pub struct RequestMessage {
    pub(in crate::server) table: Table,
    pub(in crate::server) action: Action,
}
