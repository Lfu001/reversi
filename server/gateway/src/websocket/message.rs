use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WsMessageType {
    Step,
    GameState,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WsMessage {
    message_type: WsMessageType,
    data: String,
}

impl WsMessage {
    /// Creates a new message.
    pub fn new(message_type: WsMessageType, data: String) -> Self {
        Self { message_type, data }
    }

    /// Returns reference to the message type.
    pub fn message_type(&self) -> &WsMessageType {
        &self.message_type
    }

    /// Returns reference to the message data.
    pub fn data(&self) -> &str {
        &self.data
    }
}
