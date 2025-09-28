use common::Position;
use serde::{Deserialize, Serialize};

/// A request for a suggestion from a model.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SuggestionRequest {
    /// The name of the model to use.
    pub model: String,
    /// A unique ID for the request.
    pub request_id: String,
}

/// A response with a list of suggested positions from a model.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SuggestionResponse {
    /// A unique ID for the request. Same value as the corresponding [`SuggestionRequest::request_id`].
    pub request_id: String,
    /// A list of suggested positions.
    pub positions: Vec<SuggestedPosition>,
}

/// A single suggested position with its confidence score.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SuggestedPosition {
    /// The suggested position.
    pub position: Position,
    /// The confidence score of the suggestion.
    pub confidence: f64,
}

/// A response from a model with a list of suggested positions.
#[derive(Deserialize)]
pub struct ModelResponse {
    /// A list of suggested positions.
    pub positions: Vec<SuggestedPosition>,
}
