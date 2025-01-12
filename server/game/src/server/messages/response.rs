use crate::game_logic::state::{JudgeResult, Position, Table};
use actix_web::{body::BoxBody, http::header::ContentType, HttpRequest, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

/// A response message from the game endpoint.
#[derive(Serialize, Deserialize)]
pub struct ResponseMessage {
    pub(in crate::server) table: Table,
    pub(in crate::server) puttable_positions: Vec<Position>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::server) judge_result: Option<JudgeResult>,
}

impl Responder for ResponseMessage {
    type Body = BoxBody;

    fn respond_to(self, _req: &HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}
