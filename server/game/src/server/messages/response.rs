use actix_web::{body::BoxBody, http::header::ContentType, HttpRequest, HttpResponse, Responder};
use common::{JudgeResult, Position, StateResponseMessage, Table};

/// A wrapper for a [`StateResponseMessage`] that implements [`Responder`].
pub struct StateResponseMessageExt(StateResponseMessage);

impl StateResponseMessageExt {
    /// Creates a new [`StateResponseMessageExt`].
    pub fn new(
        table: Table,
        puttable_positions: Vec<Position>,
        judge_result: Option<JudgeResult>,
    ) -> Self {
        Self(StateResponseMessage::new(
            table,
            puttable_positions,
            judge_result,
        ))
    }
}

impl Responder for StateResponseMessageExt {
    type Body = BoxBody;

    fn respond_to(self, _: &HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self.0).unwrap();

        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}
