use crate::{
    game_logic::{action::get_puttable_positions, state::Table},
    server::messages::response::ResponseMessage,
};
use actix_web::Responder;

/// A handler for creating new table.
///
/// # Returns
///
/// A new table and puttable positions.
pub async fn create_new_table() -> impl Responder {
    let table = Table::new();
    let puttable_positions = get_puttable_positions(table.board(), table.turn());

    ResponseMessage {
        table,
        puttable_positions,
        judge_result: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App};

    /// Test for creating new table
    #[actix_web::test]
    async fn test_create_new_table() {
        let app =
            test::init_service(App::new().route("/new", web::get().to(create_new_table))).await;
        let req = test::TestRequest::get().uri("/new").to_request();
        let resp = test::call_service(&app, req).await;

        assert!(resp.status().is_success());
    }
}
