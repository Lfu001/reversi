use crate::{
    game_logic::controller::Controller,
    server::messages::{request::RequestMessage, response::ResponseMessage},
};
use actix_web::{error, web, Responder, Result};

/// A handler for step table.
///
/// # Arguments
///
/// * `req` - A request message from the client.
///
/// # Returns
///
/// An updated table, puttable positions, and judge result if the game is over.
pub async fn step_table(req: web::Json<RequestMessage>) -> Result<impl Responder> {
    let RequestMessage { mut table, action } = req.into_inner();
    let step_result = Controller::step(&mut table, action);

    match step_result {
        Ok(step_result) => Ok(ResponseMessage {
            table,
            puttable_positions: step_result.puttable_positions,
            judge_result: step_result.judge_result,
        }),
        Err(()) => Err(error::ErrorBadRequest("Invalid action.")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game_logic::{
            action::{Action, PutConfig},
            state::{Column, DiskColor, Row, Table},
        },
        position,
    };
    use actix_web::{http::StatusCode, test, web, App};

    /// Test for step table by valid action "PutDisk"
    #[actix_web::test]
    async fn test_step_table_put() {
        let app = test::init_service(App::new().route("/step", web::post().to(step_table))).await;
        let req = test::TestRequest::post()
            .uri("/step")
            .set_json(&RequestMessage {
                table: Table::new(),
                action: Action::PutDisk(PutConfig::new(
                    DiskColor::Dark,
                    position!(Row::Three, Column::D),
                )),
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    /// Test for step table by invalid action "PutDisk"
    #[actix_web::test]
    async fn test_step_table_invalid() {
        let app = test::init_service(App::new().route("/step", web::post().to(step_table))).await;
        let req = test::TestRequest::post()
            .uri("/step")
            .set_json(&RequestMessage {
                table: Table::new(),
                action: Action::PutDisk(PutConfig::new(
                    DiskColor::Dark,
                    position!(Row::One, Column::A),
                )),
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
    }

    /// Test for step table by valid action "PassTurn"
    #[actix_web::test]
    async fn test_step_table_pass() {
        let mut table = Table::new();
        let board = table.board_mut();
        board.set_disk(position!(Row::Four, Column::C), DiskColor::Dark);
        board.set_disk(position!(Row::Four, Column::D), DiskColor::Dark);
        board.set_disk(position!(Row::Five, Column::E), DiskColor::Dark);
        board.set_disk(position!(Row::Five, Column::F), DiskColor::Dark);
        board.set_disk(position!(Row::Five, Column::G), DiskColor::Dark);
        board.set_disk(position!(Row::Five, Column::H), DiskColor::Dark);
        board.set_disk(position!(Row::Six, Column::F), DiskColor::Light);
        board.set_disk(position!(Row::Six, Column::H), DiskColor::Light);
        board.set_disk(position!(Row::Seven, Column::D), DiskColor::Dark);
        board.set_disk(position!(Row::Seven, Column::E), DiskColor::Dark);
        board.set_disk(position!(Row::Seven, Column::F), DiskColor::Dark);
        board.set_disk(position!(Row::Seven, Column::G), DiskColor::Dark);
        board.set_disk(position!(Row::Seven, Column::H), DiskColor::Light);
        board.set_disk(position!(Row::Eight, Column::H), DiskColor::Light);
        table.set_turn(DiskColor::Dark);

        let app = test::init_service(App::new().route("/step", web::post().to(step_table))).await;
        let req = test::TestRequest::post()
            .uri("/step")
            .set_json(&RequestMessage {
                table,
                action: Action::PassTurn(DiskColor::Dark),
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    /// Test for step table by invalid action "PassTurn"
    #[actix_web::test]
    async fn test_step_table_invalid_pass() {
        let app = test::init_service(App::new().route("/step", web::post().to(step_table))).await;
        let req = test::TestRequest::post()
            .uri("/step")
            .set_json(&RequestMessage {
                table: Table::new(),
                action: Action::PassTurn(DiskColor::Dark),
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
