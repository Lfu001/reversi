use crate::messages::response::StateResponseMessageExt;
use actix_web::{error, web, Responder, Result};
use common::StepRequestMessage;
use reversi_core::controller::Controller;

/// A handler for step table.
///
/// # Arguments
///
/// * `req` - A request message from the client.
///
/// # Returns
///
/// An updated table, puttable positions, and judge result if the game is over.
pub async fn step_table(req: web::Json<StepRequestMessage>) -> Result<impl Responder> {
    let message = req.into_inner();
    let mut table = message.table().to_owned();
    let step_result = Controller::step(&mut table, message.action().to_owned());

    match step_result {
        Ok(step_result) => Ok(StateResponseMessageExt::new(
            table,
            step_result.puttable_positions,
            step_result.judge_result,
        )),
        Err(err) => Err(error::ErrorBadRequest(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{http::StatusCode, test, App};
    use common::{position, Action, Column, DiskColor, Position, PutConfig, Row, Table};
    use reversi_core::state::BoardExt;

    /// Test for step table by valid action "PutDisk"
    #[actix_web::test]
    async fn test_step_table_put() {
        let app = test::init_service(App::new().route("/step", web::post().to(step_table))).await;
        let req = test::TestRequest::post()
            .uri("/step")
            .set_json(StepRequestMessage::new(
                Table::default(),
                Action::PutDisk(PutConfig::new(
                    DiskColor::Dark,
                    position!(Row::Three, Column::D),
                )),
            ))
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
            .set_json(StepRequestMessage::new(
                Table::default(),
                Action::PutDisk(PutConfig::new(
                    DiskColor::Dark,
                    position!(Row::One, Column::A),
                )),
            ))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
    }

    /// Test for step table by valid action "PassTurn"
    #[actix_web::test]
    async fn test_step_table_pass() {
        let mut table = Table::default();
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
            .set_json(StepRequestMessage::new(
                table,
                Action::PassTurn(DiskColor::Dark),
            ))
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
            .set_json(StepRequestMessage::new(
                Table::default(),
                Action::PassTurn(DiskColor::Dark),
            ))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
