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
            step_result.puttable_positions.into_iter().collect(),
            step_result.judge_result,
        )),
        Err(err) => Err(error::ErrorBadRequest(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{http::StatusCode, test, App};
    use common::{
        position, Action, BitPosition, Bitboard, Column, DiskColor, Position, PutConfig, Row, Table,
    };

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

    fn xy_to_bit(row: Row, column: Column) -> u64 {
        let position = Position::new(row, column);
        BitPosition::from(position).0
    }

    /// Test for step table by valid action "PassTurn"
    #[actix_web::test]
    async fn test_step_table_pass() {
        let mut table = Table::default();
        let board = table.board();
        let dark_mask = xy_to_bit(Row::Four, Column::C)
            | xy_to_bit(Row::Four, Column::D)
            | xy_to_bit(Row::Five, Column::E)
            | xy_to_bit(Row::Five, Column::F)
            | xy_to_bit(Row::Five, Column::G)
            | xy_to_bit(Row::Five, Column::H)
            | xy_to_bit(Row::Seven, Column::D)
            | xy_to_bit(Row::Seven, Column::E)
            | xy_to_bit(Row::Seven, Column::F)
            | xy_to_bit(Row::Seven, Column::G);
        let light_mask = xy_to_bit(Row::Six, Column::F)
            | xy_to_bit(Row::Six, Column::H)
            | xy_to_bit(Row::Seven, Column::H)
            | xy_to_bit(Row::Eight, Column::H);
        let new_board = Bitboard::new(
            (board.dark_plane() | dark_mask) & !light_mask,
            (board.light_plane() | light_mask) & !dark_mask,
        );
        table.set_board(new_board);
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
