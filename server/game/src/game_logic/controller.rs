use super::action::{get_puttable_positions, ActionExt};
use common::{Action, DiskColor, JudgeResult, Position, Table, Winner};
use std::cmp::Ordering;

/// The result of the action execution.
pub struct StepResult {
    pub judge_result: Option<JudgeResult>,
    pub puttable_positions: Vec<Position>,
}

/// The controller of the game.
pub struct Controller;

impl Controller {
    /// Execute an action.
    ///
    /// # Arguments
    ///
    /// * `table` - A table of the game. This table will be modified by the action.
    /// * `action` - An action to execute.
    ///
    /// # Returns
    ///
    /// The result of the action execution. If the game is over after the action, judge result is returned.
    /// Otherwise, next puttable positions are returned. `Err(())` if the action is invalid.
    pub fn step(table: &mut Table, action: Action) -> Result<StepResult, ()> {
        let result = action.execute(table);
        if result.is_err() {
            return Err(());
        }

        if Controller::is_game_over(table) {
            Ok(StepResult {
                judge_result: Some(Controller::judge(table)),
                puttable_positions: vec![],
            })
        } else {
            Ok(StepResult {
                judge_result: None,
                puttable_positions: get_puttable_positions(table.board(), table.turn()),
            })
        }
    }

    /// Check if the game is over.
    ///
    /// # Arguments
    ///
    /// * `table` - A table of the game.
    ///     
    /// # Returns
    ///
    /// `true` if the game is over, otherwise `false`.
    fn is_game_over(table: &Table) -> bool {
        // Nobody can put a disk
        [DiskColor::Dark, DiskColor::Light]
            .iter()
            .all(|color| get_puttable_positions(table.board(), *color).is_empty())
    }

    /// Judge the winner of the game. Assumes that the game is over.
    ///
    /// # Arguments
    ///
    /// * `table` - A table of the game.
    ///
    /// # Returns
    ///
    /// The winner of the game and the number of disks of each color.
    ///
    /// # Panics
    ///
    /// Panics if the game is not over.
    fn judge(table: &Table) -> JudgeResult {
        debug_assert!(Controller::is_game_over(table));

        let (dark_count, light_count) =
            table
                .board()
                .board()
                .iter()
                .fold((0, 0), |(acc_dark, acc_light), disk| match disk {
                    Some(color) => match color {
                        DiskColor::Dark => (acc_dark + 1, acc_light),
                        DiskColor::Light => (acc_dark, acc_light + 1),
                    },
                    None => (acc_dark, acc_light),
                });

        let winner = match dark_count.cmp(&light_count) {
            Ordering::Less => Winner::Win(DiskColor::Light),
            Ordering::Equal => Winner::Draw,
            Ordering::Greater => Winner::Win(DiskColor::Dark),
        };

        JudgeResult::new(dark_count, light_count, winner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_logic::state::BoardExt;
    use common::{position, Column, Row};
    use num_traits::FromPrimitive;

    #[test]
    fn test_is_game_over() {
        // Initial state
        {
            let table = Table::default();
            assert!(!Controller::is_game_over(&table));
        }

        // Continuable game
        {
            let mut table = Table::default();
            let board = table.board_mut();
            board.set_disk(position!(Row::Four, Column::C), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::D), DiskColor::Dark);
            table.set_turn(DiskColor::Light);
            assert!(!Controller::is_game_over(&table));
        }

        // No more empty square
        {
            let mut table = Table::default();
            let board = table.board_mut();
            for row in 0..8 {
                for column in 0..8 {
                    board.set_disk(
                        position!(
                            FromPrimitive::from_i8(row).unwrap(),
                            FromPrimitive::from_i8(column).unwrap()
                        ),
                        DiskColor::Dark,
                    );
                }
            }
            table.set_turn(DiskColor::Light);
            assert!(Controller::is_game_over(&table));
        }

        // There are empty squares but nobody can put
        {
            let mut table = Table::default();
            let board = table.board_mut();
            board.set_disk(position!(Row::Two, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Three, Column::C), DiskColor::Dark);
            board.set_disk(position!(Row::Three, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Three, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::B), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::C), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::C), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Six, Column::D), DiskColor::Dark);
            table.set_turn(DiskColor::Light);
            assert!(Controller::is_game_over(&table));
        }
    }

    #[test]
    fn test_judge() {
        // Dark wins
        {
            let mut table = Table::default();
            let board = table.board_mut();
            board.set_disk(position!(Row::Three, Column::C), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::G), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::G), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::H), DiskColor::Dark);
            board.set_disk(position!(Row::Six, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Six, Column::G), DiskColor::Dark);
            board.set_disk(position!(Row::Seven, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Eight, Column::E), DiskColor::Light);
            board.set_disk(position!(Row::Eight, Column::G), DiskColor::Dark);
            table.set_turn(DiskColor::Light);

            assert_eq!(
                Controller::judge(&table).winner(),
                Winner::Win(DiskColor::Dark)
            );
        }

        // Light wins
        {
            let mut table = Table::default();
            for row in 0..8 {
                for column in 0..8 {
                    table.board_mut().set_disk(
                        position!(Row::from_u8(row).unwrap(), Column::from_u8(column).unwrap()),
                        DiskColor::Light,
                    );
                }
            }
            table.set_turn(DiskColor::Dark);

            assert_eq!(
                Controller::judge(&table).winner(),
                Winner::Win(DiskColor::Light)
            );
        }

        // Draw
        {
            let mut table = Table::default();
            let board = table.board_mut();
            board.set_disk(position!(Row::One, Column::A), DiskColor::Dark);
            board.set_disk(position!(Row::One, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Two, Column::B), DiskColor::Dark);
            board.set_disk(position!(Row::Two, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Three, Column::C), DiskColor::Dark);
            board.set_disk(position!(Row::Three, Column::E), DiskColor::Light);
            board.set_disk(position!(Row::Three, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Three, Column::G), DiskColor::Light);
            board.set_disk(position!(Row::Four, Column::D), DiskColor::Light);
            board.set_disk(position!(Row::Four, Column::E), DiskColor::Light);
            board.set_disk(position!(Row::Four, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::C), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::D), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::E), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Six, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Six, Column::H), DiskColor::Dark);
            board.set_disk(position!(Row::Seven, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Seven, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Seven, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Seven, Column::G), DiskColor::Dark);
            board.set_disk(position!(Row::Seven, Column::H), DiskColor::Dark);
            board.set_disk(position!(Row::Eight, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Eight, Column::H), DiskColor::Dark);
            table.set_turn(DiskColor::Dark);

            assert_eq!(Controller::judge(&table).winner(), Winner::Draw);
        }
    }
}
