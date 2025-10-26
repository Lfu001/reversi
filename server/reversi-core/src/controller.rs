use super::action::{ActionExt, get_puttable_positions};
use crate::action::PuttablePositions;
use common::{Action, DiskColor, JudgeResult, Table, Winner};
use std::cmp::Ordering;

/// The result of the action execution.
pub struct StepResult {
    pub judge_result: Option<JudgeResult>,
    pub puttable_positions: PuttablePositions,
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
    /// Otherwise, next puttable positions are returned. `Err(String)` if the action is invalid.
    pub fn step(table: &mut Table, action: Action) -> Result<StepResult, String> {
        action.execute(table)?;

        if Controller::is_game_over(table) {
            Ok(StepResult {
                judge_result: Some(Controller::judge(table)),
                puttable_positions: PuttablePositions(0),
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

        let dark_count = table.board().dark_plane().count_ones();
        let light_count = table.board().light_plane().count_ones();

        let winner = match dark_count.cmp(&light_count) {
            Ordering::Less => Winner::Win(DiskColor::Light),
            Ordering::Equal => Winner::Draw,
            Ordering::Greater => Winner::Win(DiskColor::Dark),
        };

        JudgeResult::new(
            dark_count.try_into().unwrap(),
            light_count.try_into().unwrap(),
            winner,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{BitPosition, Bitboard, Column, Position, Row};

    fn xy_to_bit(row: Row, column: Column) -> u64 {
        let position = Position::new(row, column);
        BitPosition::from(position).0
    }

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
            let board = table.board();
            let mask = xy_to_bit(Row::Four, Column::C) | xy_to_bit(Row::Four, Column::D);
            let new_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            table.set_board(new_board);
            table.set_turn(DiskColor::Light);
            assert!(!Controller::is_game_over(&table));
        }

        // No more empty square
        {
            let mut table = Table::default();
            let fill = !0u64;
            let new_board = Bitboard::new(fill, 0);
            table.set_board(new_board);
            table.set_turn(DiskColor::Light);
            assert!(Controller::is_game_over(&table));
        }

        // There are empty squares but nobody can put
        {
            let mut table = Table::default();
            let board = table.board();
            let mask = xy_to_bit(Row::Two, Column::D)
                | xy_to_bit(Row::Three, Column::C)
                | xy_to_bit(Row::Three, Column::D)
                | xy_to_bit(Row::Three, Column::E)
                | xy_to_bit(Row::Four, Column::B)
                | xy_to_bit(Row::Four, Column::C)
                | xy_to_bit(Row::Four, Column::D)
                | xy_to_bit(Row::Four, Column::E)
                | xy_to_bit(Row::Four, Column::F)
                | xy_to_bit(Row::Five, Column::C)
                | xy_to_bit(Row::Five, Column::D)
                | xy_to_bit(Row::Five, Column::E)
                | xy_to_bit(Row::Six, Column::D);
            let new_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            table.set_board(new_board);
            table.set_turn(DiskColor::Light);
            assert!(Controller::is_game_over(&table));
        }
    }

    #[test]
    fn test_judge() {
        // Dark wins
        {
            let mut table = Table::default();
            let board = table.board();
            let mask = xy_to_bit(Row::Three, Column::C)
                | xy_to_bit(Row::Four, Column::D)
                | xy_to_bit(Row::Four, Column::E)
                | xy_to_bit(Row::Four, Column::F)
                | xy_to_bit(Row::Four, Column::G)
                | xy_to_bit(Row::Five, Column::D)
                | xy_to_bit(Row::Five, Column::E)
                | xy_to_bit(Row::Five, Column::F)
                | xy_to_bit(Row::Five, Column::G)
                | xy_to_bit(Row::Five, Column::H)
                | xy_to_bit(Row::Six, Column::E)
                | xy_to_bit(Row::Six, Column::G)
                | xy_to_bit(Row::Seven, Column::F)
                | xy_to_bit(Row::Eight, Column::E)
                | xy_to_bit(Row::Eight, Column::G);
            let new_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            table.set_board(new_board);
            table.set_turn(DiskColor::Light);

            assert_eq!(
                Controller::judge(&table).winner(),
                Winner::Win(DiskColor::Dark)
            );
        }

        // Light wins
        {
            let mut table = Table::default();
            let fill = !0u64;
            let new_board = Bitboard::new(0, fill);
            table.set_board(new_board);
            table.set_turn(DiskColor::Dark);

            assert_eq!(
                Controller::judge(&table).winner(),
                Winner::Win(DiskColor::Light)
            );
        }

        // Draw
        {
            let mut table = Table::default();
            let board = table.board();
            let dark_mask = xy_to_bit(Row::One, Column::A)
                | xy_to_bit(Row::Two, Column::B)
                | xy_to_bit(Row::Three, Column::C)
                | xy_to_bit(Row::Six, Column::F)
                | xy_to_bit(Row::Six, Column::H)
                | xy_to_bit(Row::Seven, Column::D)
                | xy_to_bit(Row::Seven, Column::E)
                | xy_to_bit(Row::Seven, Column::F)
                | xy_to_bit(Row::Seven, Column::G)
                | xy_to_bit(Row::Seven, Column::H)
                | xy_to_bit(Row::Eight, Column::F)
                | xy_to_bit(Row::Eight, Column::H);
            let light_mask = xy_to_bit(Row::One, Column::F)
                | xy_to_bit(Row::Two, Column::F)
                | xy_to_bit(Row::Three, Column::E)
                | xy_to_bit(Row::Three, Column::F)
                | xy_to_bit(Row::Three, Column::G)
                | xy_to_bit(Row::Four, Column::D)
                | xy_to_bit(Row::Four, Column::E)
                | xy_to_bit(Row::Four, Column::F)
                | xy_to_bit(Row::Five, Column::C)
                | xy_to_bit(Row::Five, Column::D)
                | xy_to_bit(Row::Five, Column::E)
                | xy_to_bit(Row::Five, Column::F);
            let new_board = Bitboard::new(
                (board.dark_plane() | dark_mask) & !light_mask,
                (board.light_plane() | light_mask) & !dark_mask,
            );
            table.set_board(new_board);
            table.set_turn(DiskColor::Dark);

            assert_eq!(Controller::judge(&table).winner(), Winner::Draw);
        }
    }
}
