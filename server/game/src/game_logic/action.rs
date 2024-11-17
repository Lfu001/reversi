use super::state::{Board, DiskColor, Position, Table};
use crate::position;
use num_traits::FromPrimitive;

/// A configuration of the put action.
#[derive(Clone)]
pub struct PutConfig {
    /// A color of the disk.
    color: DiskColor,
    /// A position of the square in the board.
    position: Position,
}

impl PutConfig {
    /// Returns a read-only reference to the color of the disk.
    pub fn color(&self) -> &DiskColor {
        &self.color
    }

    /// Returns a read-only reference to the position.
    pub fn position(&self) -> &Position {
        &self.position
    }
}

/// Reversi actions.
#[derive(Clone)]
pub enum Action {
    /// Put a disk to the square.
    PutDisk(PutConfig),
    /// Pass the turn.
    PassTurn(DiskColor),
}

impl Action {
    /// Executes the action.
    ///
    /// # Arguments
    ///
    /// * `table` - A table of the game.
    ///
    /// # Panics
    ///
    /// Panics if the action cannot be executed.
    pub fn execute(&self, table: &mut Table) {
        debug_assert!(self.check_inputs(&table));
        // Update the board
        match self {
            Action::PutDisk(config) => {
                let flip_positions =
                    get_flip_positions(table.board(), table.turn(), &config.position);
                let board = table.board_mut();
                board.set_disk(config.position, *config.color());
                for position in flip_positions {
                    board.set_disk(position, *config.color());
                }
            }
            Action::PassTurn(_) => (),
        }

        // Update the history
        let history = table.history_mut();
        history.push(self.clone());

        // Update the turn
        table.set_turn(table.turn().opposite());
    }

    /// Checks if the action can be executed.
    ///
    /// # Arguments
    ///
    /// * `table` - A table of the game.
    ///
    /// # Returns
    ///
    /// `true` if the action can be executed, otherwise `false`.
    fn check_inputs(&self, table: &Table) -> bool {
        match self {
            Action::PutDisk(config) => {
                // Check whether the turn is correct.
                if config.color() != table.turn() {
                    return false;
                }
                // Check whether the square is empty.
                if table.board().get_disk(config.position()).is_some() {
                    return false;
                }
                // Check whether the disk can be put.
                if get_flip_positions(table.board(), table.turn(), &config.position()).len() == 0 {
                    return false;
                }

                true
            }
            Action::PassTurn(color) => {
                // Check the turn is correct.
                if color != table.turn() {
                    return false;
                }
                // Check whether the disk cannot be put.
                if get_puttable_positions(table.board(), table.turn()).len() > 0 {
                    return false;
                }

                true
            }
        }
    }
}

/// Returns a list of positions that can be put.
///
/// # Arguments
///
/// * `board` - A board of the game.
/// * `turn` - A current turn color.
///
/// # Returns
///
/// A list of positions that can be put.
pub fn get_puttable_positions(board: &Board, turn: &DiskColor) -> Vec<Position> {
    let mut positions = vec![];
    for row in 0..8 {
        for column in 0..8 {
            let position = position!(
                FromPrimitive::from_i8(row).unwrap(),
                FromPrimitive::from_i8(column).unwrap()
            );
            if board.get_disk(&position).is_some() {
                continue;
            }
            if get_flip_positions(board, turn, &position).len() > 0 {
                positions.push(position);
            };
        }
    }
    positions
}

/// Returns a list of positions that can be flipped.
///
/// # Arguments
///
/// * `board` - A board of the game.
/// * `turn` - A current turn color.
/// * `position` - A position to put the disk.
///
/// # Returns
///
/// A list of positions that can be flipped.
fn get_flip_positions(board: &Board, turn: &DiskColor, position: &Position) -> Vec<Position> {
    if board.get_disk(position).is_some() {
        return vec![];
    }

    let mut positions = vec![];
    let rays = board.get_rays(&position);
    for ray in rays {
        let mut local_positions = vec![];
        for (square, pos) in ray {
            match square {
                Some(color) => {
                    if color == *turn {
                        positions.append(&mut local_positions);
                        break;
                    }
                    local_positions.push(pos);
                }
                None => break,
            };
        }
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_logic::state::{Board, Column, Row};

    #[test]
    fn test_get_puttable_positions() {
        // Check 1
        let mut table = Table::new();
        // Dark turn
        let puttable = get_puttable_positions(table.board(), table.turn());
        assert_eq!(
            puttable,
            vec![
                position!(Row::Three, Column::D),
                position!(Row::Four, Column::C),
                position!(Row::Five, Column::F),
                position!(Row::Six, Column::E),
            ]
        );

        // Check 2
        table = Table::new();
        table
            .board_mut()
            .set_disk(position!(Row::Five, Column::F), DiskColor::Dark);
        table
            .board_mut()
            .set_disk(position!(Row::Six, Column::F), DiskColor::Light);
        // Dark turn
        let puttable = get_puttable_positions(table.board(), table.turn());
        assert_eq!(
            puttable,
            vec![
                position!(Row::Three, Column::D),
                position!(Row::Four, Column::C),
                position!(Row::Six, Column::E),
                position!(Row::Seven, Column::F),
            ]
        );
    }

    #[test]
    fn test_get_flip_positions() {
        // Check 1
        let mut table = Table::new();
        // Dark turn
        let flip = get_flip_positions(
            table.board(),
            table.turn(),
            &position!(Row::Four, Column::B),
        );
        assert_eq!(flip, vec![]);

        // Check 2
        table = Table::new();
        table
            .board_mut()
            .set_disk(position!(Row::Five, Column::F), DiskColor::Dark);
        table
            .board_mut()
            .set_disk(position!(Row::Six, Column::F), DiskColor::Light);
        table
            .board_mut()
            .set_disk(position!(Row::Five, Column::E), DiskColor::Dark);
        table
            .board_mut()
            .set_disk(position!(Row::Six, Column::E), DiskColor::Dark);
        // Light turn
        table.set_turn(DiskColor::Light);
        let flip = get_flip_positions(
            table.board(),
            table.turn(),
            &position!(Row::Four, Column::F),
        );
        assert_eq!(
            flip,
            vec![
                position!(Row::Four, Column::E),
                position!(Row::Five, Column::F)
            ]
        );
    }

    #[test]
    fn test_check_inputs() {
        // Put disk
        // Invalid position
        {
            // Not flipping any disk
            let table = Table::new();
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::One, Column::A),
            });
            assert_eq!(action.check_inputs(&table), false);

            // The square is not empty
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::Four, Column::E),
            });
            assert_eq!(action.check_inputs(&table), false);
        }
        // Invalid turn
        {
            // Dark's turn but light try to put
            let mut table = Table::new();
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Light,
                position: position!(Row::Four, Column::F),
            });
            assert_eq!(action.check_inputs(&table), false);

            // Light's turn but dark try to put
            let board = table.board_mut();
            board.set_disk(position!(Row::Five, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::F), DiskColor::Dark);
            table.set_turn(DiskColor::Light);
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::Four, Column::C),
            });
            assert_eq!(action.check_inputs(&table), false);
        }

        // Pass turn
        // Invalid turn
        {
            // Dark's turn but light try to pass
            let mut table = Table::new();
            let board = table.board_mut();
            board.set_disk(position!(Row::Three, Column::D), DiskColor::Light);
            board.set_disk(position!(Row::Three, Column::E), DiskColor::Light);
            board.set_disk(position!(Row::Three, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Four, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::D), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::F), DiskColor::Light);
            let action = Action::PassTurn(DiskColor::Light);
            assert_eq!(action.check_inputs(&table), false);

            // Light's turn but dark try to pass
            let mut table = Table::new();
            table.set_turn(DiskColor::Light);
            let board = table.board_mut();
            board.set_disk(position!(Row::Four, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Four, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::F), DiskColor::Dark);
            board.set_disk(position!(Row::Six, Column::D), DiskColor::Dark);
            board.set_disk(position!(Row::Six, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Six, Column::F), DiskColor::Dark);
            let action = Action::PassTurn(DiskColor::Dark);
            assert_eq!(action.check_inputs(&table), false);
        }
        // Invalid pass
        {
            // Dark can put but try to pass
            let table = Table::new();
            let action = Action::PassTurn(DiskColor::Dark);
            assert_eq!(action.check_inputs(&table), false);

            // Light can put but try to pass
            let mut table = Table::new();
            table.set_turn(DiskColor::Light);
            let board = table.board_mut();
            board.set_disk(position!(Row::Five, Column::E), DiskColor::Dark);
            board.set_disk(position!(Row::Five, Column::F), DiskColor::Dark);
            let action = Action::PassTurn(DiskColor::Light);
            assert_eq!(action.check_inputs(&table), false);
        }

        // Valid put
        {
            let table = Table::new();
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::Four, Column::C),
            });
            assert_eq!(action.check_inputs(&table), true);
        }
        // Valid pass
        {
            let mut table = Table::new();
            table.set_turn(DiskColor::Light);
            let board = table.board_mut();
            board.set_disk(position!(Row::Three, Column::D), DiskColor::Light);
            board.set_disk(position!(Row::Three, Column::E), DiskColor::Light);
            board.set_disk(position!(Row::Three, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Four, Column::F), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::D), DiskColor::Light);
            board.set_disk(position!(Row::Five, Column::F), DiskColor::Light);
            let action = Action::PassTurn(DiskColor::Light);
            assert_eq!(action.check_inputs(&table), true);
        }
    }

    #[test]
    fn test_execute() {
        let mut table = Table::new();
        let mut expected_board = Board::new();

        // Dark put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::Four, Column::C),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::Four, Column::C), DiskColor::Dark);
            expected_board.set_disk(position!(Row::Four, Column::D), DiskColor::Dark);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Light,
                position: position!(Row::Three, Column::C),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::Three, Column::C), DiskColor::Light);
            expected_board.set_disk(position!(Row::Four, Column::D), DiskColor::Light);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Dark);
        }

        // Dark put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::Two, Column::C),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::Two, Column::C), DiskColor::Dark);
            expected_board.set_disk(position!(Row::Three, Column::C), DiskColor::Dark);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Light,
                position: position!(Row::Two, Column::B),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::Two, Column::B), DiskColor::Light);
            expected_board.set_disk(position!(Row::Three, Column::C), DiskColor::Light);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Dark);
        }

        // Dark put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::Six, Column::E),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::Six, Column::E), DiskColor::Dark);
            expected_board.set_disk(position!(Row::Five, Column::E), DiskColor::Dark);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Light,
                position: position!(Row::One, Column::C),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::One, Column::C), DiskColor::Light);
            expected_board.set_disk(position!(Row::Two, Column::C), DiskColor::Light);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Dark);
        }

        // Dark put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Dark,
                position: position!(Row::One, Column::A),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::One, Column::A), DiskColor::Dark);
            expected_board.set_disk(position!(Row::Two, Column::B), DiskColor::Dark);
            expected_board.set_disk(position!(Row::Three, Column::C), DiskColor::Dark);
            expected_board.set_disk(position!(Row::Four, Column::D), DiskColor::Dark);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Light,
                position: position!(Row::Three, Column::A),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::Three, Column::A), DiskColor::Light);
            expected_board.set_disk(position!(Row::Two, Column::B), DiskColor::Light);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Dark);
        }

        // Dark pass
        {
            let action = Action::PassTurn(DiskColor::Dark);
            action.execute(&mut table);
            let board = table.board();
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig {
                color: DiskColor::Light,
                position: position!(Row::Five, Column::C),
            });
            action.execute(&mut table);
            let board = table.board();
            expected_board.set_disk(position!(Row::Five, Column::C), DiskColor::Light);
            expected_board.set_disk(position!(Row::Four, Column::C), DiskColor::Light);
            expected_board.set_disk(position!(Row::Three, Column::C), DiskColor::Light);
            assert_eq!(board.raw_board(), expected_board.raw_board());
            assert_eq!(table.turn(), &DiskColor::Dark);
        }
    }
}
