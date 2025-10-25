use crate::state::DiskColorExt;
use common::{Action, BitBoard, BitPosition, Column, DiskColor, Position, Row, Table};
use num_traits::FromPrimitive;

/// A bit mask where column A is 0 and other columns are 1.
const NOT_A_COLUMN: u64 = 0x7f7f7f7f7f7f7f7f;
/// A bit mask where column H is 0 and other columns are 1.
const NOT_H_COLUMN: u64 = 0xfefefefefefefefe;

/// A trait which provides an extension method for the [`Action`].
pub trait ActionExt {
    /// Executes the action.
    ///
    /// # Arguments
    ///
    /// * `table` - A table of the game.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the action is executed successfully, `Err(String)` if the action is invalid.
    fn execute(&self, table: &mut Table) -> Result<(), String>;
    /// Checks if the action can be executed.
    ///
    /// # Arguments
    ///
    /// * `table` - A table of the game.
    ///
    /// # Returns
    ///
    /// `true` if the action can be executed, otherwise `false`.
    fn check_inputs(&self, table: &Table) -> bool;
}

impl ActionExt for Action {
    fn execute(&self, table: &mut Table) -> Result<(), String> {
        if !self.check_inputs(table) {
            return Err("Invalid action".to_string());
        }
        // Update the board
        match self {
            Action::PutDisk(config) => {
                let flip_positions =
                    get_flip_positions(table.board(), table.turn(), config.position());
                let board = table.board_mut();
                board.set_disk(*config.position(), config.color());
                for position in flip_positions {
                    board.set_disk(position, config.color());
                }
            }
            Action::PassTurn(_) => {}
        }

        // Update the history
        table.push_history(self.clone());

        // Update the turn
        table.set_turn(table.turn().opposite());

        Ok(())
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
                if get_flip_positions(table.board(), table.turn(), config.position()).is_empty() {
                    return false;
                }

                true
            }
            Action::PassTurn(color) => {
                // Check the turn is correct.
                if *color != table.turn() {
                    return false;
                }
                // Check whether the disk cannot be put.
                if !get_puttable_positions(table.board(), table.turn()).is_empty() {
                    return false;
                }

                true
            }
        }
    }
}

/// A bit board that represents positions on the board that can be put.
pub struct PuttablePositions(pub u64);

impl PuttablePositions {
    /// Checks if there are no puttable positions.
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Returns a list of positions that can be put.
    pub fn to_vec(&self) -> Vec<Position> {
        let mut positions = Vec::new();
        for idx in 0..64 {
            if self.0 & (1 << (63 - idx)) != 0 {
                positions.push(Position::from((
                    Row::from_u8(idx / 8).unwrap(),
                    Column::from_u8(idx % 8).unwrap(),
                )));
            }
        }
        positions
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
pub fn get_puttable_positions(board: &BitBoard, turn: DiskColor) -> PuttablePositions {
    let (player_board, opponent_board) = match turn {
        DiskColor::Dark => (board.dark_plane(), board.light_plane()),
        DiskColor::Light => (board.light_plane(), board.dark_plane()),
    };
    let empty_board = !(player_board | opponent_board);

    let opponent_not_a = opponent_board & NOT_A_COLUMN;
    let opponent_not_h = opponent_board & NOT_H_COLUMN;

    let mut legal_moves = 0;
    let mut rev;

    // Right (>> 1)
    rev = (player_board >> 1) & opponent_not_a;
    rev |= (rev >> 1) & opponent_not_a;
    rev |= (rev >> 1) & opponent_not_a;
    rev |= (rev >> 1) & opponent_not_a;
    rev |= (rev >> 1) & opponent_not_a;
    rev |= (rev >> 1) & opponent_not_a;
    legal_moves |= (rev >> 1) & NOT_A_COLUMN;

    // Left (<< 1)
    rev = (player_board << 1) & opponent_not_h;
    rev |= (rev << 1) & opponent_not_h;
    rev |= (rev << 1) & opponent_not_h;
    rev |= (rev << 1) & opponent_not_h;
    rev |= (rev << 1) & opponent_not_h;
    rev |= (rev << 1) & opponent_not_h;
    legal_moves |= (rev << 1) & NOT_H_COLUMN;

    // Down (>> 8)
    rev = (player_board >> 8) & opponent_board;
    rev |= (rev >> 8) & opponent_board;
    rev |= (rev >> 8) & opponent_board;
    rev |= (rev >> 8) & opponent_board;
    rev |= (rev >> 8) & opponent_board;
    rev |= (rev >> 8) & opponent_board;
    legal_moves |= rev >> 8;

    // Up (<< 8)
    rev = (player_board << 8) & opponent_board;
    rev |= (rev << 8) & opponent_board;
    rev |= (rev << 8) & opponent_board;
    rev |= (rev << 8) & opponent_board;
    rev |= (rev << 8) & opponent_board;
    rev |= (rev << 8) & opponent_board;
    legal_moves |= rev << 8;

    // Right down (>> 9)
    rev = (player_board >> 9) & opponent_not_a;
    rev |= (rev >> 9) & opponent_not_a;
    rev |= (rev >> 9) & opponent_not_a;
    rev |= (rev >> 9) & opponent_not_a;
    rev |= (rev >> 9) & opponent_not_a;
    rev |= (rev >> 9) & opponent_not_a;
    legal_moves |= (rev >> 9) & NOT_A_COLUMN;

    // Left down (>> 7)
    rev = (player_board >> 7) & opponent_not_h;
    rev |= (rev >> 7) & opponent_not_h;
    rev |= (rev >> 7) & opponent_not_h;
    rev |= (rev >> 7) & opponent_not_h;
    rev |= (rev >> 7) & opponent_not_h;
    rev |= (rev >> 7) & opponent_not_h;
    legal_moves |= (rev >> 7) & NOT_H_COLUMN;

    // Right up (<< 7)
    rev = (player_board << 7) & opponent_not_a;
    rev |= (rev << 7) & opponent_not_a;
    rev |= (rev << 7) & opponent_not_a;
    rev |= (rev << 7) & opponent_not_a;
    rev |= (rev << 7) & opponent_not_a;
    rev |= (rev << 7) & opponent_not_a;
    legal_moves |= (rev << 7) & NOT_A_COLUMN;

    // Left up (<< 9)
    rev = (player_board << 9) & opponent_not_h;
    rev |= (rev << 9) & opponent_not_h;
    rev |= (rev << 9) & opponent_not_h;
    rev |= (rev << 9) & opponent_not_h;
    rev |= (rev << 9) & opponent_not_h;
    rev |= (rev << 9) & opponent_not_h;
    legal_moves |= (rev << 9) & NOT_H_COLUMN;

    PuttablePositions(legal_moves & empty_board)
}

/// A bit board that represents positions on the board that can be flipped.
struct FlipPositions(pub u64);

impl FlipPositions {
    /// Checks if there are no positions that can be flipped.
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
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
fn get_flip_positions(board: &BitBoard, turn: DiskColor, position: BitPosition) -> FlipPositions {
    let (player_board, opponent_board) = match turn {
        DiskColor::Dark => (board.dark_plane(), board.light_plane()),
        DiskColor::Light => (board.light_plane(), board.dark_plane()),
    };
    let mut flipped_positions = FlipPositions(0u64);

    /// A direction and a mask for that direction.
    struct Direction {
        /// A shift value for a direction. Positive values mean left shift and negative values mean right shift.
        shift: i8,
        /// A bit mask to prevent wrapping around the board.
        mask: u64,
    }

    const ALL_BITS: u64 = 0xffffffffffffffff;
    const DIRECTIONS: [Direction; 8] = [
        Direction {
            shift: -1,
            mask: NOT_A_COLUMN,
        }, // Right
        Direction {
            shift: 1,
            mask: NOT_H_COLUMN,
        }, // Left
        Direction {
            shift: -8,
            mask: ALL_BITS,
        }, // Down
        Direction {
            shift: 8,
            mask: ALL_BITS,
        }, // Up
        Direction {
            shift: -9,
            mask: NOT_A_COLUMN,
        }, // Right down
        Direction {
            shift: -7,
            mask: NOT_H_COLUMN,
        }, // Left down
        Direction {
            shift: 7,
            mask: NOT_A_COLUMN,
        }, // Right up
        Direction {
            shift: 9,
            mask: NOT_H_COLUMN,
        }, // Left up
    ];

    for direction in DIRECTIONS {
        let mut line = 0u64;
        let mut scanner = position.0;

        loop {
            if direction.shift > 0 {
                scanner <<= direction.shift as u32;
            } else {
                scanner >>= -direction.shift as u32;
            }
            // Remove bits that wrapped around
            scanner &= direction.mask;

            if scanner == 0 || (scanner & opponent_board) == 0 {
                // If the line ends or there are no opponent disks
                break;
            }

            line |= scanner;
        }

        if line != 0 && (scanner & player_board) != 0 {
            // If there are continuous opponent disks and the line ends with player's disk
            flipped_positions.0 |= line;
        }
    }

    flipped_positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Column, PutConfig, Row, position};
    use num_traits::FromPrimitive;

    fn xy_to_bit(row: u8, column: u8) -> u64 {
        let position = position!(Row::from_u8(row).unwrap(), Column::from_u8(column).unwrap());
        BitPosition::from(position).0
    }

    #[test]
    fn test_get_puttable_positions() {
        // Check 1
        let mut table = Table::default();
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
        table = Table::default();
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
        let mut table = Table::default();
        // Dark turn
        let flip = get_flip_positions(
            table.board(),
            table.turn(),
            &position!(Row::Four, Column::B),
        );
        assert_eq!(flip, vec![]);

        // Check 2
        table = Table::default();
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
            let table = Table::default();
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::One, Column::A),
            ));
            assert!(!action.check_inputs(&table));

            // The square is not empty
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Four, Column::E),
            ));
            assert!(!action.check_inputs(&table));
        }
        // Invalid turn
        {
            // Dark's turn but light try to put
            let mut table = Table::default();
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Light,
                position!(Row::Four, Column::F),
            ));
            assert!(!action.check_inputs(&table));

            // Light's turn but dark try to put
            let board = table.board();
            let mask = xy_to_bit(Row::Five as u8, Column::E as u8)
                | xy_to_bit(Row::Five as u8, Column::F as u8);
            let new_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            table.set_board(new_board);
            table.set_turn(DiskColor::Light);
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Four, Column::C),
            ));
            assert!(!action.check_inputs(&table));
        }

        // Pass turn
        // Invalid turn
        {
            // Dark's turn but light try to pass
            let mut table = Table::default();
            let board = table.board();
            let mask = xy_to_bit(Row::Three as u8, Column::D as u8)
                | xy_to_bit(Row::Three as u8, Column::E as u8)
                | xy_to_bit(Row::Three as u8, Column::F as u8)
                | xy_to_bit(Row::Four as u8, Column::F as u8)
                | xy_to_bit(Row::Five as u8, Column::D as u8)
                | xy_to_bit(Row::Five as u8, Column::F as u8);
            let new_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Light);
            assert!(!action.check_inputs(&table));

            // Light's turn but dark try to pass
            let mut table = Table::default();
            table.set_turn(DiskColor::Light);
            let board = table.board();
            let mask = xy_to_bit(Row::Four as u8, Column::D as u8)
                | xy_to_bit(Row::Four as u8, Column::F as u8)
                | xy_to_bit(Row::Five as u8, Column::F as u8)
                | xy_to_bit(Row::Six as u8, Column::D as u8)
                | xy_to_bit(Row::Six as u8, Column::E as u8)
                | xy_to_bit(Row::Six as u8, Column::F as u8);
            let new_board = BitBoard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Dark);
            assert!(!action.check_inputs(&table));
        }
        // Invalid pass
        {
            // Dark can put but try to pass
            let table = Table::default();
            let action = Action::PassTurn(DiskColor::Dark);
            assert!(!action.check_inputs(&table));

            // Light can put but try to pass
            let mut table = Table::default();
            table.set_turn(DiskColor::Light);
            let board = table.board();
            let mask = xy_to_bit(Row::Five as u8, Column::E as u8)
                | xy_to_bit(Row::Five as u8, Column::F as u8);
            let new_board = BitBoard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Light);
            assert!(!action.check_inputs(&table));
        }

        // Valid put
        {
            let table = Table::default();
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Four, Column::C),
            ));
            assert!(action.check_inputs(&table));
        }
        // Valid pass
        {
            let mut table = Table::default();
            table.set_turn(DiskColor::Light);
            let board = table.board();
            let mask = xy_to_bit(Row::Three as u8, Column::D as u8)
                | xy_to_bit(Row::Three as u8, Column::E as u8)
                | xy_to_bit(Row::Three as u8, Column::F as u8)
                | xy_to_bit(Row::Four as u8, Column::F as u8)
                | xy_to_bit(Row::Five as u8, Column::D as u8)
                | xy_to_bit(Row::Five as u8, Column::F as u8);
            let new_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Light);
            assert!(action.check_inputs(&table));
        }
    }

    #[test]
    fn test_execute() {
        let mut table = Table::default();
        let mut expected_board: BitBoard;

        // Dark put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Four, Column::C),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::Four as u8, Column::C as u8)
                | xy_to_bit(Row::Four as u8, Column::D as u8);
            expected_board = BitBoard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Light,
                position!(Row::Three, Column::C),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::Three as u8, Column::C as u8)
                | xy_to_bit(Row::Four as u8, Column::D as u8);
            expected_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Dark);
        }

        // Dark put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Two, Column::C),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::Two as u8, Column::C as u8)
                | xy_to_bit(Row::Three as u8, Column::C as u8);
            expected_board = BitBoard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Light,
                position!(Row::Two, Column::B),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::Two as u8, Column::B as u8)
                | xy_to_bit(Row::Three as u8, Column::C as u8);
            expected_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Dark);
        }

        // Dark put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Six, Column::E),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::Six as u8, Column::E as u8)
                | xy_to_bit(Row::Five as u8, Column::E as u8);
            expected_board = BitBoard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Light,
                position!(Row::One, Column::C),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::One as u8, Column::C as u8)
                | xy_to_bit(Row::Two as u8, Column::C as u8);
            expected_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Dark);
        }

        // Dark put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::One, Column::A),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::One as u8, Column::A as u8)
                | xy_to_bit(Row::Two as u8, Column::B as u8)
                | xy_to_bit(Row::Three as u8, Column::C as u8)
                | xy_to_bit(Row::Four as u8, Column::D as u8);
            expected_board = BitBoard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Light,
                position!(Row::Three, Column::A),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::Three as u8, Column::A as u8)
                | xy_to_bit(Row::Two as u8, Column::B as u8);
            expected_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Dark);
        }

        // Dark pass
        {
            let action = Action::PassTurn(DiskColor::Dark);
            action.execute(&mut table).unwrap();
            let board = table.board();
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Light);
        }

        // Light put
        {
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Light,
                position!(Row::Five, Column::C),
            ));
            action.execute(&mut table).unwrap();
            let board = table.board();
            let mask = xy_to_bit(Row::Five as u8, Column::C as u8)
                | xy_to_bit(Row::Four as u8, Column::C as u8)
                | xy_to_bit(Row::Three as u8, Column::C as u8);
            expected_board = BitBoard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Dark);
        }
    }
}
