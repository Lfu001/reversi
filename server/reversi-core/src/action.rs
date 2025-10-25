use crate::state::DiskColorExt;
use common::{Action, BitPosition, Bitboard, Column, DiskColor, Position, Row, Table};
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
    /// `Ok(())` if the action can be executed, `Err(String)` if the action is invalid.
    fn check_inputs(&self, table: &Table) -> Result<(), String>;
}

impl ActionExt for Action {
    fn execute(&self, table: &mut Table) -> Result<(), String> {
        self.check_inputs(table)?;

        // Update the board
        match self {
            Action::PutDisk(config) => {
                let bit_position = BitPosition::from(*config.position());
                let flip_positions =
                    get_flip_positions(table.board(), table.turn(), bit_position.clone());

                let mut new_board = place_disk(table.board(), bit_position, config.color());
                new_board = flip_disks(&new_board, flip_positions, config.color());

                table.set_board(new_board);
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
    /// `Ok(())` if the action can be executed, `Err(String)` if the action is invalid.
    fn check_inputs(&self, table: &Table) -> Result<(), String> {
        match self {
            Action::PutDisk(config) => {
                // Check whether the turn is correct.
                if config.color() != table.turn() {
                    return Err(format!(
                        "Current turn is {:?}, but tried to put a disk with the wrong turn: {:?}",
                        table.turn(),
                        config.color()
                    ));
                }
                // Check whether the square is empty.
                let bit_position = BitPosition::from(*config.position());
                let dark_plane = table.board().dark_plane();
                let light_plane = table.board().light_plane();
                if (dark_plane & bit_position.0 != 0) || (light_plane & bit_position.0 != 0) {
                    return Err(format!(
                        "Tried to put a disk on a non-empty square: {:?}",
                        config.position()
                    ));
                }
                // Check whether the disk can be put.
                if get_flip_positions(table.board(), table.turn(), bit_position).is_empty() {
                    return Err(format!(
                        "Tried to put a disk on a square that cannot be put: {:?}",
                        config.position()
                    ));
                }

                Ok(())
            }
            Action::PassTurn(color) => {
                // Check the turn is correct.
                if *color != table.turn() {
                    return Err(format!(
                        "Current turn is {:?}, but tried to pass turn with the wrong turn: {:?}",
                        table.turn(),
                        color
                    ));
                }
                // Check whether the disk cannot be put.
                if !get_puttable_positions(table.board(), table.turn()).is_empty() {
                    return Err(String::from(
                        "Tried to pass turn but there are puttable positions",
                    ));
                }

                Ok(())
            }
        }
    }
}

/// A bitboard that represents positions on the board that can be put.
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
pub fn get_puttable_positions(board: &Bitboard, turn: DiskColor) -> PuttablePositions {
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

/// A bitboard that represents positions on the board that can be flipped.
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
fn get_flip_positions(board: &Bitboard, turn: DiskColor, position: BitPosition) -> FlipPositions {
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

/// Flips all opponent's disks between the newly placed disk and existing disks of the same color.
///
/// Takes the current `board` state and flips all disks at `flip_positions` to match the `player_color`.
/// The `flip_positions` should contain only the positions of the opponent's disks that are between
/// the newly placed player's disk and an existing player's disk. Returns a new [`BitBoard`] with
/// the captured disks flipped to the player's color.
fn flip_disks(
    board: &Bitboard,
    flip_positions: FlipPositions,
    player_color: DiskColor,
) -> Bitboard {
    match player_color {
        DiskColor::Dark => Bitboard::new(
            // Add flipped positions to dark plane
            board.dark_plane() | flip_positions.0,
            // Remove flipped positions from light plane
            board.light_plane() ^ flip_positions.0,
        ),
        DiskColor::Light => Bitboard::new(
            // Remove flipped positions from dark plane
            board.dark_plane() ^ flip_positions.0,
            // Add flipped positions to light plane
            board.light_plane() | flip_positions.0,
        ),
    }
}

/// Places a new disk of the specified color on the board at the given position.
///
/// Returns a new [`BitBoard`] with the disk of `color` placed at `position`. The position
/// must be empty, and the function does not perform any validation.
fn place_disk(board: &Bitboard, position: BitPosition, color: DiskColor) -> Bitboard {
    match color {
        DiskColor::Dark => Bitboard::new(board.dark_plane() | position.0, board.light_plane()),
        DiskColor::Light => Bitboard::new(board.dark_plane(), board.light_plane() | position.0),
    }
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
    fn test_bit_puttable_position_to_vec() {
        let puttable_positions = PuttablePositions(
            0b00000000_00000001_00000000_00000000_10000000_00000000_00000000_00000000,
        );
        let expected = vec![
            position!(Row::Two, Column::H),
            position!(Row::Five, Column::A),
        ];
        assert_eq!(puttable_positions.to_vec(), expected);
    }

    mod tests_get_puttable_positions {
        use super::*;

        #[test]
        fn should_return_four_positions_for_dark_at_initial_board() {
            let initial_board = Bitboard::default();
            let turn = DiskColor::Dark;
            let expected = xy_to_bit(2, 3) | xy_to_bit(3, 2) | xy_to_bit(5, 4) | xy_to_bit(4, 5);

            let result = get_puttable_positions(&initial_board, turn);

            assert_eq!(result.0, expected);
        }

        #[test]
        fn should_return_four_positions_for_light_at_initial_board() {
            let initial_board = Bitboard::default();
            let turn = DiskColor::Light;
            let expected = xy_to_bit(4, 2) | xy_to_bit(5, 3) | xy_to_bit(2, 4) | xy_to_bit(3, 5);

            let result = get_puttable_positions(&initial_board, turn);

            assert_eq!(result.0, expected);
        }

        #[test]
        fn should_return_no_positions_when_no_moves_are_available() {
            let board = Bitboard::new(xy_to_bit(0, 0), xy_to_bit(7, 7));
            let turn = DiskColor::Dark;
            let expected = 0;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, expected);
        }

        #[test]
        fn puttable_right_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 2), xy_to_bit(3, 3)); // Player:C4, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(3, 4)); // Expect: E4 is puttable
        }

        #[test]
        fn puttable_right_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 6), xy_to_bit(3, 7)); // Player:G4, Opponent:H4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn); // A5 is empty

            assert_eq!(result.0 & xy_to_bit(4, 0), 0); // Expect: A5 is NOT puttable
        }

        #[test]
        fn puttable_left_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 4), xy_to_bit(3, 3)); // Player:E4, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(3, 2)); // Expect: C4 is puttable
        }

        #[test]
        fn puttable_left_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 1), xy_to_bit(3, 0)); // Player:B4, Opponent:A4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn); // H3 is empty
            assert_eq!(result.0 & xy_to_bit(2, 7), 0); // Expect: H3 is NOT puttable
        }

        #[test]
        fn puttable_down_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 3), xy_to_bit(3, 3)); // Player:D3, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(4, 3)); // Expect: D5 is puttable
        }

        #[test]
        fn puttable_down_edge_case() {
            let board = Bitboard::new(xy_to_bit(6, 3), xy_to_bit(7, 3)); // Player:D7, Opponent:D8
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, 0); // Expect: No moves
        }

        #[test]
        fn puttable_up_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 3), xy_to_bit(3, 3)); // Player:D5, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(2, 3)); // Expect: D3 is puttable
        }

        #[test]
        fn puttable_up_edge_case() {
            let board = Bitboard::new(xy_to_bit(1, 3), xy_to_bit(0, 3)); // Player:D2, Opponent:D1
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, 0); // Expect: No moves
        }

        #[test]
        fn puttable_right_down_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 2), xy_to_bit(3, 3)); // Player:C3, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(4, 4)); // Expect: E5 is puttable
        }

        #[test]
        fn puttable_right_down_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 6), xy_to_bit(3, 7)); // Player:G3, Opponent:H4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn); // A5 is empty

            assert_eq!(result.0 & xy_to_bit(4, 0), 0); // Expect: A5 is NOT puttable
        }

        #[test]
        fn puttable_left_down_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 4), xy_to_bit(3, 3)); // Player:E3, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(4, 2)); // Expect: C5 is puttable
        }

        #[test]
        fn puttable_left_down_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 1), xy_to_bit(3, 0)); // Player:B3, Opponent:A4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn); // H5 is empty

            assert_eq!(result.0 & xy_to_bit(4, 7), 0); // Expect: H5 is NOT puttable
        }

        #[test]
        fn puttable_right_up_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 2), xy_to_bit(3, 3)); // Player:C5, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(2, 4)); // Expect: E3 is puttable
        }

        #[test]
        fn puttable_right_up_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 6), xy_to_bit(3, 7)); // Player:G5, Opponent:H4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn); // A3 is empty

            assert_eq!(result.0 & xy_to_bit(2, 0), 0); // Expect: A3 is NOT puttable
        }

        #[test]
        fn puttable_left_up_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 4), xy_to_bit(3, 3)); // Player:E5, Opponent:D4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn);

            assert_eq!(result.0, xy_to_bit(2, 2)); // Expect: C3 is puttable
        }

        #[test]
        fn puttable_left_up_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 1), xy_to_bit(3, 0)); // Player:B5, Opponent:A4
            let turn = DiskColor::Dark;

            let result = get_puttable_positions(&board, turn); // H3 is empty

            assert_eq!(result.0 & xy_to_bit(2, 7), 0); // Expect: H3 is NOT puttable
        }
    }

    mod tests_get_flip_positions {
        use super::*;

        const TURN: DiskColor = DiskColor::Dark;

        #[test]
        fn should_flip_one_stone_horizontally() {
            let board = Bitboard::default();
            let put_position = BitPosition(xy_to_bit(2, 3));
            let expected = xy_to_bit(3, 3);

            let result = get_flip_positions(&board, TURN, put_position);

            assert_eq!(result.0, expected);
        }

        #[test]
        fn should_flip_multiple_stones_diagonally() {
            let board = Bitboard::new(xy_to_bit(4, 4), xy_to_bit(2, 2) | xy_to_bit(3, 3));
            let put_position = BitPosition(xy_to_bit(1, 1));
            let expected = xy_to_bit(2, 2) | xy_to_bit(3, 3);

            let result = get_flip_positions(&board, TURN, put_position);

            assert_eq!(result.0, expected);
        }

        #[test]
        fn should_flip_stones_in_multiple_directions_simultaneously() {
            let board = Bitboard::new(
                xy_to_bit(0, 0) | xy_to_bit(0, 2),
                xy_to_bit(1, 1) | xy_to_bit(1, 2),
            );
            let put_position = BitPosition(xy_to_bit(2, 2));
            let expected = xy_to_bit(1, 1) | xy_to_bit(1, 2);

            let result = get_flip_positions(&board, TURN, put_position);

            assert_eq!(result.0, expected);
        }

        #[test]
        fn should_flip_no_stones_if_put_position_is_not_puttable() {
            let board = Bitboard::new(xy_to_bit(0, 0), xy_to_bit(1, 0));
            let put_position = BitPosition(xy_to_bit(3, 0));
            let expected = 0;

            let result = get_flip_positions(&board, TURN, put_position);

            assert_eq!(result.0, expected);
        }

        #[test]
        fn flip_right_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 2), xy_to_bit(3, 3)); // Player:C4, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(3, 4))); // Put:E4

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_right_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 6), xy_to_bit(3, 7)); // Player:G4, Opponent:H4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(4, 0))); // Put:A5

            assert_eq!(result.0, 0); // Expect: Nothing to flip
        }

        #[test]
        fn flip_left_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 4), xy_to_bit(3, 3)); // Player:E4, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(3, 2))); // Put:C4

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_left_wraparound() {
            let board = Bitboard::new(xy_to_bit(3, 1), xy_to_bit(3, 0)); // Player:B4, Opponent:A4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(2, 7))); // Put:H3

            assert_eq!(result.0, 0); // Expect: Nothing to flip
        }

        #[test]
        fn flip_down_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 3), xy_to_bit(3, 3)); // Player:D3, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(4, 3))); // Put:D5

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_down_edge_case() {
            let board = Bitboard::new(xy_to_bit(6, 3), xy_to_bit(7, 3)); // Player:D7, Opponent:D8

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(0, 3))); // Put:D1

            assert_eq!(result.0, 0);
        }

        #[test]
        fn flip_up_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 3), xy_to_bit(3, 3)); // Player:D5, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(2, 3))); // Put:D3

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_up_edge_case() {
            let board = Bitboard::new(xy_to_bit(1, 3), xy_to_bit(0, 3)); // Player:D2, Opponent:D1

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(7, 3))); // Put:D8

            assert_eq!(result.0, 0);
        }

        #[test]
        fn flip_right_down_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 2), xy_to_bit(3, 3)); // Player:C3, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(4, 4))); // Put:E5

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_right_down_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 6), xy_to_bit(3, 7)); // Player:G3, Opponent:H4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(4, 0))); // Put:A5

            assert_eq!(result.0, 0);
        }

        #[test]
        fn flip_left_down_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 4), xy_to_bit(3, 3)); // Player:E3, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(4, 2))); // Put:C5

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_left_down_wraparound() {
            let board = Bitboard::new(xy_to_bit(2, 1), xy_to_bit(3, 0)); // Player:B3, Opponent:A4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(4, 7))); // Put:H5

            assert_eq!(result.0, 0);
        }

        #[test]
        fn flip_right_up_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 2), xy_to_bit(3, 3)); // Player:C5, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(2, 4))); // Put:E3

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_right_up_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 6), xy_to_bit(3, 7)); // Player:G5, Opponent:H4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(2, 0))); // Put:A3

            assert_eq!(result.0, 0);
        }

        #[test]
        fn flip_left_up_no_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 4), xy_to_bit(3, 3)); // Player:E5, Opponent:D4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(2, 2))); // Put:C3

            assert_eq!(result.0, xy_to_bit(3, 3)); // Expect: D4
        }

        #[test]
        fn flip_left_up_wraparound() {
            let board = Bitboard::new(xy_to_bit(4, 1), xy_to_bit(3, 0)); // Player:B5, Opponent:A4

            let result = get_flip_positions(&board, TURN, BitPosition(xy_to_bit(2, 7))); // Put:H3

            assert_eq!(result.0, 0);
        }
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
            assert!(action.check_inputs(&table).is_err());

            // The square is not empty
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Four, Column::E),
            ));
            assert!(action.check_inputs(&table).is_err());
        }
        // Invalid turn
        {
            // Dark's turn but light try to put
            let mut table = Table::default();
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Light,
                position!(Row::Four, Column::F),
            ));
            assert!(action.check_inputs(&table).is_err());

            // Light's turn but dark try to put
            let board = table.board();
            let mask = xy_to_bit(Row::Five as u8, Column::E as u8)
                | xy_to_bit(Row::Five as u8, Column::F as u8);
            let new_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            table.set_board(new_board);
            table.set_turn(DiskColor::Light);
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Four, Column::C),
            ));
            assert!(action.check_inputs(&table).is_err());
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
            let new_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Light);
            assert!(action.check_inputs(&table).is_err());

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
            let new_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Dark);
            assert!(action.check_inputs(&table).is_err());
        }
        // Invalid pass
        {
            // Dark can put but try to pass
            let table = Table::default();
            let action = Action::PassTurn(DiskColor::Dark);
            assert!(action.check_inputs(&table).is_err());

            // Light can put but try to pass
            let mut table = Table::default();
            table.set_turn(DiskColor::Light);
            let board = table.board();
            let mask = xy_to_bit(Row::Five as u8, Column::E as u8)
                | xy_to_bit(Row::Five as u8, Column::F as u8);
            let new_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Light);
            assert!(action.check_inputs(&table).is_err());
        }

        // Valid put
        {
            let table = Table::default();
            let action = Action::PutDisk(PutConfig::new(
                DiskColor::Dark,
                position!(Row::Four, Column::C),
            ));
            assert!(action.check_inputs(&table).is_ok());
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
            let new_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            table.set_board(new_board);
            let action = Action::PassTurn(DiskColor::Light);
            assert!(action.check_inputs(&table).is_ok());
        }
    }

    #[test]
    fn test_execute() {
        let mut table = Table::default();
        let mut expected_board: Bitboard;

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
            expected_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
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
            expected_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
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
            expected_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
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
            expected_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
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
            expected_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
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
            expected_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
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
            expected_board = Bitboard::new(board.dark_plane() | mask, board.light_plane() & !mask);
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
            expected_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
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
            expected_board = Bitboard::new(board.dark_plane() & !mask, board.light_plane() | mask);
            assert_eq!(*board, expected_board);
            assert_eq!(table.turn(), DiskColor::Dark);
        }
    }
}
