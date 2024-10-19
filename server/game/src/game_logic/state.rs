use crate::position;
use num_traits::FromPrimitive;

use super::action::Action;

/// A color of the disk.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DiskColor {
    /// A light side of the disk.
    Light,
    /// A dark side of the disk.
    Dark,
}

impl DiskColor {
    /// Returns the opposite color.
    pub fn opposite(&self) -> DiskColor {
        match self {
            DiskColor::Light => DiskColor::Dark,
            DiskColor::Dark => DiskColor::Light,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, FromPrimitive)]
#[repr(u8)]
pub enum Row {
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
}

#[derive(Copy, Clone, PartialEq, Debug, FromPrimitive)]
#[repr(u8)]
pub enum Column {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}

/// A position of the square in the board.
#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Position {
    /// A row of the board.
    row: Row,
    /// A column of the board.
    column: Column,
}

impl Position {
    /// Converts a position to a 1D index.
    fn to_index(&self) -> usize {
        (self.row as usize) * 8 + (self.column as usize)
    }
    /// Constructs a position from a 1D index.
    ///
    /// # Arguments
    ///
    /// * `index` - A 1D index of the board.
    fn from_index(index: usize) -> Self {
        Position {
            row: FromPrimitive::from_usize(index / 8).unwrap(),
            column: FromPrimitive::from_usize(index % 8).unwrap(),
        }
    }
    /// Constructs a position from a row and a column.
    pub fn from_row_and_column(row: Row, column: Column) -> Self {
        Position { row, column }
    }
}

/// A macro to create a position.
///
/// # Arguments
///
/// * `$row` - A row of the board.
/// * `$column` - A column of the board.
#[macro_export]
macro_rules! position {
    ($row: expr, $column: expr) => {
        $crate::game_logic::state::Position::from_row_and_column($row, $column)
    };
}

const DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),   // right
    (1, -1),  // right up
    (0, -1),  // up
    (-1, -1), // left up
    (-1, 0),  // left
    (-1, 1),  // left down
    (0, 1),   // down
    (1, 1),   // right down
];

/// A board of Reversi.
pub struct Board {
    /// An 1D expression of the board.
    board: [Option<DiskColor>; 64],
}

impl Board {
    /// Creates an new board.
    pub fn new() -> Self {
        let mut board = Board { board: [None; 64] };
        board.set_disk(position!(Row::Four, Column::D), DiskColor::Light);
        board.set_disk(position!(Row::Four, Column::E), DiskColor::Dark);
        board.set_disk(position!(Row::Five, Column::D), DiskColor::Dark);
        board.set_disk(position!(Row::Five, Column::E), DiskColor::Light);
        board
    }

    /// Sets a disk to the specified position. This operation does not consider reversi rules.
    ///
    /// # Arguments
    ///
    /// * `position` - A position of the square in the board.
    /// * `color` - A color of the disk.
    pub fn set_disk(&mut self, position: Position, color: DiskColor) {
        self.board[position.to_index() as usize] = Some(color);
    }

    /// Gets a disk from the specified position.
    ///
    /// # Arguments
    ///
    /// * `position` - A position of the square in the board.
    pub fn get_disk(&self, position: &Position) -> Option<DiskColor> {
        self.board[position.to_index() as usize]
    }

    /// Gets a read-only reference to raw expression of the board.
    pub fn raw_board(&self) -> &[Option<DiskColor>; 64] {
        &self.board
    }

    /// Gets all the rays from the specified position.
    pub fn get_rays(&self, position: &Position) -> [Vec<(Option<DiskColor>, Position)>; 8] {
        let mut rays = vec![vec![]; 8];
        for (i, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
            let mut column = position.column as i8 + dx;
            let mut row = position.row as i8 + dy;
            while column >= 0 && column < 8 && row >= 0 && row < 8 {
                let position = position!(
                    FromPrimitive::from_i8(row).unwrap(),
                    FromPrimitive::from_i8(column).unwrap()
                );
                let disk = self.get_disk(&position);
                rays[i].push((disk.to_owned(), position));
                column += dx;
                row += dy;
            }
        }
        rays.try_into().unwrap()
    }
}

/// A state of a game table.
pub struct Table {
    /// A board of Reversi.
    board: Board,
    /// A color of the next turn.
    turn: DiskColor,
    /// A sequence of actions in the game history.
    history: Vec<Action>,
}

impl Table {
    /// Creates a new table.
    pub fn new() -> Self {
        Table {
            board: Board::new(),
            turn: DiskColor::Dark,
            history: vec![],
        }
    }

    /// Returns a writable reference to the board.
    pub fn board_mut(&mut self) -> &mut Board {
        &mut self.board
    }

    /// Returns a read-only reference to the board.
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Returns a writable reference to the color of the next turn.
    pub fn set_turn(&mut self, turn: DiskColor) {
        self.turn = turn;
    }

    /// Returns a read-only reference to the color of the next turn.
    pub fn turn(&self) -> &DiskColor {
        &self.turn
    }

    /// Returns a writable reference to the history.
    pub fn history_mut(&mut self) -> &mut Vec<Action> {
        &mut self.history
    }

    /// Returns a read-only reference to the history.
    pub fn history(&self) -> &Vec<Action> {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_color() {
        assert_eq!(DiskColor::Dark.opposite(), DiskColor::Light);
        assert_eq!(DiskColor::Light.opposite(), DiskColor::Dark);
    }

    #[test]
    fn test_position_macro() {
        assert_eq!(
            position!(Row::One, Column::A),
            Position {
                row: Row::One,
                column: Column::A
            }
        );
        assert_eq!(
            position!(Row::Eight, Column::H),
            Position {
                row: Row::Eight,
                column: Column::H
            }
        );
    }

    #[test]
    fn test_to_index() {
        assert_eq!(position!(Row::One, Column::A).to_index(), 0);
        assert_eq!(position!(Row::Eight, Column::H).to_index(), 63);
        assert_eq!(position!(Row::One, Column::H).to_index(), 7);
        assert_eq!(position!(Row::Three, Column::E).to_index(), 20);
    }

    #[test]
    fn test_from_index() {
        assert_eq!(Position::from_index(0), position!(Row::One, Column::A));
        assert_eq!(Position::from_index(63), position!(Row::Eight, Column::H));
        assert_eq!(Position::from_index(7), position!(Row::One, Column::H));
        assert_eq!(Position::from_index(20), position!(Row::Three, Column::E));
    }

    #[test]
    fn test_new_board() {
        let board = Board::new();

        assert_eq!(board.get_disk(&position!(Row::One, Column::A)), None);
        assert_eq!(board.get_disk(&position!(Row::Eight, Column::H)), None);
        assert_eq!(board.get_disk(&position!(Row::Three, Column::E)), None);

        assert_eq!(
            board.get_disk(&position!(Row::Four, Column::D)),
            Some(DiskColor::Light)
        );
        assert_eq!(
            board.get_disk(&position!(Row::Four, Column::E)),
            Some(DiskColor::Dark)
        );
        assert_eq!(
            board.get_disk(&position!(Row::Five, Column::D)),
            Some(DiskColor::Dark)
        );
        assert_eq!(
            board.get_disk(&position!(Row::Five, Column::E)),
            Some(DiskColor::Light)
        );
    }

    #[test]
    fn test_get_rays() {
        let board = Board::new();
        let rays = board.get_rays(&position!(Row::Four, Column::C));

        // right
        assert_eq!(
            rays[0],
            vec![
                (Some(DiskColor::Light), position!(Row::Four, Column::D)),
                (Some(DiskColor::Dark), position!(Row::Four, Column::E)),
                (None, position!(Row::Four, Column::F)),
                (None, position!(Row::Four, Column::G)),
                (None, position!(Row::Four, Column::H))
            ]
        );
        // right up
        assert_eq!(
            rays[1],
            vec![
                (None, position!(Row::Three, Column::D)),
                (None, position!(Row::Two, Column::E)),
                (None, position!(Row::One, Column::F)),
            ]
        );
        // up
        assert_eq!(
            rays[2],
            vec![
                (None, position!(Row::Three, Column::C)),
                (None, position!(Row::Two, Column::C)),
                (None, position!(Row::One, Column::C)),
            ]
        );
        // left up
        assert_eq!(
            rays[3],
            vec![
                (None, position!(Row::Three, Column::B)),
                (None, position!(Row::Two, Column::A)),
            ]
        );
        // left
        assert_eq!(
            rays[4],
            vec![
                (None, position!(Row::Four, Column::B)),
                (None, position!(Row::Four, Column::A)),
            ]
        );
        // left down
        assert_eq!(
            rays[5],
            vec![
                (None, position!(Row::Five, Column::B)),
                (None, position!(Row::Six, Column::A)),
            ]
        );
        // down
        assert_eq!(
            rays[6],
            vec![
                (None, position!(Row::Five, Column::C)),
                (None, position!(Row::Six, Column::C)),
                (None, position!(Row::Seven, Column::C)),
                (None, position!(Row::Eight, Column::C)),
            ]
        );
        // right down
        assert_eq!(
            rays[7],
            vec![
                (Some(DiskColor::Dark), position!(Row::Five, Column::D)),
                (None, position!(Row::Six, Column::E)),
                (None, position!(Row::Seven, Column::F)),
                (None, position!(Row::Eight, Column::G)),
            ]
        );
    }
}
