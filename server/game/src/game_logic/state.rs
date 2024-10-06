/// A color of the disk.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DiskColor {
    /// A light side of the disk.
    Light,
    /// A dark side of the disk.
    Dark,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(u8)]
enum Row {
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(u8)]
enum Column {
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
#[derive(PartialEq, Debug)]
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
}

/// A macro to create a position.
///
/// # Arguments
///
/// * `$row` - A row of the board.
/// * `$column` - A column of the board.
macro_rules! position {
    ($row: expr, $column: expr) => {
        Position {
            row: $row,
            column: $column,
        }
    };
}

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
    pub fn get_disk(&self, position: Position) -> Option<DiskColor> {
        self.board[position.to_index() as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_new_board() {
        let board = Board::new();

        assert_eq!(board.get_disk(position!(Row::One, Column::A)), None);
        assert_eq!(board.get_disk(position!(Row::Eight, Column::H)), None);
        assert_eq!(board.get_disk(position!(Row::Three, Column::E)), None);

        assert_eq!(
            board.get_disk(position!(Row::Four, Column::D)),
            Some(DiskColor::Light)
        );
        assert_eq!(
            board.get_disk(position!(Row::Four, Column::E)),
            Some(DiskColor::Dark)
        );
        assert_eq!(
            board.get_disk(position!(Row::Five, Column::D)),
            Some(DiskColor::Dark)
        );
        assert_eq!(
            board.get_disk(position!(Row::Five, Column::E)),
            Some(DiskColor::Light)
        );
    }
}
