use crate::position;
use crate::{
    position::{Column, Position, Row},
    DiskColor,
};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::ops::{Index, IndexMut};

/// A board of Reversi.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct Board {
    /// An 1D expression of the board.
    #[serde(with = "BigArray")]
    board: [Option<DiskColor>; 64],
}

impl Board {
    /// Returns a reference to the board of this [`Board`].
    pub fn board(&self) -> &[Option<DiskColor>; 64] {
        &self.board
    }
}

impl Default for Board {
    fn default() -> Self {
        let mut board = [None; 64];
        for (row, column, disk_color) in [
            (Row::Four, Column::D, DiskColor::Light),
            (Row::Four, Column::E, DiskColor::Dark),
            (Row::Five, Column::D, DiskColor::Dark),
            (Row::Five, Column::E, DiskColor::Light),
        ] {
            let idx: usize = position!(row, column).into();
            board[idx] = Some(disk_color);
        }
        Self { board }
    }
}

impl Index<Position> for Board {
    type Output = Option<DiskColor>;

    fn index(&self, index: Position) -> &Self::Output {
        let idx: usize = index.into();
        &self.board[idx]
    }
}

impl Index<usize> for Board {
    type Output = Option<DiskColor>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.board[index]
    }
}

impl IndexMut<usize> for Board {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.board[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_board() {
        let board = Board::default();

        assert_eq!(board[position!(Row::One, Column::A)], None);
        assert_eq!(board[position!(Row::Eight, Column::H)], None);
        assert_eq!(board[position!(Row::Three, Column::E)], None);

        assert_eq!(
            board[position!(Row::Four, Column::D)],
            Some(DiskColor::Light)
        );
        assert_eq!(
            board[position!(Row::Four, Column::E)],
            Some(DiskColor::Dark)
        );
        assert_eq!(
            board[position!(Row::Five, Column::D)],
            Some(DiskColor::Dark)
        );
        assert_eq!(
            board[position!(Row::Five, Column::E)],
            Some(DiskColor::Light)
        );
    }

    #[test]
    fn test_board_indexing() {
        // Index<Position>
        {
            let board = Board::default();

            assert_eq!(board[position!(Row::One, Column::A)], None);
            assert_eq!(board[position!(Row::Eight, Column::H)], None);
            assert_eq!(board[position!(Row::Three, Column::E)], None);

            assert_eq!(
                board[position!(Row::Four, Column::D)],
                Some(DiskColor::Light)
            );
            assert_eq!(
                board[position!(Row::Four, Column::E)],
                Some(DiskColor::Dark)
            );
            assert_eq!(
                board[position!(Row::Five, Column::D)],
                Some(DiskColor::Dark)
            );
            assert_eq!(
                board[position!(Row::Five, Column::E)],
                Some(DiskColor::Light)
            );
        }

        // Index<usize>, IndexMut<usize>
        {
            let mut board = Board::default();
            board[0] = Some(DiskColor::Light);
            board[63] = Some(DiskColor::Dark);

            assert_eq!(board[0], Some(DiskColor::Light));
            assert_eq!(board[63], Some(DiskColor::Dark));

            assert_eq!(board[3], None);
            assert_eq!(board[27], Some(DiskColor::Light));
        }
    }
}
