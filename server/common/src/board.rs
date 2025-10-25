use crate::position;
use crate::{
    position::{Column, Position, Row},
    DiskColor,
};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use serde_with::{serde_as, DisplayFromStr};
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

impl From<Bitboard> for Board {
    fn from(value: Bitboard) -> Self {
        let dark_plane = value.dark_plane();
        let light_plane = value.light_plane();
        let mut board = [None; 64];
        for (idx, disk_color) in board.iter_mut().enumerate() {
            let bit_mask = 1 << idx;
            if (dark_plane & bit_mask) != 0 {
                *disk_color = Some(DiskColor::Dark);
            } else if (light_plane & bit_mask) != 0 {
                *disk_color = Some(DiskColor::Light);
            }
        }
        Self { board }
    }
}

/// A compact representation of a Reversi board.
///
/// This is a more efficient form of a board for operations that do not require
/// direct access to individual positions.
///
/// Each bit represents a disk at the position. The most significant bit
/// represents the disk at the top-left. The least significant bit represents
/// the disk at the bottom-right.
///
/// For each disk color, if the bit is set to `1`, it means there is a disk of
/// the corresponding color at the corresponding position. If the bit is set to
/// `0`, it means there is no disk of the corresponding color at the
/// corresponding position.
///
/// # Examples
///
/// ```
/// use common::Bitboard;
///
/// let bitboard = Bitboard::default();
/// assert_eq!(bitboard.dark_plane(), 0b00000000_00000000_00000000_00001000_00010000_00000000_00000000_00000000);
/// assert_eq!(bitboard.light_plane(), 0b00000000_00000000_00000000_00010000_00001000_00000000_00000000_00000000);
/// ```
#[serde_as]
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Bitboard {
    /// A bit representation of dark disks.
    #[serde_as(as = "DisplayFromStr")]
    dark_plane: u64,
    /// A bit representation of light disks.
    #[serde_as(as = "DisplayFromStr")]
    light_plane: u64,
}

impl Bitboard {
    /// Creates a new [`BitBoard`].
    pub fn new(dark_plane: u64, light_plane: u64) -> Self {
        Self {
            dark_plane,
            light_plane,
        }
    }

    /// Returns the dark plane of this [`BitBoard`].
    pub fn dark_plane(&self) -> u64 {
        self.dark_plane
    }

    /// Returns the light plane of this [`BitBoard`].
    pub fn light_plane(&self) -> u64 {
        self.light_plane
    }
}

impl Default for Bitboard {
    fn default() -> Self {
        let idx_4d: usize = position!(Row::Four, Column::D).into();
        let idx_4e: usize = position!(Row::Four, Column::E).into();
        let idx_5d: usize = position!(Row::Five, Column::D).into();
        let idx_5e: usize = position!(Row::Five, Column::E).into();
        let dark_plane = (1u64 << idx_4e) | (1u64 << idx_5d);
        let light_plane = (1u64 << idx_4d) | (1u64 << idx_5e);
        Self::new(dark_plane, light_plane)
    }
}

impl From<Board> for Bitboard {
    fn from(value: Board) -> Self {
        let mut dark_plane = 0u64;
        let mut light_plane = 0u64;
        for idx in 0..64 {
            if value.board[idx] == Some(DiskColor::Dark) {
                dark_plane |= 1u64 << idx;
            } else if value.board[idx] == Some(DiskColor::Light) {
                light_plane |= 1u64 << idx;
            }
        }
        Self::new(dark_plane, light_plane)
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
    fn test_board_position_indexing() {
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
    fn test_board_usize_indexing() {
        let mut board = Board::default();
        board[0] = Some(DiskColor::Light);
        board[63] = Some(DiskColor::Dark);

        assert_eq!(board[0], Some(DiskColor::Light));
        assert_eq!(board[63], Some(DiskColor::Dark));

        assert_eq!(board[3], None);
        assert_eq!(board[27], Some(DiskColor::Light));
    }

    #[test]
    fn test_board_from_bitboard() {
        let bitboard = Bitboard::default();

        let board = Board::from(bitboard);

        for idx in 0..64 {
            if idx == position!(Row::Four, Column::D).into() {
                assert_eq!(board[idx], Some(DiskColor::Light));
            } else if idx == position!(Row::Four, Column::E).into() {
                assert_eq!(board[idx], Some(DiskColor::Dark));
            } else if idx == position!(Row::Five, Column::D).into() {
                assert_eq!(board[idx], Some(DiskColor::Dark));
            } else if idx == position!(Row::Five, Column::E).into() {
                assert_eq!(board[idx], Some(DiskColor::Light));
            } else {
                assert_eq!(board[idx], None);
            }
        }
    }

    #[test]
    fn test_bitboard_default() {
        let bitboard = Bitboard::default();
        assert_eq!(
            bitboard.dark_plane(),
            0b00000000_00000000_00000000_00001000_00010000_00000000_00000000_00000000
        );
        assert_eq!(
            bitboard.light_plane(),
            0b00000000_00000000_00000000_00010000_00001000_00000000_00000000_00000000
        );
    }

    #[test]
    fn test_bitboard_from_board() {
        let board = Board::default();

        let bitboard = Bitboard::from(board);

        for idx in 0..64usize {
            let bit_mask = 1 << idx;
            if idx == position!(Row::Four, Column::D).into() {
                assert_eq!(bitboard.dark_plane() & bit_mask, 0);
                assert_eq!(bitboard.light_plane() & bit_mask, bit_mask);
            } else if idx == position!(Row::Four, Column::E).into() {
                assert_eq!(bitboard.dark_plane() & bit_mask, bit_mask);
                assert_eq!(bitboard.light_plane() & bit_mask, 0);
            } else if idx == position!(Row::Five, Column::D).into() {
                assert_eq!(bitboard.dark_plane() & bit_mask, bit_mask);
                assert_eq!(bitboard.light_plane() & bit_mask, 0);
            } else if idx == position!(Row::Five, Column::E).into() {
                assert_eq!(bitboard.dark_plane() & bit_mask, 0);
                assert_eq!(bitboard.light_plane() & bit_mask, bit_mask);
            } else {
                assert_eq!(bitboard.dark_plane() & bit_mask, 0);
                assert_eq!(bitboard.light_plane() & bit_mask, 0);
            }
        }
    }
}
