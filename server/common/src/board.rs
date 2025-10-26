use crate::position;
use crate::position::{Column, Position, Row};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
