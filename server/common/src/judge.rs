use crate::disk::DiskColor;
use serde::{Deserialize, Serialize};

/// A judge of the game.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum Winner {
    /// A winner.
    Win(DiskColor),
    /// Draw result.
    Draw,
}

/// A result of the game.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct JudgeResult {
    /// A number of dark disks on the board.
    dark_count: u8,
    /// A number of light disks on the board.
    light_count: u8,
    /// A winner of the game.
    winner: Winner,
}

impl JudgeResult {
    /// Creates a new [`JudgeResult`].
    pub fn new(dark_count: u8, light_count: u8, winner: Winner) -> Self {
        Self {
            dark_count,
            light_count,
            winner,
        }
    }

    /// Returns the winner of this [`JudgeResult`].
    pub fn winner(&self) -> Winner {
        self.winner
    }
}
