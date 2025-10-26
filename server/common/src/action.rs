use crate::{disk::DiskColor, position::Position};
use serde::{Deserialize, Serialize};

/// A configuration of the put action.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct PutConfig {
    /// A color of the disk.
    color: DiskColor,
    /// A position of the square in the board.
    position: Position,
}

impl PutConfig {
    /// Creates a new [`PutConfig`].
    pub fn new(color: DiskColor, position: Position) -> Self {
        Self { color, position }
    }

    /// Returns the color of this [`PutConfig`].
    pub fn color(&self) -> DiskColor {
        self.color
    }

    /// Returns a reference to the position of this [`PutConfig`].
    pub fn position(&self) -> &Position {
        &self.position
    }
}

/// Reversi actions.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum Action {
    /// Put a disk to the square.
    PutDisk(PutConfig),
    /// Pass the turn.
    PassTurn(DiskColor),
}
