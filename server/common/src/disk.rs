use serde::{Deserialize, Serialize};

/// A color of the disk.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum DiskColor {
    /// A light side of the disk.
    Light,
    /// A dark side of the disk.
    Dark,
}
