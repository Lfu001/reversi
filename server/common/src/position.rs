use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, PartialEq, Debug, FromPrimitive, Serialize, Deserialize)]
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

#[derive(Copy, Clone, PartialEq, Debug, FromPrimitive, Serialize, Deserialize)]
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
#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Position {
    /// A row of the board.
    row: Row,
    /// A column of the board.
    column: Column,
}

impl Position {
    /// Creates a new [`Position`].
    pub fn new(row: Row, column: Column) -> Self {
        Self { row, column }
    }

    /// Returns the row of this [`Position`].
    pub fn row(&self) -> Row {
        self.row
    }

    /// Returns the column of this [`Position`].
    pub fn column(&self) -> Column {
        self.column
    }
}

impl From<(Row, Column)> for Position {
    fn from(value: (Row, Column)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<BitPosition> for Position {
    fn from(bit_position: BitPosition) -> Self {
        let idx = 63 - bit_position.0.trailing_zeros() as usize;
        Position::from((
            Row::from_u8((idx / 8) as u8).unwrap(),
            Column::from_u8((idx % 8) as u8).unwrap(),
        ))
    }
}

impl From<Position> for usize {
    fn from(value: Position) -> Self {
        (value.row() as usize) * 8 + (value.column() as usize)
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
        Position::new($row, $column)
    };
}

#[derive(Clone)]
pub struct BitPosition(pub u64);

impl From<Position> for BitPosition {
    fn from(position: Position) -> Self {
        let idx: usize = position.into();
        BitPosition(1u64 << (63 - idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_new() {
        let row = Row::Four;
        let column = Column::D;
        let position = Position::new(row, column);
        assert_eq!(position.row(), row);
        assert_eq!(position.column(), column);
    }

    #[test]
    fn test_position_from_row_column() {
        let row = Row::Four;
        let column = Column::D;
        {
            let position: Position = (row, column).into();
            assert_eq!(position.row(), row);
            assert_eq!(position.column(), column);
        }
        {
            let position = Position::from((row, column));
            assert_eq!(position.row(), row);
            assert_eq!(position.column(), column);
        }
    }

    #[test]
    fn test_position_from_bit_position() {
        let bit_position = BitPosition(1u64 << (63 - (8 * 3 + 3)));
        let position = Position::from(bit_position);
        assert_eq!(position.row(), Row::Four);
        assert_eq!(position.column(), Column::D);
    }

    #[test]
    fn test_usize_from_position() {
        for (row, column, expected) in [
            (Row::One, Column::A, 0),
            (Row::Eight, Column::H, 63),
            (Row::One, Column::H, 7),
            (Row::Three, Column::E, 20),
        ] {
            {
                let idx: usize = position!(row, column).into();
                assert_eq!(idx, expected);
            }
            {
                let idx = usize::from(position!(row, column));
                assert_eq!(idx, expected);
            }
        }
    }

    #[test]
    fn test_position_macro() {
        assert_eq!(
            position!(Row::One, Column::A),
            Position::new(Row::One, Column::A)
        );
        assert_eq!(
            position!(Row::Eight, Column::H),
            Position::new(Row::Eight, Column::H)
        );
    }

    #[test]
    fn test_bit_position_from_position() {
        let position = Position::new(Row::Four, Column::D);
        let bit_position = BitPosition::from(position);
        assert_eq!(bit_position.0, 1u64 << (63 - (8 * 3 + 3)));
    }
}
