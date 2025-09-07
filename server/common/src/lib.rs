use num_derive::FromPrimitive;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::ops::{Index, IndexMut};

/// A color of the disk.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum DiskColor {
    /// A light side of the disk.
    Light,
    /// A dark side of the disk.
    Dark,
}

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

/// A board of Reversi.
/// TODO: implement Deref or Index trait
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

/// A state of a game table.
#[derive(Serialize, Deserialize, Clone)]
pub struct Table {
    /// A board of Reversi.
    #[serde(flatten)]
    board: Board,
    /// A color of the next turn.
    turn: DiskColor,
    /// A sequence of actions in the game history.
    history: Vec<Action>,
}

impl Table {
    /// Returns a reference to the board of this [`Table`].
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Returns a mutable reference to the board of this [`Table`].
    pub fn board_mut(&mut self) -> &mut Board {
        &mut self.board
    }

    /// Returns the turn of this [`Table`].
    pub fn turn(&self) -> DiskColor {
        self.turn
    }

    /// Sets the turn of this [`Table`].
    pub fn set_turn(&mut self, turn: DiskColor) {
        self.turn = turn;
    }

    /// Returns a reference to the history of this [`Table`].
    pub fn history(&self) -> &Vec<Action> {
        &self.history
    }

    /// Pushes an action to the history of this [`Table`].
    pub fn push_history(&mut self, action: Action) {
        self.history.push(action);
    }
}

impl Default for Table {
    fn default() -> Self {
        Self {
            board: Board::default(),
            turn: DiskColor::Dark,
            history: Vec::new(),
        }
    }
}

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

/// A request message to execute an action on the table.
#[derive(Serialize, Deserialize)]
pub struct StepRequestMessage {
    /// A table of the game.
    table: Table,
    /// An action to execute.
    action: Action,
}

impl StepRequestMessage {
    /// Creates a new [`StepRequestMessage`].
    pub fn new(table: Table, action: Action) -> Self {
        Self { table, action }
    }

    /// Returns a reference to the table of this [`StepRequestMessage`].
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Returns a reference to the action of this [`StepRequestMessage`].
    pub fn action(&self) -> &Action {
        &self.action
    }
}

/// A response message from the game endpoint.
#[derive(Serialize, Deserialize, Clone)]
pub struct StateResponseMessage {
    /// A table of the game.
    table: Table,
    /// Next puttable positions.
    puttable_positions: Vec<Position>,
    /// A result of the game.
    #[serde(skip_serializing_if = "Option::is_none")]
    judge_result: Option<JudgeResult>,
}

impl StateResponseMessage {
    /// Creates a new [`StateResponseMessage`].
    pub fn new(
        table: Table,
        puttable_positions: Vec<Position>,
        judge_result: Option<JudgeResult>,
    ) -> Self {
        Self {
            table,
            puttable_positions,
            judge_result,
        }
    }

    /// Returns a reference to the table of this [`StateResponseMessage`].
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Returns a reference to the puttable positions of this [`StateResponseMessage`].
    pub fn puttable_positions(&self) -> &Vec<Position> {
        &self.puttable_positions
    }

    /// Returns a reference to the judge result of this [`StateResponseMessage`].
    pub fn judge_result(&self) -> &Option<JudgeResult> {
        &self.judge_result
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

    #[test]
    fn test_push_history() {
        let mut table = Table::default();
        let action = Action::PutDisk(PutConfig::new(
            DiskColor::Dark,
            position!(Row::Four, Column::C),
        ));

        assert!(table.history().is_empty());

        table.push_history(action.clone());

        assert_eq!(table.history().len(), 1);
        assert_eq!(table.history()[0], action);
    }

    #[test]
    fn test_default_table() {
        let table = Table::default();

        assert_eq!(table.board(), &Board::default());
        assert_eq!(table.turn(), DiskColor::Dark);
        assert!(table.history().is_empty());
    }
}
