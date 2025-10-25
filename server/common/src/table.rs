use crate::{action::Action, board::BitBoard, disk::DiskColor};
use serde::{Deserialize, Serialize};

/// A state of a game table.
#[derive(Serialize, Deserialize, Clone)]
pub struct Table {
    /// A board of Reversi.
    board: BitBoard,
    /// A color of the next turn.
    turn: DiskColor,
    /// A sequence of actions in the game history.
    history: Vec<Action>,
}

impl Table {
    /// Returns a reference to the board of this [`Table`].
    pub fn board(&self) -> &BitBoard {
        &self.board
    }

    /// Sets the board of this [`Table`].
    pub fn set_board(&mut self, board: BitBoard) {
        self.board = board;
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
            board: BitBoard::default(),
            turn: DiskColor::Dark,
            history: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::PutConfig;
    use crate::position::{Column, Position, Row};

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

        assert_eq!(table.board(), &BitBoard::default());
        assert_eq!(table.turn(), DiskColor::Dark);
        assert!(table.history().is_empty());
    }
}
