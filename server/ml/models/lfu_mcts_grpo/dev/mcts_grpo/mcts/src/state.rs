use common::{Action, Bitboard, Column, DiskColor, Position, PutConfig, Row, Table};
use num_traits::FromPrimitive;
use reversi_core::action::get_puttable_positions;
use reversi_core::state::DiskColorExt;

/// State of the game.
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct State {
    /// A board of the game.
    board: Bitboard,
    /// A current turn color.
    turn: DiskColor,
}

impl State {
    /// Creates a new [`State`].
    pub fn new(board: Bitboard, turn: DiskColor) -> Self {
        Self { board, turn }
    }

    /// Returns the board.
    pub fn board(&self) -> &Bitboard {
        &self.board
    }

    /// Returns the current turn.
    pub fn turn(&self) -> DiskColor {
        self.turn
    }

    /// Returns a list of legal actions.
    pub fn legal_actions(&self) -> Vec<usize> {
        let puttable = get_puttable_positions(&self.board, self.turn);
        puttable.to_vec().iter().map(|p| usize::from(*p)).collect()
    }

    /// Applies an action and returns the new state.
    pub fn apply(&self, action_idx: usize) -> State {
        let mut table = Table::default();
        table.set_board(self.board);
        table.set_turn(self.turn);

        let action = {
            let row = Row::from_u8((action_idx / 8) as u8).unwrap();
            let col = Column::from_u8((action_idx % 8) as u8).unwrap();
            Action::PutDisk(PutConfig::new(self.turn, Position::new(row, col)))
        };

        // Use Controller.step() instead of direct execution
        use reversi_core::controller::Controller;
        let _ = Controller::step(&mut table, action);

        State {
            board: *table.board(),
            turn: table.turn(),
        }
    }

    /// Checks if the state is terminal.
    pub fn is_terminal(&self) -> bool {
        // Controller::is_game_over is private, so use legal_actions instead
        self.legal_actions().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let state = State::new(Bitboard::default(), DiskColor::Dark);
        assert_eq!(state.board, Bitboard::default());
        assert_eq!(state.turn, DiskColor::Dark);
    }
}
