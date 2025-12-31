use common::{Action, Bitboard, Column, DiskColor, Position, PutConfig, Row, Table};
use num_traits::FromPrimitive;
use reversi_core::action::get_puttable_positions;
use std::cmp::Ordering;

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
    pub fn legal_actions(&self) -> impl Iterator<Item = usize> + use<> {
        let puttable = get_puttable_positions(&self.board, self.turn);
        puttable.into_iter().map(usize::from)
    }

    /// Applies an action and returns the new state.
    ///
    /// If the next player has no legal moves after the action, a pass (turn skip)
    /// is automatically applied, matching the behavior of `ReversiEnvironment`.
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
        let step_result = Controller::step(&mut table, action).expect("Invalid action");

        // Auto-skip: if the next player has no legal moves, pass the turn
        if step_result.puttable_positions.is_empty() {
            let turn = table.turn();
            let _ = Controller::step(&mut table, Action::PassTurn(turn));
        }

        State {
            board: *table.board(),
            turn: table.turn(),
        }
    }

    /// Checks if the state is terminal.
    pub fn is_terminal(&self) -> bool {
        get_puttable_positions(&self.board, DiskColor::Dark).is_empty()
            && get_puttable_positions(&self.board, DiskColor::Light).is_empty()
    }

    /// Returns the terminal value for the current player.
    pub fn terminal_value(&self) -> Result<f64, String> {
        if !self.is_terminal() {
            return Err(String::from("State is not terminal"));
        }

        let dark_count = self.board.dark_plane().count_ones();
        let light_count = self.board.light_plane().count_ones();

        let winner_value = match dark_count.cmp(&light_count) {
            Ordering::Greater => 1.0, // Dark wins
            Ordering::Less => -1.0,   // Light wins
            Ordering::Equal => 0.0,
        };

        if self.turn == DiskColor::Dark {
            Ok(winner_value)
        } else {
            Ok(-winner_value)
        }
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
