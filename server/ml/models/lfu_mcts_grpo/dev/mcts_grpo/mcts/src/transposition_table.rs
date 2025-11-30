//! Transposition table implementation for MCTS.
//!
//! A transposition table is a cache that stores neural network evaluations
//! for game states. Since the same state can be reached through different
//! move sequences, caching these evaluations avoids redundant neural network
//! inference calls and significantly improves MCTS performance.

use crate::{policy::PolicyEvaluation, state::State};
use std::{collections::HashMap, sync::RwLock};

/// A thread-safe cache for storing neural network evaluations of game states.
///
/// The transposition table maps game states to their policy and value evaluations,
/// allowing MCTS to reuse previously computed neural network predictions.
/// This is particularly important in batched MCTS where multiple trees may
/// encounter the same game positions.
///
/// Thread safety is provided through `RwLock`, allowing concurrent reads
/// while ensuring exclusive writes.
pub struct TranspositionTable {
    table: RwLock<HashMap<State, PolicyEvaluation>>,
}

impl TranspositionTable {
    /// Creates a new [`TranspositionTable`].
    pub fn new() -> Self {
        Self {
            table: RwLock::new(HashMap::new()),
        }
    }

    /// Stores a neural network evaluation for a game state.
    ///
    /// If an evaluation for this state already exists, it will be overwritten.
    /// This method is thread-safe and can be called concurrently from multiple
    /// MCTS trees.
    pub fn add(&self, state: State, policy_evaluation: PolicyEvaluation) {
        let mut table = self.table.write().unwrap();
        table.insert(state, policy_evaluation);
    }

    /// Retrieves a previously stored evaluation for a game state.
    ///
    /// Returns `None` if no evaluation has been stored for this state.
    /// This method acquires a read lock, allowing concurrent lookups
    /// from multiple threads.
    pub fn get(&self, state: &State) -> Option<PolicyEvaluation> {
        let table = self.table.read().unwrap();
        table.get(state).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, Value};
    use common::{Bitboard, DiskColor};

    fn default_state() -> State {
        State::new(Bitboard::default(), DiskColor::Dark)
    }

    fn default_policy_evaluation() -> PolicyEvaluation {
        PolicyEvaluation::new(Policy([0.0; 64]), Value(0.0))
    }

    #[test]
    fn test_new() {
        let table = TranspositionTable::new();
        assert!(table.table.read().unwrap().is_empty());
    }

    #[test]
    fn test_add() {
        let table = TranspositionTable::new();
        let policy_evaluation = default_policy_evaluation();
        table.add(default_state(), policy_evaluation);
        assert!(!table.table.read().unwrap().is_empty());
    }

    #[test]
    fn test_get() {
        let table = TranspositionTable::new();
        let policy_evaluation = default_policy_evaluation();
        table.add(default_state(), policy_evaluation);
        assert!(table.get(&default_state()).is_some());
    }
}
