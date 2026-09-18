//! Transposition table implementation for MCTS.
//!
//! A transposition table is a cache that stores neural network evaluations
//! for game states. Since the same state can be reached through different
//! move sequences, caching these evaluations avoids redundant neural network
//! inference calls and significantly improves MCTS performance.

use crate::game::{policy::PolicyEvaluation, state::State};
use dashmap::DashMap;

/// A thread-safe cache for storing neural network evaluations of game states.
///
/// The transposition table maps game states to their policy and value evaluations,
/// allowing MCTS to reuse previously computed neural network predictions.
/// This is particularly important in batched MCTS where multiple trees may
/// encounter the same game positions.
///
/// Thread safety is provided through `DashMap` (sharded concurrent hash map),
/// which significantly reduces lock contention compared to `RwLock<HashMap>`.
pub struct TranspositionTable {
    table: DashMap<State, PolicyEvaluation>,
}

impl TranspositionTable {
    /// Creates a new [`TranspositionTable`].
    pub fn new() -> Self {
        Self {
            table: DashMap::new(),
        }
    }

    /// Stores a neural network evaluation for a game state.
    ///
    /// If an evaluation for this state already exists, it will be overwritten.
    /// This method is thread-safe and can be called concurrently from multiple
    /// MCTS trees.
    pub fn add(&self, state: State, policy_evaluation: PolicyEvaluation) {
        self.table.insert(state, policy_evaluation);
    }

    /// Retrieves a previously stored evaluation for a game state.
    ///
    /// Returns `None` if no evaluation has been stored for this state.
    pub fn get(
        &self,
        state: &State,
    ) -> Option<dashmap::mapref::one::Ref<'_, State, PolicyEvaluation>> {
        // DashMap returns a Ref, but we need owned data (PolicyEvaluation is small/cloneable)
        self.table.get(state)
    }

    /// Attempts to reserve a state for inference atomically.
    ///
    /// This method prevents duplicate inference requests by using an atomic
    /// check-and-insert pattern. When multiple threads encounter the same
    /// unexplored state, only one will successfully reserve it.
    ///
    /// Returns:
    /// - `Reserved` if this thread successfully reserved the state (should queue for inference)
    /// - `AlreadyReserved` if another thread has reserved it but inference is pending
    /// - `AlreadyEvaluated(value)` if inference result is already available
    pub fn try_reserve(&self, state: State) -> ReserveResult {
        use dashmap::mapref::entry::Entry;

        match self.table.entry(state) {
            Entry::Occupied(entry) => {
                let eval = entry.get();
                if eval.is_pending() {
                    ReserveResult::AlreadyReserved
                } else {
                    ReserveResult::AlreadyEvaluated(eval.value())
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(PolicyEvaluation::pending());
                ReserveResult::Reserved
            }
        }
    }
}

/// Result of attempting to reserve a state in the transposition table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReserveResult {
    /// This thread successfully reserved the state for inference.
    Reserved,
    /// Another thread has already reserved this state (inference pending).
    AlreadyReserved,
    /// The state has already been evaluated (value available).
    AlreadyEvaluated(f64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::policy::{Policy, Value};
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
        assert!(table.table.is_empty());
    }

    #[test]
    fn test_add() {
        let table = TranspositionTable::new();
        let policy_evaluation = default_policy_evaluation();
        table.add(default_state(), policy_evaluation);
        assert!(!table.table.is_empty());
    }

    #[test]
    fn test_get() {
        let table = TranspositionTable::new();
        let policy_evaluation = default_policy_evaluation();
        table.add(default_state(), policy_evaluation);
        assert!(table.get(&default_state()).is_some());
    }

    #[test]
    fn test_try_reserve_new_state() {
        let table = TranspositionTable::new();
        let state = default_state();

        // First reservation should succeed
        let result = table.try_reserve(state);
        assert_eq!(result, ReserveResult::Reserved);

        // State should now be in table with pending evaluation
        let eval = table.get(&state).unwrap();
        assert!(eval.is_pending());
    }

    #[test]
    fn test_try_reserve_already_reserved() {
        let table = TranspositionTable::new();
        let state = default_state();

        // First reservation succeeds
        assert_eq!(table.try_reserve(state), ReserveResult::Reserved);

        // Second reservation returns AlreadyReserved (still pending)
        assert_eq!(table.try_reserve(state), ReserveResult::AlreadyReserved);
    }

    #[test]
    fn test_try_reserve_already_evaluated() {
        let table = TranspositionTable::new();
        let state = default_state();

        // Add a completed evaluation
        let eval = PolicyEvaluation::new(Policy([0.0; 64]), Value(0.5));
        table.add(state, eval);

        // try_reserve should return AlreadyEvaluated with the value
        match table.try_reserve(state) {
            ReserveResult::AlreadyEvaluated(v) => assert!((v - 0.5).abs() < 1e-10),
            _ => panic!("Expected AlreadyEvaluated"),
        }
    }

    #[test]
    fn test_pending_overwritten_by_add() {
        let table = TranspositionTable::new();
        let state = default_state();

        // Reserve the state (creates pending entry)
        assert_eq!(table.try_reserve(state), ReserveResult::Reserved);
        assert!(table.get(&state).unwrap().is_pending());

        // Add actual evaluation (overwrites pending)
        let eval = PolicyEvaluation::new(Policy([0.0; 64]), Value(0.7));
        table.add(state, eval);

        // Now should return AlreadyEvaluated
        match table.try_reserve(state) {
            ReserveResult::AlreadyEvaluated(v) => assert!((v - 0.7).abs() < 1e-10),
            _ => panic!("Expected AlreadyEvaluated after add"),
        }
    }
}
