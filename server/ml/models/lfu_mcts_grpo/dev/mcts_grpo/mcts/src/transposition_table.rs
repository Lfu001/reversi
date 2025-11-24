use crate::{policy::PolicyEvaluation, state::State};
use std::{collections::HashMap, sync::RwLock};

/// Transposition table for MCTS
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

    /// Adds a record.
    pub fn add(&self, state: State, policy_evaluation: PolicyEvaluation) {
        let mut table = self.table.write().unwrap();
        table.insert(state, policy_evaluation);
    }

    /// Returns a record.
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
