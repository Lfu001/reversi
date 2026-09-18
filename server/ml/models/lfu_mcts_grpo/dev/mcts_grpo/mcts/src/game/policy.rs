/// A probability distribution over all 64 possible board positions.
///
/// Represents the policy output from a neural network model, where each
/// element is the probability of placing a disk at that position.
/// The 64 positions correspond to an 8x8 Reversi board, indexed row-major.
#[derive(Debug, Clone)]
pub struct Policy(pub [f64; 64]);

/// The estimated value of a game state from the current player's perspective.
///
/// Represents the value output from a neural network model, typically in
/// the range \[-1.0, 1.0\], where positive values favor the current player
/// and negative values favor the opponent.
#[derive(Debug, Clone)]
pub struct Value(pub f64);

/// A combined policy and value evaluation from a neural network model.
///
/// This struct pairs the move probability distribution (policy) with
/// the position evaluation (value), representing a complete neural network
/// prediction for a game state.
#[derive(Debug, Clone)]
pub struct PolicyEvaluation {
    /// A policy.
    policy: Policy,
    /// A value.
    value: Value,
}

impl PolicyEvaluation {
    /// Creates a new [`PolicyEvaluation`].
    pub fn new(policy: Policy, value: Value) -> Self {
        Self { policy, value }
    }

    /// Creates a pending evaluation used as a placeholder in the transposition table.
    ///
    /// A pending evaluation indicates that another thread has reserved this state
    /// for inference but the result is not yet available. The value is set to NaN
    /// as a sentinel value.
    pub fn pending() -> Self {
        Self {
            policy: Policy([0.0; 64]),
            value: Value(f64::NAN),
        }
    }

    /// Returns true if this evaluation is pending (awaiting inference).
    pub fn is_pending(&self) -> bool {
        self.value.0.is_nan()
    }

    /// Returns the policy.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Returns the value.
    pub fn value(&self) -> f64 {
        self.value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let policy = Policy([0.0; 64]);
        let value = Value(0.0);
        let policy_evaluation = PolicyEvaluation::new(policy, value);
        assert_eq!(policy_evaluation.policy.0, [0.0; 64]);
        assert_eq!(policy_evaluation.value.0, 0.0);
    }

    #[test]
    fn test_pending() {
        let pending = PolicyEvaluation::pending();
        assert!(pending.is_pending());
        assert!(pending.value().is_nan());
    }

    #[test]
    fn test_is_pending_false_for_normal() {
        let normal = PolicyEvaluation::new(Policy([0.0; 64]), Value(0.5));
        assert!(!normal.is_pending());
    }
}
