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
}
