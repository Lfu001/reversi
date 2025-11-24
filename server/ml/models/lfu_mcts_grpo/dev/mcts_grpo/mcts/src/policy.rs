#[derive(Debug, Clone)]
pub struct Policy(pub [f64; 64]);

#[derive(Debug, Clone)]
pub struct Value(pub f64);

/// A pair of a policy and a value.
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
