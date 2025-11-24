pub mod queue;
pub mod worker;

use crate::policy::PolicyEvaluation;
use crate::state::State;

/// Trait for model evaluation.
pub trait ModelEvaluator {
    /// Evaluates a batch of states.
    fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation>;
}
