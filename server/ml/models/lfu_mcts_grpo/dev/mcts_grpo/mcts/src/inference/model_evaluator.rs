//! Neural network model evaluation trait for MCTS.

use crate::game::{policy::PolicyEvaluation, state::State};

/// Trait for neural network model evaluation in MCTS.
///
/// Implementors of this trait provide the ability to evaluate game states
/// using a neural network model, returning both policy (move probabilities)
/// and value (position evaluation) predictions.
pub trait ModelEvaluator {
    /// Evaluates a batch of states and returns policy and value predictions.
    ///
    /// Takes a slice of game states and returns a vector of policy evaluations,
    /// where each evaluation contains a probability distribution over possible
    /// moves (policy) and an estimated value of the position.
    ///
    /// The batch processing allows for efficient GPU utilization when using
    /// neural network models for evaluation.
    fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation>;
}
