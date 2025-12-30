//! Node selection strategies for MCTS.

use crate::arena::{Arena, NodeId};
use crate::dirichlet::Dirichlet;
use crate::transposition_table::TranspositionTable;

/// Configuration for PUCT (Polynomial Upper Confidence Trees) calculation.
#[derive(Debug, Clone, Copy)]
pub struct PuctConfig {
    /// Constant to adjust exploration strength (typically 1.0 to 2.0).
    pub c_puct: f64,
    /// Epsilon parameter for Dirichlet noise mixing at root node.
    pub dirichlet_epsilon: f64,
    /// Alpha parameter for Dirichlet noise distribution.
    pub dirichlet_alpha: f64,
}

/// PUCT (Polynomial Upper Confidence Trees) selection strategy.
pub struct PuctStrategy {
    config: PuctConfig,
}

impl PuctStrategy {
    /// Creates a new [`PuctStrategy`] with the given `config` parameters.
    pub fn new(config: PuctConfig) -> Self {
        Self { config }
    }

    /// Returns the config.
    pub fn config(&self) -> &PuctConfig {
        &self.config
    }

    /// Calculates the PUCT score for a child node.
    ///
    /// Formula: PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))
    pub fn calculate_score(
        &self,
        q_value: f64,
        visit_count: u32,
        parent_visit_count: u32,
        prior_probability: f64,
    ) -> f64 {
        let exploration_term =
            self.config.c_puct * prior_probability * (parent_visit_count as f64).sqrt()
                / (1.0 + visit_count as f64);

        q_value + exploration_term
    }
}

/// Child selection with Dirichlet noise support.
pub struct ChildSelection {
    dirichlet_distribution: Dirichlet,
}

impl ChildSelection {
    /// Creates a new [`ChildSelection`] with the given `dirichlet_alpha` parameter.
    pub fn new(dirichlet_alpha: f64) -> Self {
        Self {
            dirichlet_distribution: Dirichlet::new(dirichlet_alpha, 33),
        }
    }

    /// Selects the best child node using PUCT strategy.
    ///
    /// Returns the `NodeId` of the best child, or `None` if no children.
    pub fn select_best_child(
        &mut self,
        arena: &Arena,
        parent_id: NodeId,
        strategy: &PuctStrategy,
        tt: &TranspositionTable,
        is_root: bool,
        rng: &mut impl rand::Rng,
    ) -> Option<NodeId> {
        let parent = arena.get(parent_id);
        let policy_evaluation = tt.get(parent.state())?;
        let parent_visit_count = parent.visit_count();
        let dirichlet_epsilon = strategy.config().dirichlet_epsilon;

        // Collect children
        let children: Vec<NodeId> = arena.children(parent_id).collect();
        if children.is_empty() {
            return None;
        }

        // Generate noise if at root
        let noise = if is_root {
            self.dirichlet_distribution.sample(rng, children.len())
        } else {
            None
        };

        // Calculate PUCT scores for each child
        children
            .iter()
            .enumerate()
            .map(|(idx, &child_id)| {
                let child = arena.get(child_id);
                let mut prior_probability = if let Some(action) = child.action() {
                    policy_evaluation.policy().0[action]
                } else {
                    0.0
                };

                if let Some(noise_vec) = &noise {
                    prior_probability = (1.0 - dirichlet_epsilon) * prior_probability
                        + dirichlet_epsilon * noise_vec[idx];
                }

                let q_value = child.q_value();
                let score = strategy.calculate_score(
                    q_value,
                    child.visit_count(),
                    parent_visit_count,
                    prior_probability,
                );
                (child_id, score)
            })
            .max_by(|(_, score_a), (_, score_b)| {
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(child_id, _)| child_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::policy::{Policy, PolicyEvaluation, Value};
    use crate::state::State;
    use common::{Bitboard, DiskColor};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn default_state() -> State {
        State::new(Bitboard::default(), DiskColor::Dark)
    }

    #[test]
    fn test_puct_score() {
        let config = PuctConfig {
            c_puct: 1.0,
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: 1.0,
        };
        let strategy = PuctStrategy::new(config);

        // Q(s,a) = 0.5, N(s,a) = 10, N(s) = 100, P(s,a) = 0.3
        // exploration = 1.0 * 0.3 * √100 / (1 + 10) ≈ 0.2727
        // PUCT ≈ 0.7727
        let score = strategy.calculate_score(0.5, 10, 100, 0.3);
        assert!((score - 0.7727).abs() < 0.01);
    }

    #[test]
    fn test_select_best_child() {
        let config = PuctConfig {
            c_puct: 1.0,
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: 1.0,
        };
        let strategy = PuctStrategy::new(config);
        let tt = TranspositionTable::new();
        let parent_state = default_state();

        let mut policy_arr = [0.0; 64];
        policy_arr[0] = 0.2;
        policy_arr[1] = 0.5;
        policy_arr[2] = 0.3;
        let policy = Policy(policy_arr);
        tt.add(parent_state, PolicyEvaluation::new(policy, Value(0.0)));

        // Build arena
        let mut arena = Arena::new();
        let parent_id = arena.allocate(Node::new(parent_state, None));

        // Set parent visit count by modifying node
        arena.get_mut(parent_id).increment_visit_count();
        for _ in 0..14 {
            arena.get_mut(parent_id).increment_visit_count();
        }

        // Add children
        let mut child0 = Node::new(default_state(), Some(0));
        for _ in 0..10 {
            child0.increment_visit_count();
            child0.add_evaluation(0.7);
        }
        let child0_id = arena.add_child(parent_id, child0);

        let mut child1 = Node::new(default_state(), Some(1));
        for _ in 0..5 {
            child1.increment_visit_count();
            child1.add_evaluation(0.4);
        }
        let child1_id = arena.add_child(parent_id, child1);

        let child2 = Node::new(default_state(), Some(2));
        let child2_id = arena.add_child(parent_id, child2);

        let mut rng = StdRng::seed_from_u64(42);
        let mut selection = ChildSelection::new(config.dirichlet_alpha);
        let best = selection.select_best_child(&arena, parent_id, &strategy, &tt, false, &mut rng);

        // Child 2 should be selected (highest exploration bonus due to 0 visits)
        assert_eq!(best, Some(child2_id));

        // Verify order: since we prepend children, child2 is first
        let _ = (child0_id, child1_id); // Suppress unused warnings
    }
}
