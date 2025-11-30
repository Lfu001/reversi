use crate::{
    dirichlet::sample_dirichlet, node::Node, state::State, transposition_table::TranspositionTable,
};
use std::{cell::RefCell, rc::Rc};

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

/// Trait for node selection strategies in MCTS.
pub trait SelectionStrategy {
    fn calculate_score(&self, node: &Node, parent_visit_count: u32, prior_probability: f64) -> f64;
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

    /// Returns the config
    pub fn config(&self) -> &PuctConfig {
        &self.config
    }
}

impl SelectionStrategy for PuctStrategy {
    /// Calculates the PUCT score for a node.
    ///
    /// Evaluates the given `node` using its Q-value and exploration term, considering
    /// the `parent_visit_count` and action's `prior_probability` from the neural network policy.
    /// Formula: PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))
    fn calculate_score(&self, node: &Node, parent_visit_count: u32, prior_probability: f64) -> f64 {
        let q_value = node.average_value();
        let exploration_term =
            self.config.c_puct * prior_probability * (parent_visit_count as f64).sqrt()
                / (1.0 + node.visit_count() as f64);

        q_value + exploration_term
    }
}

/// Selects the best child node from a list of candidates using PUCT strategy.
///
/// Evaluates all `children` nodes considering the `parent_visit_count`, applies the PUCT
/// selection `strategy`, retrieves policy priors from `transposition_table` for `parent_state`,
/// optionally adds Dirichlet noise if `is_root` is true, and uses `rng` for noise generation.
/// Returns the index of the child with the highest PUCT score.
pub fn select_best_child(
    children: &[Rc<RefCell<Node>>],
    parent_visit_count: u32,
    strategy: &PuctStrategy,
    transposition_table: &TranspositionTable,
    parent_state: &State,
    is_root: bool,
    rng: &mut impl rand::Rng,
) -> Option<usize> {
    let policy_evaluation = transposition_table.get(parent_state)?;

    let dirichlet_epsilon = strategy.config().dirichlet_epsilon;
    let dirichlet_alpha = strategy.config().dirichlet_alpha;

    let noise = if is_root && !children.is_empty() {
        sample_dirichlet(rng, dirichlet_alpha, children.len())
    } else {
        None
    };

    children
        .iter()
        .enumerate()
        .map(|(idx, child)| {
            let child_ref = child.borrow();
            let mut prior_probability = if let Some(action) = child_ref.action() {
                policy_evaluation.policy().0[action]
            } else {
                0.0
            };

            if let Some(noise_vec) = &noise {
                prior_probability = (1.0 - dirichlet_epsilon) * prior_probability
                    + dirichlet_epsilon * noise_vec[idx];
            }

            let score = strategy.calculate_score(&child_ref, parent_visit_count, prior_probability);
            (idx, score)
        })
        .max_by(|(_, score_a), (_, score_b)| {
            score_a
                .partial_cmp(score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::policy::{Policy, PolicyEvaluation, Value};
    use common::{Bitboard, DiskColor};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn default_state() -> State {
        State::new(Bitboard::default(), DiskColor::Dark)
    }

    fn create_node(visit_count: u32, total_value: f64, action: Option<usize>) -> Rc<RefCell<Node>> {
        let node = Node::new(default_state(), action);
        node.borrow_mut().set_visit_count(visit_count);
        let sum = total_value * visit_count as f64;
        let current_sum = node.borrow().sum_evaluation();
        node.borrow_mut().add_evaluation(sum - current_sum); // Set to target value
        node
    }

    #[test]
    fn test_puct_basic() {
        let config = PuctConfig {
            c_puct: 1.0,
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: 1.0,
        };
        let strategy = PuctStrategy::new(config);
        let node = create_node(10, 0.5, Some(0));
        let prior_probability = 0.3;

        let score = strategy.calculate_score(&node.borrow(), 100, prior_probability);

        // Q(s,a) = 0.5
        // exploration = 1.0 * 0.3 * √100 / (1 + 10) ≈ 0.2727
        // PUCT ≈ 0.7727
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
        let transposition_table = TranspositionTable::new();
        let parent_state = default_state();

        let mut policy_arr = [0.0; 64];
        policy_arr[0] = 0.2;
        policy_arr[1] = 0.5;
        policy_arr[2] = 0.3;
        let policy = Policy(policy_arr);
        let value = Value(0.0);
        transposition_table.add(parent_state.clone(), PolicyEvaluation::new(policy, value));

        let children = vec![
            create_node(10, 0.7, Some(0)), // P=0.2, Q=0.7, N=10
            create_node(5, 0.4, Some(1)),  // P=0.5, Q=0.4, N=5
            create_node(0, 0.0, Some(2)),  // P=0.3, Q=0.0, N=0
        ];

        let mut rng = StdRng::seed_from_u64(42);
        let best = select_best_child(
            &children,
            15,
            &strategy,
            &transposition_table,
            &parent_state,
            false,
            &mut rng,
        );

        // Child 0: 0.7 + 1.0 * 0.2 * sqrt(15) / 11 = 0.7 + 0.07 = 0.77
        // Child 1: 0.4 + 1.0 * 0.5 * sqrt(15) / 6 = 0.4 + 0.32 = 0.72
        // Child 2: 0.0 + 1.0 * 0.3 * sqrt(15) / 1 = 1.16

        // Child 2 should be selected (highest exploration bonus due to 0 visits)
        assert_eq!(best, Some(2));
    }

    #[test]
    fn test_node_update() {
        let node = Node::new(default_state(), None);
        assert_eq!(node.borrow().average_value(), 0.0);

        // Manually update for test since update method might not exist or be different
        node.borrow_mut().increment_visit_count();
        node.borrow_mut().add_evaluation(1.0);
        assert_eq!(node.borrow().visit_count(), 1);
        assert_eq!(node.borrow().average_value(), 1.0);

        node.borrow_mut().increment_visit_count();
        node.borrow_mut().add_evaluation(0.0);
        assert_eq!(node.borrow().visit_count(), 2);
        assert_eq!(node.borrow().average_value(), 0.5);
    }
}
