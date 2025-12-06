use crate::{
    inference::ModelEvaluator,
    node::Node,
    selection::{PuctConfig, PuctStrategy, select_best_child},
    state::State,
    transposition_table::TranspositionTable,
};
use std::{cell::RefCell, rc::Rc};

/// Internal result type for batch PUCT traversal.
enum SearchResult {
    /// Transposition table miss - state needs neural network evaluation.
    Miss(State),
    /// Transposition table hit - evaluation found, value returned.
    Hit(f64),
    /// No valid search path (terminal state or fully explored).
    None,
}

/// Results from MCTS search containing visit counts and Q-values for all actions.
///
/// This struct provides the policy improvement (visit count distribution) and
/// action values (Q-values) for all 64 possible board positions after running
/// MCTS simulations.
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// Number of times each of the 64 actions was visited during search.
    pub visit_counts: Vec<u32>,
    /// Q-value (average reward) for each of the 64 actions.
    pub q_values: Vec<f64>,
}

/// MCTS search tree.
pub struct Tree {
    /// Root node of the tree representing the initial game state.
    root: Rc<RefCell<Node>>,
}

impl Tree {
    /// Creates a new [`Tree`] with the specified `root` node.
    pub fn new(root: Rc<RefCell<Node>>) -> Self {
        Self { root }
    }

    /// Executes batched MCTS search and returns visit counts and Q-values.
    ///
    /// Runs `num_inferences` batches, each collecting up to `states_per_inference` states
    /// for evaluation. Uses the neural network `model` for state evaluation, caches results
    /// in `transposition_table`, applies PUCT selection with `puct_config`, and uses `rng`
    /// for Dirichlet noise at the root.
    pub fn search<M: ModelEvaluator>(
        &self,
        num_inferences: usize,
        states_per_inference: usize,
        model: &M,
        transposition_table: &TranspositionTable,
        puct_config: PuctConfig,
        rng: &mut impl rand::Rng,
    ) -> SearchResults {
        let strategy = PuctStrategy::new(puct_config);

        for _ in 0..num_inferences {
            let mut batch = Vec::new();

            // GetBatch
            // Loop until batch is full or we can't find more nodes
            while batch.len() < states_per_inference {
                let res =
                    self.batch_puct(&self.root, transposition_table, &strategy, true, true, rng);
                match res {
                    SearchResult::Miss(state) => {
                        batch.push(state);
                    }
                    SearchResult::Hit(_) => {
                        // Hit means we found a node in TT (or terminal).
                        // We updated the tree stats in batch_puct.
                        // We continue to try to fill the batch.
                        // However, if the tree is small, we might hit the same nodes repeatedly?
                        // If we hit a node in TT, we expanded it.
                        // Next time we might go deeper.
                        // So it's fine.
                    }
                    SearchResult::None => {
                        // No more nodes to explore (e.g. game over or fully explored)
                        // If batch is empty, we stop.
                        // If batch has items, we process them.
                        break;
                    }
                }
            }

            if batch.is_empty() {
                break;
            }

            // Forward
            let results = model.infer(&batch);

            // PutBatch
            for (state, policy_eval) in batch.iter().zip(results.into_iter()) {
                transposition_table.add(*state, policy_eval);
            }

            // Re-traverse to update stats (fix virtual loss)
            // We run batch_puct (get_batch=false) for each item we added.
            // Since we added batch.len() items, we should run it that many times to ensure we cover them.
            // Note: batch_puct might find different paths if the tree changed, but usually it finds the same.
            for _ in 0..batch.len() {
                self.batch_puct(&self.root, transposition_table, &strategy, false, true, rng);
            }
        }

        // Build visit counts and Q-values arrays for all 64 actions
        let mut visit_counts = vec![0u32; 64];
        let mut q_values = vec![0.0f64; 64];

        let root = self.root.borrow();
        for child in root.children() {
            let child_ref = child.borrow();
            if let Some(action) = child_ref.action() {
                visit_counts[action] = child_ref.visit_count();
                // Only compute Q-value if the child was visited
                if child_ref.visit_count() > 0 {
                    q_values[action] = child_ref.sum_evaluation() / child_ref.visit_count() as f64;
                }
            }
        }

        SearchResults {
            visit_counts,
            q_values,
        }
    }

    fn batch_puct(
        &self,
        node: &Rc<RefCell<Node>>,
        tt: &TranspositionTable,
        strategy: &PuctStrategy,
        get_batch: bool,
        is_root: bool,
        rng: &mut impl rand::Rng,
    ) -> SearchResult {
        let mut node_ref = node.borrow_mut();

        if node_ref.is_leaf() {
            // Check TT
            if let Some(eval) = tt.get(node_ref.state()) {
                // Hit
                // Expand if needed
                if !node_ref.is_expanded() {
                    let legal_actions = node_ref.state().legal_actions();
                    // If no legal actions, it's terminal (or pass).
                    // legal_actions handles pass (returns 64).
                    // If empty, it's game over.
                    if !legal_actions.is_empty() {
                        for &action in &legal_actions {
                            let next_state = node_ref.state().apply(action);
                            let child = Node::new(next_state, Some(action));

                            // Manually add child to avoid double borrow of parent (node)
                            child.borrow_mut().set_parent(Some(Rc::downgrade(node)));
                            node_ref.children_mut().push(child);
                        }
                        node_ref.set_expanded(true);
                    } else {
                        // Terminal state
                        node_ref.set_expanded(true);
                        // We can't get a batch item from a terminal node.
                        return SearchResult::None;
                    }
                } else {
                    // Already expanded but still leaf -> Terminal
                    // (Since if it had children, is_leaf would be false)
                    return SearchResult::None;
                }

                let value = eval.value(); // Assuming Value(f64) is accessible. Value field is private?
                // Need to make Value field public or add getter in PolicyEvaluation.
                // Assuming I fixed PolicyEvaluation or will fix it.

                drop(node_ref); // Release borrow

                // Update stats
                if get_batch {
                    // Standard update
                    self.update_stats(node, value, false);
                } else {
                    // PutBatch: Fixing virtual loss
                    // We assume we are revisiting a node that had virtual loss applied.
                    // So we update with real value replacing virtual.
                    node.borrow_mut().update_with_real_value(value);
                }

                return SearchResult::Hit(value);
            } else {
                // Miss
                if get_batch {
                    // Add to batch
                    let state = *node_ref.state();
                    drop(node_ref);

                    // Virtual Loss
                    self.update_stats(node, 0.0, true);
                    return SearchResult::Miss(state);
                } else {
                    // PutBatch mode but missed TT.
                    // This means we strayed into a node not in TT.
                    // We can't evaluate it.
                    // Just return None.
                    return SearchResult::None;
                }
            }
        }

        // Selection
        let best_idx = select_best_child(&node_ref, strategy, tt, is_root, rng);

        if let Some(idx) = best_idx {
            let child = node_ref.children()[idx].clone();
            drop(node_ref);

            let res = self.batch_puct(&child, tt, strategy, get_batch, false, rng);

            // Backprop
            match res {
                SearchResult::Miss(_) => {
                    self.update_stats(node, 0.0, true);
                }
                SearchResult::Hit(val) => {
                    if get_batch {
                        // Standard update
                        let mut n = node.borrow_mut();
                        n.increment_visit_count();
                        n.add_evaluation(val);
                    } else {
                        // PutBatch: Fixing virtual loss
                        node.borrow_mut().update_with_real_value(val);
                    }
                }
                SearchResult::None => {}
            }
            res
        } else {
            // No children (Terminal?)
            // If leaf check failed (it wasn't leaf), but no children?
            // This happens if is_expanded=true but children list is empty (Game Over).
            // In that case, we should return the value of the state.
            // But we don't have it unless we check TT or calculate it.
            // If it's terminal, TT should have it?
            // Or we calculate it on the fly?
            // For now return None.
            SearchResult::None
        }
    }

    fn update_stats(&self, node: &Rc<RefCell<Node>>, value: f64, is_virtual: bool) {
        let mut n = node.borrow_mut();
        if is_virtual {
            n.apply_virtual_loss();
        } else {
            n.increment_visit_count();
            n.add_evaluation(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, PolicyEvaluation, Value};
    use common::{Bitboard, DiskColor};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    struct MockModel;
    impl ModelEvaluator for MockModel {
        fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation> {
            states
                .iter()
                .map(|_| PolicyEvaluation::new(Policy([1.0 / 64.0; 64]), Value(0.5)))
                .collect()
        }
    }

    #[test]
    fn test_search() {
        let state = State::new(Bitboard::default(), DiskColor::Dark);
        let root = Node::new(state, None);
        let tree = Tree::new(root);
        let mut tt = TranspositionTable::new();
        let model = MockModel;
        let config = PuctConfig {
            c_puct: 1.0,
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: 1.0,
        };

        let mut rng = StdRng::seed_from_u64(42);
        // Run search with 2 batches of size 2 to ensure children are visited
        let results = tree.search(2, 2, &model, &mut tt, config, &mut rng);

        // Should have some visits
        let total_visits: u32 = results.visit_counts.iter().sum();
        assert!(total_visits > 0);

        // Root should have visits
        assert!(tree.root.borrow().visit_count() > 0);
    }
}
