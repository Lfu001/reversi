use crate::{
    inference::ModelEvaluator,
    node::Node,
    selection::{ChildSelection, PuctConfig, PuctStrategy},
    state::State,
    transposition_table::TranspositionTable,
};
use common::DiskColor;
use std::{cell::RefCell, rc::Rc};

/// Internal result type for batch PUCT traversal.
#[derive(Debug)]
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
        pbar: Option<indicatif::ProgressBar>,
    ) -> SearchResults {
        let strategy = PuctStrategy::new(puct_config);
        let mut child_selection = ChildSelection::new(puct_config.dirichlet_alpha);
        let mut batch = Vec::with_capacity(states_per_inference);
        for _ in 0..num_inferences {
            batch.clear();
            // GetBatch
            // Loop until batch is full or we can't find more nodes
            while batch.len() < states_per_inference {
                let res = self.batch_puct(
                    &self.root,
                    transposition_table,
                    &strategy,
                    &mut child_selection,
                    true,
                    true,
                    rng,
                );
                match res {
                    SearchResult::Miss(state) => {
                        batch.push(state);
                    }
                    SearchResult::Hit(_) => {
                        // Hit means we found a node in TT (or terminal).
                        // We updated the tree stats in batch_puct.
                    }
                    SearchResult::None => {
                        // No more nodes to explore (e.g. game over or fully explored)
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
            for _ in 0..batch.len() {
                self.batch_puct(
                    &self.root,
                    transposition_table,
                    &strategy,
                    &mut child_selection,
                    false,
                    true,
                    rng,
                );
            }

            if let Some(pb) = &pbar {
                pb.inc(1);
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
        child_selection: &mut ChildSelection,
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
                    let current_state = *node_ref.state();
                    if current_state.is_terminal() {
                        // Terminal state
                        node_ref.set_expanded(true);
                        // We can't get a batch item from a terminal node.
                        return SearchResult::None;
                    } else {
                        let children = current_state.legal_actions().map(|action| {
                            let next_state = current_state.apply(action);
                            let child = Node::new(next_state, Some(action));
                            child.borrow_mut().set_parent(Some(Rc::downgrade(node)));
                            child
                        });
                        for child in children {
                            node_ref.children_mut().push(child);
                        }
                        node_ref.set_expanded(true);
                    }
                    drop(node_ref); // Release borrow
                } else {
                    // Already expanded but still leaf -> Terminal
                    // (Since if it had children, is_leaf would be false)
                    return SearchResult::None;
                }

                let value = eval.value().value();

                // Determine if we should negate the value (Turn Alternation)
                // We need to re-borrow the node because the mutable borrow `node_ref` was dropped above.
                let node_ref = node.borrow();

                let parent_turn = get_parent_turn(&node_ref);

                let current_turn = node_ref.state().turn();

                // Drop borrow
                drop(node_ref);

                let value_for_parent = if parent_turn != current_turn {
                    -value
                } else {
                    value
                };

                // Update stats
                if get_batch {
                    // Standard update
                    self.update_stats(node, value_for_parent, false);
                } else {
                    // PutBatch: Fixing virtual loss
                    node.borrow_mut().update_with_real_value(value_for_parent);
                }

                return SearchResult::Hit(value_for_parent);
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
                    return SearchResult::None;
                }
            }
        }

        // Selection
        let best_idx = child_selection.select_best_child(&node_ref, strategy, tt, is_root, rng);

        if let Some(idx) = best_idx {
            let child = node_ref.children()[idx].clone();
            drop(node_ref);

            let res = self.batch_puct(&child, tt, strategy, child_selection, get_batch, false, rng);

            // Backprop
            match res {
                SearchResult::Miss(_) => {
                    self.update_stats(node, 0.0, true);
                }
                SearchResult::Hit(val) => {
                    if get_batch {
                        let mut n = node.borrow_mut();
                        n.increment_visit_count();
                        n.add_evaluation(val);
                    } else {
                        // PutBatch
                        node.borrow_mut().update_with_real_value(val);
                    }

                    // Prepare return value for Grandparent
                    let node_ref = node.borrow();
                    let current_turn = node_ref.state().turn();

                    let parent_turn = get_parent_turn(&node_ref);

                    let val_for_gp = if parent_turn != current_turn {
                        -val
                    } else {
                        val
                    };

                    return SearchResult::Hit(val_for_gp);
                }
                SearchResult::None => {}
            }
            res
        } else {
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

fn get_parent_turn(node: &Node) -> DiskColor {
    node.parent()
        .and_then(|pw| pw.upgrade())
        .and_then(|prc| prc.try_borrow().ok().map(|p| p.state().turn()))
        .unwrap_or_else(|| node.state().turn())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{PolicyEvaluation, Value};
    use common::{Bitboard, DiskColor};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[cfg(test)]
    mod tests_utils {
        use super::*;
        use crate::policy::{Policy, PolicyEvaluation, Value};
        use crate::state::State;
        use common::Bitboard;

        pub fn create_default_config() -> PuctConfig {
            PuctConfig {
                c_puct: 1.0,
                dirichlet_epsilon: 0.0,
                dirichlet_alpha: 1.0,
            }
        }

        pub fn create_mock_policy() -> Policy {
            Policy([1.0 / 64.0; 64])
        }

        /// Helper to create a bitboard with specific positions occupied.
        /// indices are 0-63.
        pub fn bitboard_with_stones(dark: &[usize], light: &[usize]) -> Bitboard {
            let mut d = 0u64;
            let mut l = 0u64;
            for &idx in dark {
                d |= 1 << idx;
            }
            for &idx in light {
                l |= 1 << idx;
            }
            Bitboard::new(d, l)
        }

        pub struct MockValModel {
            pub val: f64,
        }

        impl ModelEvaluator for MockValModel {
            fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation> {
                states
                    .iter()
                    .map(|_| PolicyEvaluation::new(create_mock_policy(), Value(self.val)))
                    .collect()
            }
        }
    }

    #[test]
    fn test_search() {
        use tests_utils::*;
        // (1) Setup
        let state = State::new(Bitboard::default(), DiskColor::Dark);
        let root = Node::new(state, None);
        let tree = Tree::new(root.clone());
        let tt = TranspositionTable::new();
        let model = MockValModel { val: 0.5 };
        let config = create_default_config();
        let mut rng = StdRng::seed_from_u64(42);

        // (2) Execution
        tree.search(2, 2, &model, &tt, config, &mut rng, None);

        // (3) Assertion
        assert!(root.borrow().visit_count() > 0);
    }

    #[test]
    fn test_value_propagation_inversion() {
        use tests_utils::*;
        // (1) Setup: Parent (Dark) -> Child (Light). Standard turn alternation.
        let parent_state = State::new(Bitboard::default(), DiskColor::Dark);
        let parent = Node::new(parent_state, None);

        let child_state = State::new(Bitboard::default(), DiskColor::Light);
        let child = Node::new(child_state, Some(0));
        Node::add_child(&parent, child);

        let tree = Tree::new(parent.clone());
        let tt = TranspositionTable::new();
        tt.add(
            parent_state,
            PolicyEvaluation::new(create_mock_policy(), Value(0.0)),
        );

        let model = MockValModel { val: 0.8 }; // Child (Light) evaluated at 0.8
        let config = create_default_config();
        let mut rng = StdRng::seed_from_u64(42);

        // (2) Execution
        tree.search(1, 1, &model, &tt, config, &mut rng, None);

        // (3) Assertion: Parent (Dark) should see -0.8
        let val = parent.borrow().average_value();
        assert!((val - (-0.8)).abs() < 1e-6, "Expected -0.8, got {}", val);
    }

    #[test]
    fn test_turn_skip_propagation() {
        use tests_utils::*;
        // (1) Setup: Parent (Dark) -> Child (Dark). Turn skip case.
        let parent_state = State::new(Bitboard::default(), DiskColor::Dark);
        let parent = Node::new(parent_state, None);

        // Use distinct bitboard for child to trigger inference
        let child_bb = bitboard_with_stones(&[0], &[1]); // H8 Dark, H7 White
        let child_state = State::new(child_bb, DiskColor::Dark);
        let child = Node::new(child_state, Some(0));
        Node::add_child(&parent, child);

        let tree = Tree::new(parent.clone());
        let tt = TranspositionTable::new();
        tt.add(
            parent_state,
            PolicyEvaluation::new(create_mock_policy(), Value(0.0)),
        );

        let model = MockValModel { val: 0.8 }; // Child (Dark) evaluated at 0.8
        let config = create_default_config();
        let mut rng = StdRng::seed_from_u64(42);

        // (2) Execution
        tree.search(1, 1, &model, &tt, config, &mut rng, None);

        // (3) Assertion: Parent (Dark) should see 0.8 (no inversion)
        let val = parent.borrow().average_value();
        assert!((val - 0.8).abs() < 1e-6, "Expected 0.8, got {}", val);
    }

    #[test]
    fn test_three_generation_propagation() {
        use tests_utils::*;
        // (1) Setup: Grandparent (Dark) -> Parent (Light) -> Child (Dark)
        // Turn: Dark -> Light -> Dark (Alternating)
        let gp_state = State::new(Bitboard::default(), DiskColor::Dark);
        let gp = Node::new(gp_state, None);

        let p_state = State::new(Bitboard::default(), DiskColor::Light);
        let p = Node::new(p_state, Some(0));
        Node::add_child(&gp, p.clone());

        let c_bb = bitboard_with_stones(&[0], &[1]);
        let c_state = State::new(c_bb, DiskColor::Dark);
        let c = Node::new(c_state, Some(1));
        Node::add_child(&p, c);

        let tree = Tree::new(gp.clone());
        let tt = TranspositionTable::new();
        // Root must be in TT for selection
        tt.add(
            gp_state,
            PolicyEvaluation::new(create_mock_policy(), Value(0.0)),
        );
        // Parent must be in TT for selection to its child
        tt.add(
            p_state,
            PolicyEvaluation::new(create_mock_policy(), Value(0.0)),
        );

        let model = MockValModel { val: 0.8 }; // Child (Dark) evaluated at 0.8
        let config = create_default_config();
        let mut rng = StdRng::seed_from_u64(42);

        // (2) Execution: Need at least 1 search directed to the leaf
        tree.search(1, 1, &model, &tt, config, &mut rng, None);

        // (3) Assertion
        let gp_val = gp.borrow().average_value();
        let p_val = p.borrow().average_value();

        // Child (Dark) 0.8 -> Parent (Light) -0.8 -> Grandparent (Dark) 0.8
        assert!(
            (p_val - (-0.8)).abs() < 1e-6,
            "Parent expected -0.8, got {}",
            p_val
        );
        assert!(
            (gp_val - 0.8).abs() < 1e-6,
            "Grandparent expected 0.8, got {}",
            gp_val
        );
    }
}
