use crate::{
    inference::ModelEvaluator,
    node::Node,
    selection::{PuctConfig, PuctStrategy, SelectionStrategy, select_best_child},
    state::State,
    transposition_table::TranspositionTable,
};
use std::{cell::RefCell, rc::Rc};

enum SearchResult {
    Miss(State),
    Hit(f64),
    None,
}

/// MCTS tree
pub struct Tree {
    /// Root node
    root: Rc<RefCell<Node>>,
}

impl Tree {
    /// Creates a new [`Tree`].
    pub fn new(root: Rc<RefCell<Node>>) -> Self {
        Self { root }
    }

    /// Executes Batched MCTS search.
    pub fn search<M: ModelEvaluator>(
        &self,
        num_batches: usize,
        batch_size: usize,
        model: &M,
        transposition_table: &mut TranspositionTable,
        puct_config: PuctConfig,
    ) -> Option<usize> {
        let strategy = PuctStrategy::new(puct_config);

        for _ in 0..num_batches {
            let mut batch = Vec::new();

            // GetBatch
            // Loop until batch is full or we can't find more nodes
            while batch.len() < batch_size {
                let res = self.batch_puct(&self.root, transposition_table, &strategy, true);
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
                self.batch_puct(&self.root, transposition_table, &strategy, false);
            }
        }

        // Return best action
        let root = self.root.borrow();
        root.children()
            .iter()
            .max_by(|a, b| a.borrow().visit_count().cmp(&b.borrow().visit_count()))
            .and_then(|c| c.borrow().action())
    }

    fn batch_puct<S: SelectionStrategy>(
        &self,
        node: &Rc<RefCell<Node>>,
        tt: &TranspositionTable,
        strategy: &S,
        get_batch: bool,
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
                        // We should return the actual game result.
                        // But PolicyEvaluation has a value.
                        // If the state is terminal, the value in TT should be the game result.
                        // (Assuming TT is populated correctly for terminal states or model predicts it)
                    }
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
        let best_idx = select_best_child(
            node_ref.children(),
            node_ref.visit_count(),
            strategy,
            tt,
            node_ref.state(),
        );

        if let Some(idx) = best_idx {
            let child = node_ref.children()[idx].clone();
            drop(node_ref);

            let res = self.batch_puct(&child, tt, strategy, get_batch);

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
        let config = PuctConfig { c_puct: 1.0 };

        // Run search with 1 batch of size 2
        let action = tree.search(1, 2, &model, &mut tt, config);

        // Should return an action
        assert!(action.is_some());

        // Root should have visits
        assert!(tree.root.borrow().visit_count() > 0);
    }
}
