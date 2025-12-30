//! Node structure for MCTS tree.
//!
//! Nodes are stored in an arena allocator and linked via indices.

use crate::arena::NodeId;
use crate::state::State;

/// Node in the MCTS tree (arena-allocated).
///
/// All pointer-like relationships use `Option<NodeId>` instead of actual pointers,
/// enabling storage in a contiguous `Vec` for better cache locality.
#[derive(Debug, Clone)]
pub struct Node {
    /// Parent node index (None for root).
    pub parent: Option<NodeId>,
    /// First child index (None if no children).
    pub first_child: Option<NodeId>,
    /// Next sibling index (None if last sibling).
    pub next_sibling: Option<NodeId>,
    /// Game state at this node.
    state: State,
    /// Action that led to this node (None for root).
    action: Option<usize>,
    /// Visit count.
    visit_count: u32,
    /// Sum of evaluations.
    sum_evaluation: f64,
    /// Whether node is expanded (children generated).
    is_expanded: bool,
}

impl Node {
    /// Creates a new [`Node`] with the given `state` and optional `action`.
    pub fn new(state: State, action: Option<usize>) -> Self {
        Self {
            parent: None,
            first_child: None,
            next_sibling: None,
            state,
            action,
            visit_count: 0,
            sum_evaluation: 0.0,
            is_expanded: false,
        }
    }

    /// Returns the state of this node.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Returns the action that led to this node.
    pub fn action(&self) -> Option<usize> {
        self.action
    }

    /// Returns the visit count.
    pub fn visit_count(&self) -> u32 {
        self.visit_count
    }

    /// Increments the visit count by 1.
    pub fn increment_visit_count(&mut self) {
        self.visit_count += 1;
    }

    /// Returns the sum of evaluations.
    pub fn sum_evaluation(&self) -> f64 {
        self.sum_evaluation
    }

    /// Adds the given `value` to the sum of evaluations.
    pub fn add_evaluation(&mut self, value: f64) {
        self.sum_evaluation += value;
    }

    /// Returns whether the node is expanded.
    pub fn is_expanded(&self) -> bool {
        self.is_expanded
    }

    /// Sets the expanded flag.
    pub fn set_expanded(&mut self, expanded: bool) {
        self.is_expanded = expanded;
    }

    /// Returns the Q-value (average evaluation).
    ///
    /// Returns 0.0 if the node has not been visited.
    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.sum_evaluation / self.visit_count as f64
        }
    }

    /// Applies virtual loss (Virtual Mean).
    ///
    /// Adds 1 to visit count and adds current Q-value to sum,
    /// keeping the average value unchanged.
    pub fn apply_virtual_loss(&mut self) {
        let current_mean = self.q_value();
        self.visit_count += 1;
        self.sum_evaluation += current_mean;
    }

    /// Updates the node by replacing virtual loss with real value.
    ///
    /// Assumes `apply_virtual_loss` was called before.
    pub fn update_with_real_value(&mut self, real_value: f64) {
        let current_mean = self.q_value();
        self.sum_evaluation -= current_mean;
        self.sum_evaluation += real_value;
    }

    /// Returns true if the node has no children.
    pub fn is_leaf(&self) -> bool {
        self.first_child.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Bitboard, DiskColor};

    fn default_state() -> State {
        State::new(Bitboard::default(), DiskColor::Dark)
    }

    #[test]
    fn test_new() {
        let node = Node::new(default_state(), None);
        assert!(node.parent.is_none());
        assert!(node.first_child.is_none());
        assert!(node.next_sibling.is_none());
        assert_eq!(node.visit_count(), 0);
        assert_eq!(node.sum_evaluation(), 0.0);
    }

    #[test]
    fn test_is_leaf() {
        let node = Node::new(default_state(), None);
        assert!(node.is_leaf());
    }

    #[test]
    fn test_q_value() {
        let mut node = Node::new(default_state(), None);
        assert_eq!(node.q_value(), 0.0);

        node.increment_visit_count();
        node.add_evaluation(0.8);
        assert_eq!(node.q_value(), 0.8);

        node.increment_visit_count();
        node.add_evaluation(0.4);
        assert!((node.q_value() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_virtual_loss() {
        let mut node = Node::new(default_state(), None);
        node.increment_visit_count();
        node.add_evaluation(0.8);

        let q_before = node.q_value();
        node.apply_virtual_loss();

        // Q-value should remain approximately the same
        assert!((node.q_value() - q_before).abs() < 1e-10);
        assert_eq!(node.visit_count(), 2);
    }

    #[test]
    fn test_update_with_real_value() {
        let mut node = Node::new(default_state(), None);
        node.increment_visit_count();
        node.add_evaluation(0.5);

        node.apply_virtual_loss();
        node.update_with_real_value(1.0);

        // After update: (0.5 - 0.5 + 1.0) / 2 = 0.75
        assert!((node.q_value() - 0.75).abs() < 1e-10);
    }
}
