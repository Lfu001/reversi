use crate::state::State;
use std::cell::RefCell;
use std::rc::Rc;
use std::rc::Weak;

/// Node in the MCTS tree
pub struct Node {
    /// A parent of the node.
    parent: Option<Weak<RefCell<Node>>>,
    /// Children of the node.
    children: Vec<Rc<RefCell<Node>>>,
    /// A state of the game.
    state: State,
    /// A visit count of the node.
    visit_count: u32,
    /// An action that led to this node.
    action: Option<usize>,
    /// A sum of evaluations of the node.
    sum_evaluation: f64,
    /// Whether this node has been expanded with child nodes.
    is_expanded: bool,
}

impl Node {
    /// Creates a new [`Node`] with the given `state` and optional `action`.
    ///
    /// The `action` represents the move that led to this state, or `None` for the root node.
    pub fn new(state: State, action: Option<usize>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Node {
            parent: None,
            children: Vec::new(),
            state,
            visit_count: 0,
            action,
            sum_evaluation: 0.0,
            is_expanded: false,
        }))
    }

    /// Returns the state of this node.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Sets the parent reference to the given weak pointer `parent`.
    pub fn set_parent(&mut self, parent: Option<Weak<RefCell<Node>>>) {
        self.parent = parent;
    }

    /// Returns the children of this node.
    pub fn children(&self) -> &Vec<Rc<RefCell<Node>>> {
        &self.children
    }

    /// Returns a mutable reference to the children of this node.
    pub fn children_mut(&mut self) -> &mut Vec<Rc<RefCell<Node>>> {
        &mut self.children
    }

    /// Returns the visit count.
    pub fn visit_count(&self) -> u32 {
        self.visit_count
    }

    /// Increments the visit count.
    pub fn increment_visit_count(&mut self) {
        self.visit_count += 1;
    }

    /// Returns the action.
    pub fn action(&self) -> Option<usize> {
        self.action
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

    /// Sets the expanded flag to the given `expanded` value.
    pub fn set_expanded(&mut self, expanded: bool) {
        self.is_expanded = expanded;
    }

    /// Returns the average value of the node.
    pub fn average_value(&self) -> f64 {
        if self.visit_count == 0 {
            // μFPU: Use average value of visited siblings
            if let Some(parent_weak) = &self.parent
                && let Some(parent_rc) = parent_weak.upgrade()
            {
                // Note: We use try_borrow here because we might be borrowing the parent
                // in a context where it's already borrowed.
                if let Ok(parent) = parent_rc.try_borrow() {
                    return parent.average_q_value_of_visited_children().unwrap_or(0.0);
                }
            }
            0.0
        } else {
            self.q_value()
        }
    }

    /// Calculates the average Q-value of all visited children.
    ///
    /// Returns `Some` with the average Q-value if there are visited children, or `None` if all children are unvisited.
    pub fn average_q_value_of_visited_children(&self) -> Option<f64> {
        let mut sum_q = 0.0;
        let mut count = 0;
        for child in &self.children {
            // We use try_borrow to safely handle cases where a child might be currently borrowed
            // (e.g., the child that called this method via its parent).
            // RefCell allows multiple immutable borrows, so this is safe.
            if let Ok(child_ref) = child.try_borrow()
                && child_ref.visit_count > 0
            {
                sum_q += child_ref.q_value();
                count += 1;
            }
        }
        if count > 0 {
            Some(sum_q / count as f64)
        } else {
            None
        }
    }

    /// Returns the Q-value (average evaluation) based on visits.
    ///
    /// Panics if `visit_count` is 0. Should only be called after verifying the node has been visited.
    fn q_value(&self) -> f64 {
        assert!(self.visit_count > 0);
        self.sum_evaluation / self.visit_count as f64
    }

    /// Returns true if the node is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Applies virtual loss (Virtual Mean).
    /// Adds 1 to visit count and adds current average value to sum.
    pub fn apply_virtual_loss(&mut self) {
        let current_mean = self.average_value();
        self.visit_count += 1;
        self.sum_evaluation += current_mean;
    }

    /// Updates the node with the real value, replacing the virtual mean.
    ///
    /// Takes the actual evaluation `real_value` and replaces the virtual loss placeholder.
    /// Assumes virtual loss was applied beforehand (visit count is already incremented).
    pub fn update_with_real_value(&mut self, real_value: f64) {
        // Remove the virtual mean contribution
        // Since we added average_value() to sum, and visit_count was incremented.
        // The current average_value() should be the same as before if we added the mean.
        // So we subtract the current average_value().
        let current_mean = self.average_value();
        self.sum_evaluation -= current_mean;
        self.sum_evaluation += real_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Bitboard, DiskColor};

    fn default_state() -> State {
        State::new(Bitboard::default(), DiskColor::Dark)
    }

    impl Node {
        /// Returns the parent of this node.
        pub fn parent(&self) -> &Option<Weak<RefCell<Node>>> {
            &self.parent
        }
        /// Sets the visit count to the specified `count` value.
        pub fn set_visit_count(&mut self, count: u32) {
            self.visit_count = count;
        }
        /// Adds a child node to the current node.
        ///
        /// Establishes bidirectional linking: sets the `parent` weak reference in the `child`,
        /// and adds the `child` strong reference to the `parent`'s children list.
        pub fn add_child(parent: &Rc<RefCell<Node>>, child: Rc<RefCell<Node>>) {
            // Set weak reference to parent in the child node
            child.borrow_mut().set_parent(Some(Rc::downgrade(parent)));
            // Add strong reference to child in the parent node
            parent.borrow_mut().children.push(child);
        }
    }

    #[test]
    fn test_new() {
        let node = Node::new(default_state(), None);
        assert!(node.borrow().parent().is_none());
        assert!(node.borrow().children().is_empty());
    }

    #[test]
    fn test_add_child() {
        let node = Node::new(default_state(), None);
        let child = Node::new(default_state(), None);
        Node::add_child(&node, child);
        assert!(node.borrow().children().len() == 1);
    }

    #[test]
    fn test_is_leaf() {
        let node = Node::new(default_state(), None);
        assert!(node.borrow().is_leaf());
    }

    #[test]
    fn test_is_not_leaf() {
        let node = Node::new(default_state(), None);
        let child = Node::new(default_state(), None);
        Node::add_child(&node, child);
        assert!(!node.borrow().is_leaf());
    }

    #[test]
    fn test_mu_fpu() {
        // Create parent
        let parent = Node::new(default_state(), None);

        // Child 1: Visited, Value 0.8
        let child1 = Node::new(default_state(), Some(1));
        child1.borrow_mut().set_visit_count(10);
        child1.borrow_mut().add_evaluation(8.0); // Avg = 0.8
        Node::add_child(&parent, child1);

        // Child 2: Visited, Value 0.4
        let child2 = Node::new(default_state(), Some(2));
        child2.borrow_mut().set_visit_count(5);
        child2.borrow_mut().add_evaluation(2.0); // Avg = 0.4
        Node::add_child(&parent, child2);

        // Child 3: Unvisited
        let child3 = Node::new(default_state(), Some(3));
        Node::add_child(&parent, child3.clone());

        // Check Child 3's value. Should be average of (0.8 + 0.4) / 2 = 0.6
        assert!((child3.borrow().average_value() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_mu_fpu_no_visited_siblings() {
        let parent = Node::new(default_state(), None);
        let child = Node::new(default_state(), Some(1));
        Node::add_child(&parent, child.clone());

        // No visited siblings -> 0.0
        assert_eq!(child.borrow().average_value(), 0.0);
    }
}
