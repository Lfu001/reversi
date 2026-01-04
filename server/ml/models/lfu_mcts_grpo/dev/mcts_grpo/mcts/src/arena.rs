//! Arena allocator for MCTS nodes.
//!
//! Provides contiguous storage for all tree nodes in a single `Vec<Node>`,
//! enabling better cache locality compared to scattered heap allocations.

use crate::node::Node;

/// Index into the arena.
pub type NodeId = usize;

/// Arena allocator for MCTS nodes.
///
/// All nodes are stored contiguously in a `Vec`, with parent/child
/// relationships represented as indices (`NodeId`).
#[derive(Clone)]
pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    /// Creates a new empty [`Arena`].
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Allocates a new node and returns its ID.
    pub fn allocate(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Returns a reference to the node at the given ID.
    ///
    /// # Panics
    /// Panics if the ID is out of bounds.
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    /// Returns a mutable reference to the node at the given ID.
    ///
    /// # Panics
    /// Panics if the ID is out of bounds.
    pub fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }

    /// Adds a child node to a parent and returns the child's ID.
    ///
    /// The child is linked as the first child of the parent, with
    /// previous first_child becoming the new child's next_sibling.
    pub fn add_child(&mut self, parent_id: NodeId, mut child: Node) -> NodeId {
        child.parent = Some(parent_id);
        child.next_sibling = self.nodes[parent_id].first_child;

        let child_id = self.allocate(child);
        self.nodes[parent_id].first_child = Some(child_id);
        child_id
    }

    /// Returns an iterator over the children of a node.
    pub fn children(&self, id: NodeId) -> ChildrenIter<'_> {
        ChildrenIter {
            arena: self,
            current: self.nodes[id].first_child,
        }
    }

    /// Returns the number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over children of a node.
pub struct ChildrenIter<'a> {
    arena: &'a Arena,
    current: Option<NodeId>,
}

impl<'a> Iterator for ChildrenIter<'a> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;
        self.current = self.arena.nodes[current].next_sibling;
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;
    use common::{Bitboard, DiskColor};

    fn default_state() -> State {
        State::new(Bitboard::default(), DiskColor::Dark)
    }

    #[test]
    fn test_arena_new() {
        let arena = Arena::new();
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_allocate_and_get() {
        let mut arena = Arena::new();
        let node = Node::new(default_state(), None);
        let id = arena.allocate(node);

        assert_eq!(id, 0);
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(id).state(), &default_state());
    }

    #[test]
    fn test_add_child() {
        let mut arena = Arena::new();
        let parent_id = arena.allocate(Node::new(default_state(), None));
        let child_id = arena.add_child(parent_id, Node::new(default_state(), Some(0)));

        assert_eq!(arena.get(child_id).parent, Some(parent_id));
        assert_eq!(arena.get(parent_id).first_child, Some(child_id));
    }

    #[test]
    fn test_children_iterator() {
        let mut arena = Arena::new();
        let parent_id = arena.allocate(Node::new(default_state(), None));

        let child1_id = arena.add_child(parent_id, Node::new(default_state(), Some(0)));
        let child2_id = arena.add_child(parent_id, Node::new(default_state(), Some(1)));
        let child3_id = arena.add_child(parent_id, Node::new(default_state(), Some(2)));

        let children: Vec<_> = arena.children(parent_id).collect();
        // Children are added in reverse order (prepend)
        assert_eq!(children, vec![child3_id, child2_id, child1_id]);
    }
}
