//! MCTS search tree structure.
//!
//! The `Tree` holds nodes in an arena allocator and provides access to the
//! root node. Search logic is handled separately in the `search` module.

use crate::arena::{Arena, NodeId};
use crate::node::Node;
use crate::state::State;

/// MCTS search tree
///
/// Contains the arena of all nodes and keeps track of the root node.
/// Search algorithms operate on this structure but are defined elsewhere.
pub struct Tree {
    arena: Arena,
    root: NodeId,
}

impl Tree {
    /// Creates a new [`Tree`] with the given root state.
    pub fn new(root_state: State) -> Self {
        let mut arena = Arena::new();
        let root = arena.allocate(Node::new(root_state, None));
        Self { arena, root }
    }

    /// Returns the root node ID.
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Returns a reference to the arena.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Returns a mutable reference to the arena.
    pub fn arena_mut(&mut self) -> &mut Arena {
        &mut self.arena
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
        let tree = Tree::new(default_state());
        assert_eq!(tree.root(), 0);
        assert_eq!(tree.arena().len(), 1);
    }
}
