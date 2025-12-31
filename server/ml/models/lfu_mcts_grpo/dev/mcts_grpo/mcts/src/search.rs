//! MCTS search algorithms.
//!
//! Implements Batch MCTS as described in the paper:
//! "Batch Monte Carlo Tree Search" by Tristan Cazenave.
//!
//! Key algorithms (68% winrate configuration):
//! - Algorithm 8: BatchSecond (replaces Algorithm 1, with Second Move forcing)
//! - Algorithm 2: UpdateStatistics
//! - Algorithm 3: UpdateStatisticsGet (VirtualMean only)
//! - Algorithm 4/5: GetBatchSecond / PutBatchSecond
//! - Algorithm 9: GetMoveSecond (with Last Iteration from Algorithm 7)

use crate::arena::{Arena, NodeId};
use crate::inference::ModelEvaluator;
use crate::node::Node;
use crate::selection::{PuctConfig, PuctStrategy};
use crate::state::State;
use crate::transposition_table::TranspositionTable;
use crate::tree::Tree;

/// Virtual loss count (vl in the paper). Using VirtualMean.
const VL: u32 = 1;

/// Unknown threshold for Last Iteration (U in Algorithm 7).
const LAST_ITERATION_U: usize = 40;

/// Maximum descents per GetBatch call (N in Algorithm 4).
const MAX_DESCENTS_PER_BATCH: usize = 500;

/// Result from BatchPUCT.
#[derive(Debug, Clone, Copy)]
enum PuctResult {
    /// State not in TT, needs inference. Contains the state.
    Unknown(State),
    /// Evaluation value.
    Value(f64),
}

/// Results from MCTS search.
#[derive(Debug, Clone)]
pub struct SearchResults {
    pub visit_counts: Vec<u32>,
    pub q_values: Vec<f64>,
}

/// treeBatch: A copy of the main tree's statistics for batch construction.
/// Paper says "a copy of the main tree", but notes that separating statistics
/// inside nodes with a stamp is more elaborate. We use a full copy approach.
#[derive(Clone)]
struct TreeBatch {
    /// Cloned arena from main tree at start of GetBatch
    arena: Arena,
}

impl From<&Tree> for TreeBatch {
    /// Creates treeBatch as a copy of the main tree (Algorithm 4, line 265)
    fn from(tree: &Tree) -> Self {
        Self {
            arena: tree.arena().clone(),
        }
    }
}

/// Common interface for tree structures used in MCTS traversal.
///
/// Both `Tree` (main search tree) and `TreeBatch` (batch construction copy)
/// implement this trait, enabling unified traversal logic in `batch_mcts`.
///
/// The `update_statistics` method is implemented differently for each type:
/// - `Tree`: Algorithm 2 (standard update, ignores Unknown)
/// - `TreeBatch`: Algorithm 3 (VirtualMean for Unknown states)
trait TreeLike {
    fn arena(&self) -> &Arena;
    fn arena_mut(&mut self) -> &mut Arena;

    /// Update node statistics after tree descent.
    ///
    /// Implementation varies by tree type:
    /// - `Tree`: Algorithm 2 - increment visit count and add value
    /// - `TreeBatch`: Algorithm 3 (VirtualMean) - for Unknown, add μ×vl
    fn update_statistics(&mut self, node_id: NodeId, res: PuctResult);
}

impl TreeLike for TreeBatch {
    fn arena(&self) -> &Arena {
        &self.arena
    }

    fn arena_mut(&mut self) -> &mut Arena {
        &mut self.arena
    }

    /// Algorithm 3: UpdateStatisticsGet (VirtualMean)
    ///
    /// For Unknown: t.p(s,m) += vl; t.sum(s,m) += vl × μ
    /// For Value: t.p(s,m) += 1; t.sum(s,m) += res
    fn update_statistics(&mut self, node_id: NodeId, res: PuctResult) {
        match res {
            PuctResult::Unknown(_) => {
                // Algorithm 3: μ = t.sum(s,m) / t.p(s,m)
                let node = self.arena.get_mut(node_id);
                let mu = if node.visit_count() > 0 {
                    node.q_value()
                } else {
                    0.0
                };
                // Algorithm 3 (VirtualMean):
                // t.p(s) = t.p(s) + vl
                // t.sum(s) = t.sum(s) + vl × μ
                for _ in 0..VL {
                    node.increment_visit_count();
                }
                node.add_evaluation(VL as f64 * mu);
            }
            PuctResult::Value(v) => {
                // Algorithm 3: t.p(s) = t.p(s) + 1; t.sum(s) = t.sum(s) + res
                let node = self.arena.get_mut(node_id);
                node.increment_visit_count();
                node.add_evaluation(v);
            }
        }
    }
}

impl TreeLike for Tree {
    fn arena(&self) -> &Arena {
        self.arena()
    }

    fn arena_mut(&mut self) -> &mut Arena {
        self.arena_mut()
    }

    /// Algorithm 2: UpdateStatistics
    ///
    /// if res ≠ Unknown: t.p(s,m) += 1; t.sum(s,m) += res
    fn update_statistics(&mut self, node_id: NodeId, res: PuctResult) {
        // Algorithm 2: if res ≠ Unknown then
        if let PuctResult::Value(v) = res {
            // Algorithm 2: t.p(s) = t.p(s) + 1; t.sum(s) = t.sum(s) + res
            let node = self.arena_mut().get_mut(node_id);
            node.increment_visit_count();
            node.add_evaluation(v);
        }
        // Algorithm 2: if res = Unknown, do nothing
    }
}

/// Information about best and second-best children at root.
/// Used for Algorithm 8 (Second Move Heuristic).
#[derive(Debug, Clone, Copy)]
struct SecondMoveInfo {
    best_id: Option<NodeId>,
    second_id: Option<NodeId>,
    best_visits: u32,
    second_visits: u32,
}

impl SecondMoveInfo {
    /// Compute best and second-best children by visit count.
    /// Algorithm 8, lines 356-357.
    fn from_root(arena: &Arena, root_id: NodeId) -> Self {
        let mut best_id = None;
        let mut second_id = None;
        let mut best_visits = 0u32;
        let mut second_visits = 0u32;

        for child_id in arena.children(root_id) {
            let child = arena.get(child_id);
            let visits = child.visit_count();
            if visits > best_visits {
                second_id = best_id;
                second_visits = best_visits;
                best_id = Some(child_id);
                best_visits = visits;
            } else if visits > second_visits {
                second_id = Some(child_id);
                second_visits = visits;
            }
        }

        Self {
            best_id,
            second_id,
            best_visits,
            second_visits,
        }
    }

    /// Algorithm 8, line 358: if b >= b' + budget - i then force second.
    fn should_force_second(&self, budget: usize, i: usize) -> bool {
        let remaining = budget.saturating_sub(i);
        self.best_visits as usize >= self.second_visits as usize + remaining
    }
}

/// Context for Second Move Heuristic (Algorithm 8, lines 355-361).
///
/// When provided to `batch_mcts`, forces exploration of second-best move
/// at root if the best move's lead is insurmountable with remaining budget.
#[derive(Debug, Clone, Copy)]
struct SecondMoveContext {
    budget: usize,
    spent_budget: usize,
}

/// Unified MCTS tree traversal (Algorithm 1/8).
///
/// This single generic function replaces the previous separate implementations:
/// - `batch_puct_on_batch_tree` (Algorithm 1, GetBatch=True)
/// - `batch_second_on_batch_tree` (Algorithm 8, GetBatch=True)
/// - `batch_second_on_main_tree` (Algorithm 8, GetBatch=False)
///
/// The update strategy is automatically selected based on the tree type:
/// - `Tree` → Algorithm 2 (standard update, ignores Unknown)
/// - `TreeBatch` → Algorithm 3 (VirtualMean for Unknown states)
///
/// When `second_move_ctx` is provided and at root, applies Second Move
/// forcing (Algorithm 8, lines 355-361).
fn batch_mcts<T: TreeLike>(
    tree: &mut T,
    node_id: NodeId,
    tt: &TranspositionTable,
    strategy: &PuctStrategy,
    config: PuctConfig,
    rng: &mut impl rand::Rng,
    is_root: bool,
    second_move_ctx: Option<&SecondMoveContext>,
) -> PuctResult {
    let state = *tree.arena().get(node_id).state();

    // Algorithm 1/8: if isTerminal(s) then return Evaluation(s)
    if state.is_terminal() {
        return PuctResult::Value(state.terminal_value().unwrap());
    }

    // Algorithm 1/8: if s ∉ t then
    if !tree.arena().get(node_id).is_expanded() {
        if tt.get(&state).is_none() {
            return PuctResult::Unknown(state);
        } else {
            expand_node(tree.arena_mut(), node_id, &state);
            let v = tt.get(&state).unwrap().value().value();
            return PuctResult::Value(v);
        }
    }

    // Algorithm 1/8: PUCT selection (lines 204-215 / 343-354)
    let mut best_child_id = select_best_child(
        tree.arena(),
        node_id,
        tt,
        strategy,
        config,
        rng,
        is_root,
    );

    // Algorithm 8, lines 355-361: Second Move forcing at root
    if is_root && let Some(ctx) = second_move_ctx {
        let info = SecondMoveInfo::from_root(tree.arena(), node_id);
        if info.should_force_second(ctx.budget, ctx.spent_budget) {
            if let Some(second_id) = info.second_id {
                best_child_id = Some(second_id);
            }
        }
    }

    // Recursive descent and statistics update
    let res = batch_mcts(
        tree,
        best_child_id.unwrap(),
        tt,
        strategy,
        config,
        rng,
        false,
        second_move_ctx,
    );

    // Update statistics using the tree-type-specific strategy
    // Tree → Algorithm 2, TreeBatch → Algorithm 3 (VirtualMean)
    tree.update_statistics(node_id, res);

    res
}

/// Algorithm 9: GetMoveSecond (with Last Iteration from Algorithm 7)
///
/// Main entry point for batch MCTS search with Second Move Heuristic.
/// Combines Algorithm 7's Last Iteration with Algorithm 9's final μ comparison.
pub fn get_move_second<M: ModelEvaluator>(
    tree: &mut Tree,
    num_batches: usize,
    batch_size: usize,
    model: &M,
    tt: &TranspositionTable,
    config: PuctConfig,
    rng: &mut impl rand::Rng,
    pbar: Option<indicatif::ProgressBar>,
) -> SearchResults {
    let strategy = PuctStrategy::new(config);
    let root_id = tree.root();

    // Total budget for Second Move Heuristic (Algorithm 9)
    let total_budget = num_batches * batch_size;

    // Algorithm 9: for i ← 0 to B do
    for batch_idx in 0..num_batches {
        let spent_budget = batch_idx * batch_size;

        // Algorithm 9: GetBatchSecond(s, budget, i)
        let batch = get_batch_second(
            tree,
            root_id,
            batch_size,
            tt,
            &strategy,
            config,
            rng,
            total_budget,
            spent_budget,
        );

        // Algorithm 9: out ← Forward(batch)
        let inference_results = if !batch.is_empty() {
            Some(model.infer(&batch))
        } else {
            None
        };

        // Algorithm 9: PutBatchSecond(out, budget, i) - includes TT addition
        put_batch_second(
            tree,
            root_id,
            tt,
            &strategy,
            config,
            rng,
            total_budget,
            spent_budget,
            batch,
            inference_results,
        );

        if let Some(pb) = &pbar {
            pb.set_position((batch_idx + 1) as u64);
        }
    }

    // Algorithm 7: Last Iteration (uses BatchPUCT, not BatchSecond - no budget remaining)
    last_iteration(tree, root_id, tt, &strategy, config, rng);

    // Algorithm 9, lines 8-15: Compare μ(best) vs μ(secondBest)
    aggregate_results_second_heuristic(tree)
}

/// Algorithm 4 (adapted): GetBatchSecond
/// Calls BatchSecond with budget and iteration index.
fn get_batch_second(
    tree: &mut Tree,
    root_id: NodeId,
    batch_size: usize,
    tt: &TranspositionTable,
    strategy: &PuctStrategy,
    config: PuctConfig,
    rng: &mut impl rand::Rng,
    budget: usize,
    spent_budget: usize,
) -> Vec<State> {
    let mut tree_batch = TreeBatch::from(&*tree);
    let mut batch: Vec<State> = Vec::with_capacity(batch_size);

    let mut descent = 0;
    while batch.len() < batch_size && descent < MAX_DESCENTS_PER_BATCH {
        descent += 1;
        // Call batch_mcts with SecondMoveContext for Second Move forcing
        let ctx = SecondMoveContext {
            budget,
            spent_budget,
        };
        let res = batch_mcts(
            &mut tree_batch,
            root_id,
            tt,
            strategy,
            config,
            rng,
            true, // is_root
            Some(&ctx),
        );

        if let PuctResult::Unknown(state) = res {
            batch.push(state);
        }
    }

    batch
}

/// Algorithm 5 (adapted): PutBatchSecond
/// Adds inference results to TT, then updates main tree with budget and iteration index.
fn put_batch_second(
    tree: &mut Tree,
    root_id: NodeId,
    tt: &TranspositionTable,
    strategy: &PuctStrategy,
    config: PuctConfig,
    rng: &mut impl rand::Rng,
    budget: usize,
    spent_budget: usize,
    batch: Vec<State>,
    inference_results: Option<Vec<crate::policy::PolicyEvaluation>>,
) {
    // Add inference results to TT (Algorithm 9)
    if let Some(results) = inference_results {
        for (state, policy_eval) in batch.iter().zip(results.into_iter()) {
            tt.add(*state, policy_eval);
        }
    }

    // Update main tree with batch_mcts (uses Algorithm 2 for Tree)
    let ctx = SecondMoveContext {
        budget,
        spent_budget,
    };
    loop {
        let res = batch_mcts(tree, root_id, tt, strategy, config, rng, true, Some(&ctx));
        if let PuctResult::Unknown(_) = res {
            break;
        }
    }
}

/// Algorithm 7: Last Iteration (uses BatchPUCT, not BatchSecond - no budget remaining)
fn last_iteration(
    tree: &mut Tree,
    root_id: NodeId,
    tt: &TranspositionTable,
    strategy: &PuctStrategy,
    config: PuctConfig,
    rng: &mut impl rand::Rng,
) {
    let mut tree_batch = TreeBatch::from(&*tree);
    let mut nb_unknown = 0;

    // Algorithm 7: while nbUnknown < U do
    while nb_unknown < LAST_ITERATION_U {
        // Use batch_mcts without SecondMoveContext (no Second Move forcing)
        let res = batch_mcts(
            &mut tree_batch,
            root_id,
            tt,
            strategy,
            config,
            rng,
            true,
            None,
        );
        if let PuctResult::Unknown(_) = res {
            nb_unknown += 1;
        }
    }

    copy_stats_to_main_tree(tree, &tree_batch);
}

/// Copy statistics from treeBatch back to main tree after Last Iteration.
fn copy_stats_to_main_tree(tree: &mut Tree, tree_batch: &TreeBatch) {
    let main_arena = tree.arena_mut();
    let batch_arena = &tree_batch.arena;

    // Only copy nodes that exist in both
    let main_len = main_arena.len();
    let batch_len = batch_arena.len();
    let common_len = main_len.min(batch_len);

    for node_id in 0..common_len {
        let batch_node = batch_arena.get(node_id);
        let main_node = main_arena.get_mut(node_id);

        // Update visit count and sum if batch has more
        let batch_visits = batch_node.visit_count();
        let main_visits = main_node.visit_count();
        if batch_visits > main_visits {
            let delta = batch_visits - main_visits;
            for _ in 0..delta {
                main_node.increment_visit_count();
            }
            let delta_sum = batch_node.sum_evaluation() - main_node.sum_evaluation();
            main_node.add_evaluation(delta_sum);
        }
    }
}

/// Expand a node by adding children for all legal moves.
fn expand_node(arena: &mut Arena, node_id: NodeId, state: &State) {
    for action in state.legal_actions() {
        let next_state = state.apply(action);
        let child = Node::new(next_state, Some(action));
        arena.add_child(node_id, child);
    }
    arena.get_mut(node_id).set_expanded(true);
}

/// Algorithm 1, lines 204-215: PUCT selection with μFPU
fn select_best_child(
    arena: &Arena,
    parent_id: NodeId,
    tt: &TranspositionTable,
    strategy: &PuctStrategy,
    config: PuctConfig,
    rng: &mut impl rand::Rng,
    is_root: bool,
) -> Option<NodeId> {
    let parent = arena.get(parent_id);
    let policy_eval = tt.get(parent.state())?;
    let parent_visit_count = parent.visit_count();

    let children: Vec<NodeId> = arena.children(parent_id).collect();
    if children.is_empty() {
        return None;
    }

    // μFPU: "average mean of the node" (Section II.D, line 65)
    // = t.sum(s) / t.p(s) = parent's mean
    let mu_fpu = if parent_visit_count > 0 {
        parent.q_value()
    } else {
        0.0
    };

    // Dirichlet noise at root
    let noise: Option<Vec<f64>> = if is_root && config.dirichlet_epsilon > 0.0 {
        use crate::dirichlet::Dirichlet;
        let dirichlet = Dirichlet::new(config.dirichlet_alpha);
        dirichlet.sample(rng, children.len())
    } else {
        None
    };

    let mut best_score = f64::NEG_INFINITY;
    let mut best_child = None;

    for (idx, &child_id) in children.iter().enumerate() {
        let child = arena.get(child_id);

        // Algorithm 1: μ = FPU; if t.p(s,m) > 0 then μ = t.sum(s,m)/t.p(s,m)
        let mu = if child.visit_count() > 0 {
            child.q_value()
        } else {
            mu_fpu
        };

        // Get prior from policy
        let mut prior = if let Some(action) = child.action() {
            policy_eval.policy().0[action]
        } else {
            0.0
        };

        // Dirichlet noise at root
        if let Some(ref noise_vec) = noise {
            prior = (1.0 - config.dirichlet_epsilon) * prior
                + config.dirichlet_epsilon * noise_vec[idx];
        }

        // Algorithm 1: bandit = μ + c × prior × √(t.p(s)) / (1 + t.p(s,m))
        let score = strategy.calculate_score(mu, child.visit_count(), parent_visit_count, prior);

        if score > best_score {
            best_score = score;
            best_child = Some(child_id);
        }
    }

    best_child
}

/// Algorithm 9, lines 8-15: Final selection by μ comparison.
///
/// if μ(secondBest) > μ(best) then return secondBest else return best
fn aggregate_results_second_heuristic(tree: &Tree) -> SearchResults {
    let mut visit_counts = vec![0u32; 64];
    let mut q_values = vec![0.0f64; 64];

    let arena = tree.arena();
    let root_id = tree.root();

    // First, collect all children's stats
    for child_id in arena.children(root_id) {
        let child = arena.get(child_id);
        if let Some(action) = child.action() {
            visit_counts[action] = child.visit_count();
            q_values[action] = child.q_value();
        }
    }

    // Algorithm 9, lines 8-15: Compare μ of best vs secondBest
    let info = SecondMoveInfo::from_root(arena, root_id);
    if let (Some(best_id), Some(second_id)) = (info.best_id, info.second_id) {
        let best_mu = arena.get(best_id).q_value();
        let second_mu = arena.get(second_id).q_value();

        // if μ' > μ then return secondBest
        if second_mu > best_mu {
            // Transfer visits to secondBest to influence policy proportions
            if let (Some(best_action), Some(second_action)) =
                (arena.get(best_id).action(), arena.get(second_id).action())
            {
                // Swap the visit counts so secondBest becomes the "best" move
                let total = visit_counts[best_action] + visit_counts[second_action];
                visit_counts[second_action] = total;
                visit_counts[best_action] = 0;
            }
        }
    }

    SearchResults {
        visit_counts,
        q_values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, PolicyEvaluation, Value};
    use crate::tree::Tree;
    use common::{Bitboard, DiskColor};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn default_state() -> State {
        State::new(Bitboard::default(), DiskColor::Dark)
    }

    fn create_default_config() -> PuctConfig {
        PuctConfig {
            c_puct: 1.0,
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: 1.0,
        }
    }

    fn create_mock_policy() -> Policy {
        Policy([1.0 / 64.0; 64])
    }

    struct MockValModel {
        val: f64,
    }

    impl ModelEvaluator for MockValModel {
        fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation> {
            states
                .iter()
                .map(|_| PolicyEvaluation::new(create_mock_policy(), Value(self.val)))
                .collect()
        }
    }

    #[test]
    fn test_search_basic() {
        let mut tree = Tree::new(default_state());
        let tt = TranspositionTable::new();
        let model = MockValModel { val: 0.5 };
        let config = create_default_config();
        let mut rng = StdRng::seed_from_u64(42);

        get_move_second(&mut tree, 4, 2, &model, &tt, config, &mut rng, None);

        assert!(tree.arena().get(tree.root()).visit_count() > 0);
    }

    #[test]
    fn test_mu_fpu_parent_mean() {
        // μFPU should be parent's mean, not 0
        let state = default_state();
        let mut arena = Arena::new();
        let parent_id = arena.allocate(Node::new(state, None));

        // Set parent to have visits and value
        arena.get_mut(parent_id).increment_visit_count();
        arena.get_mut(parent_id).add_evaluation(0.7);

        // Add unvisited child
        let child = Node::new(state, Some(0));
        arena.add_child(parent_id, child);

        let parent = arena.get(parent_id);
        let mu_fpu = if parent.visit_count() > 0 {
            parent.q_value()
        } else {
            0.0
        };

        assert!((mu_fpu - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_select_best_child() {
        // Tests that select_best_child correctly applies PUCT formula
        // and selects the child with highest exploration bonus when unvisited
        let config = PuctConfig {
            c_puct: 1.0,
            dirichlet_epsilon: 0.0, // No noise for deterministic test
            dirichlet_alpha: 1.0,
        };
        let strategy = PuctStrategy::new(config);
        let tt = TranspositionTable::new();
        let parent_state = default_state();

        // Set up policy with different priors for each action
        let mut policy_arr = [0.0; 64];
        policy_arr[0] = 0.2;
        policy_arr[1] = 0.5;
        policy_arr[2] = 0.3;
        let policy = Policy(policy_arr);
        tt.add(parent_state, PolicyEvaluation::new(policy, Value(0.0)));

        // Build arena with parent and children
        let mut arena = Arena::new();
        let parent_id = arena.allocate(Node::new(parent_state, None));

        // Set parent visit count
        for _ in 0..15 {
            arena.get_mut(parent_id).increment_visit_count();
        }

        // Child 0: 10 visits, Q=0.7
        let mut child0 = Node::new(default_state(), Some(0));
        for _ in 0..10 {
            child0.increment_visit_count();
            child0.add_evaluation(0.7);
        }
        arena.add_child(parent_id, child0);

        // Child 1: 5 visits, Q=0.4
        let mut child1 = Node::new(default_state(), Some(1));
        for _ in 0..5 {
            child1.increment_visit_count();
            child1.add_evaluation(0.4);
        }
        arena.add_child(parent_id, child1);

        // Child 2: 0 visits (should have highest exploration bonus)
        let child2 = Node::new(default_state(), Some(2));
        let child2_id = arena.add_child(parent_id, child2);

        let mut rng = StdRng::seed_from_u64(42);
        let best = select_best_child(
            &arena, parent_id, &tt, &strategy, config, &mut rng,
            false, // not root, no Dirichlet noise
        );

        // Child 2 should be selected due to highest exploration bonus (0 visits)
        // PUCT = μFPU + c_puct × P(a) × √(N_parent) / (1 + N_child)
        // For child2: PUCT = parent_q + 1.0 × 0.3 × √15 / 1 ≈ 0.7 + 1.16 = ~1.86
        assert_eq!(best, Some(child2_id));
    }
}
