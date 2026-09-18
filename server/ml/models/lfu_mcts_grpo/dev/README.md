# Tree-Structured Terminal Trajectories for Othello Self-Play: Experimental Plan

[日本語版](README_ja.md)

Revised 2026-09-18. This document specifies a research question and experimental protocol, not demonstrated results. The existing `mcts-grpo` implementation does not implement this plan; implementation and validation remain future work.

## 1. Research question and scope

**Can generating tree-structured trajectories from the same board position all the way to game termination, and learning the relative quality of branches from their outcomes, produce a stronger agent with less computation than conventional self-play?**

Supervision comes from actual terminal wins, draws, and losses. This replaces the earlier proposal to add an auxiliary loss based on MCTS Q-values estimated at intermediate positions. The tree generates training data; it does not require search during evaluation or deployment.

The primary baseline is independent self-play trajectories with policy and value updates from terminal returns. An AlphaZero-style system is an additional practical baseline. An improvement over independent self-play does not establish superiority over AlphaZero.

Separate three hypotheses:

- H1: Sharing computation and storage for common prefixes reduces the cost of generating a specified collection of terminal trajectories.
- H2: Comparing sibling branches from the same position provides more effective updates than ordinary terminal-return updates.
- H3: After accounting for correlated trajectories, reduced diversity, tree management, and training costs, the total cost of reaching a target playing strength decreases.

H1 alone is not research success. More terminal leaves without improved playing strength do not support H3.

## 2. Relationship to Tree-GRPO and design choices

The inspiration from Tree-GRPO is shared-prefix trajectory generation and relative learning from terminal rewards. The original paper studies groups of LLM trajectories for the same input. This plan adapts those ideas to an alternating-turn game; it does not inherit the paper's theoretical guarantees or empirical results.

| Approach | Benefit | Limitation | Timing |
|---|---|---|---|
| Shallow trees with fixed width and level count, with locations drawn separately for each tree before rollout | Avoids concentrating on specific move numbers while keeping sampling and budgets traceable | Some expansions may be uninformative | Primary experiment |
| Adaptive expansion based on uncertainty or other signals | May concentrate resources on informative positions | Introduces selection bias and correction requirements | After foundational experiments |
| MCTS trees guided by PUCT or similar search | Can combine learning with strong action selection | Makes sharing, search, and learning effects harder to separate | Follow-up research |

Initially, neither predicted values nor observed outcomes select expansion locations. The value head supplies a training baseline only; it never substitutes for playing a trajectory to termination.

## 3. Minimal tree generation

1. Freeze the collection policy as π_old at the start of each round. Both players sample legal moves from it. Weights do not change during collection.
2. Draw root positions using a common rule across methods. The initial proposal is 0–8 random legal placements from the initial board. Separate training, tuning, and final-evaluation seed sets.
3. Use width 4 and at most 2 branching levels. If the root has H empty squares, uniformly sample `min(2, H)` distinct locations without replacement from `{0, …, H−1}` before rollout. Location d means immediately before choosing the next move after d disk placements from the root; d=0 branches at the root. All paths within a tree share its sampled schedule; each new tree draws a new schedule. Passes do not count as placements.
4. At each scheduled branching node, draw 4 actions independently with replacement from π_old. Repeated actions are allowed; each sample uses an independent continuation random stream. Do not force four distinct actions.
5. Generate single paths between branching locations and play every path to termination. Each small tree has at most 16 terminal leaves. If a path terminates before a scheduled location, skip that expansion. Do not relocate skipped expansions or redraw locations after observing outcomes. H=0 has no branching and is handled as terminal.
6. Reuse the policy output at a shared parent, but do not collapse sibling samples that selected the same action into one terminal trial. Record sample counts and distinct-action counts separately.

Schedule sampling uses a separate random stream from action sampling and is independent of value estimates and observed outcomes. Locations are drawn without replacement; actions are drawn with replacement. Conditional on the sampled schedule, sibling continuation trials use independent randomness.

Process forced passes and terminal detection before checking the schedule immediately before the next disk placement. Execute each scheduled expansion at most once per path. A pass leaves the placement count unchanged but must not trigger another expansion at the same location.

For illustration, if all paths last 60 placements and the sampled locations happen to be 20 and 40, the tree generates 20 + 4×20 + 16×20 = 420 new transitions, compared with 16×60 = 960 for independent paths. This is not a prescription to fix the primary experiment at moves 20 and 40. It is an idealized transition count excluding tree management, training, and batching effects, not a wall-clock speedup claim.

Othello inference depends on the current board. Unlike sharing long LLM histories, the main savings are policy evaluations and environment transitions along a common prefix. Give the independent baseline normal batched inference rather than an unnecessarily sequential implementation.

Uniform schedule sampling does not make the board-position distribution uniform. Reachability depends on the self-play policy, and early termination prevents some deep expansions. Record scheduled expansions, executed expansions, and sibling-comparison update weights by root-relative move count, phase measured by occupied squares, player to move, and legal-action count. Measure the resulting bias; do not count leaves as independent samples.

Compare no branching, one randomly located level, and two randomly located levels. Separately, hold width and level count fixed when ablating location selection: uniform sampling over all candidate locations, phase-stratified sampling, and fixed locations. The initial stratified condition partitions candidate locations into three contiguous, approximately equal intervals, chooses up to two nonempty intervals with equal probability, then samples one location uniformly from each. The fixed condition uses locations 20 and 40 without replacing out-of-range or unreached locations. B and C use the same schedule-generation rule. Report differences in executed expansions and actual cost. Defer large hyperparameter searches until the minimal configuration has demonstrated value.

## 4. Outcome aggregation and learning objective

### 4.1 Perspective and terminal returns

Store each terminal outcome z from Black's perspective: +1 for a Black win, 0 for a draw, and −1 for a Black loss. At each comparison, express every return from the parent position's player-to-move perspective. Passes make depth parity or unconditional sign reversal incorrect.

The return R_j of branch sample j is the conditional average of its descendant terminal outcomes. Recursively average child samples equally at each node, retaining Black's perspective during aggregation and converting when constructing a training target. If early termination produces unequal descendant counts, do not flatten all leaves and accidentally change branch weights.

This estimates Monte Carlo returns under the frozen self-play policy π_old, not minimax results against optimal play or absolute move strength.

### 4.2 Initial advantage estimator

For K sibling samples at the same position, use a leave-one-out baseline:

    A_j = R_j − (1 / (K−1)) Σ_{k≠j} R_k

Initially keep the fixed reward scale of −1 to +1 rather than dividing by the group's standard deviation. Avoid amplifying small differences from small samples.

The intended benefit is reduced update variance through same-position comparisons, not changing the expected policy-gradient objective. This may not outperform a well-trained value baseline; the B/C comparison tests that question.

At nodes without sibling samples, use A = R − V_old(s). V_old is the frozen collection model's value estimate, not a replacement for the actual terminal return. Policy and value heads share a small network.

If all sibling returns are equal, their relative advantages are zero. Do not inject noise to manufacture preferences. Record the fraction of such groups.

This estimator is a proposal for this experiment, not a reproduction of the original intra/inter-tree advantage. Do not normalize Q-values across unrelated positions and players in a minibatch. Comparisons across multiple trees for the same root, closer to the original formulation, are a later ablation.

### 4.3 Policy updates and shared-prefix weights

Start with small PPO-style updates:

    r_j = π_θ(a_j | s) / π_old(a_j | s)
    f_j = min(r_j A_j, clip(r_j,1−ε,1＋ε) A_j)
    L_policy = − (1 / (60 × N_roots)) Σ_tree Σ_edge w_edge f_edge

Store the log of the actual sampling probability, including legal-action masking and temperature. Apply the same mask and temperature transformation to the numerator. Initially fix temperature to 1 and test that all ratios equal 1 at θ=old. Average over sampled actions rather than summing ratios over the entire action set without probability weights. Detach advantages and old probabilities from gradients.

Do not update a shared prefix once per terminal leaf. Assign each root weight 1 and divide the path weight by K for each of its K sampled branches. Sum weighted edge losses, divide by the common fixed constant 60, and average over roots. The K samples at a branching node sum to their parent's weight. For A with K independent paths from a root, assign each path weight 1/K and use the same constant 60. Do not normalize by realized trajectory length or leaf count. Branching must not multiply a shared move's weight or introduce a length-dependent objective.

Train the value head with MSE to the mean terminal-return target using the same perspectives and path weights. If an entropy term is included, use identical coefficients across groups.

This is a PPO-style surrogate objective on trees collected under a frozen policy. It does not assert an unbiased gradient for the complete self-play process or monotonic improvement in playing strength. Validate the conditional expectation of the leave-one-out baseline, given a sampled schedule, and the interpretation of path weights in a small enumerable game.

Initial settings are clipping width 0.2 and one epoch per collected dataset. Measure update size before trying additional epochs. Persisting old trees does not justify unrestricted mixing of old model generations into policy updates.

## 5. Comparison experiments

| Group | Generation | Learning | Purpose |
|---|---|---|---|
| A: independent self-play | Independent terminal trajectories | PPO with terminal return minus V_old | Primary baseline |
| B: tree self-play | The proposed tree structure | PPO with terminal return minus V_old | Tree collection and return aggregation |
| C: proposed method | The same structure as B | Sibling comparisons at branches; otherwise identical to B | Additional benefit of branch comparisons |
| D: AlphaZero-style | Validated MCTS self-play | Visit-distribution policy loss plus terminal-outcome value loss | Practical search-based comparison |

Match networks, initial weights, optimizers, root-generation rules, legal masks, evaluation, and tuning budgets for A/B/C. B/C use randomized locations per tree in the primary experiment. First compare short B/C updates from the same saved trees and model initialization; then run separate online training experiments.

Use the pilot to set a common target for new environment transitions per collection round. Finish small trees or games already in progress before updating, and count target overshoot in actual cost. Include the one-epoch training cost. Report differences in root count, leaf count, and update count. If using ordinary inference batching or same-generation board-evaluation caches, give A the same opportunity and memory limits.

For H1, replay the same sampled tree and randomness with shared-prefix reuse enabled and disabled. Match output data and learning weights; compare generation time, inference count, and memory. This implementation control alone does not establish a playing-strength advantage over independent self-play.

Do not use the current MCTS implementation unchanged as D. Validate perspectives, terminal handling, concurrency, and visit counts first. Beating A without beating D is not superiority over AlphaZero.

## 6. Compute, playing strength, and statistics

The primary metric is collection plus training time to a fixed target strength on the same machine. Treat runs that do not reach it within budget as non-attainment. Also compare equal-budget scores and area under the learning curve.

Record separately:

- Collection, updates, persistence, and evaluation time. Report research time excluding evaluation and operational time including it.
- Positions evaluated, training forward/backward sample counts, and batch-size distributions. A backward pass or a different model is not an equivalent single inference.
- Completed trees, terminal outcomes, distinct positions/actions, sharing ratio, zero-advantage groups, and expansion/update distributions by phase, player, and legal-action count.
- RSS, peak device memory, waiting time, and persisted data size.

Primary evaluation uses the policy without search to measure learned playing strength, with action selection fixed across groups. For the practical comparison including AlphaZero, additionally report equal per-move search-time budgets separately from policy-only results.

Opponents include random, greedy, mobility, and independently prepared frozen reference agents. Predefine opponents, aggregate weights, and tuning/final opening sets. Play both colors from each opening and use score = (wins + 0.5×draws) / games. Preserve opponent-specific results.

Initial planning values are 3 training seeds for small comparisons and at least 5 for confirmation. A starting evaluation budget is 100 openings × 2 colors = 200 games per opponent per seed. Finalize counts using pilot variance and cost before confirmation, rather than adding games until a favorable result appears.

Leaves from one tree are not independent replications. Report training reproducibility across seeds and account for paired-opening dependence in evaluation uncertainty. Do not use final openings for tuning or best-model selection. Check overlap using board, player, and symmetries. Different seeds do not guarantee unseen positions; identify shared standard positions explicitly. Do not claim evaluation positions were never reached during training or that the experiment establishes generalization to unseen boards.

After a pilot, fix an attainable target before confirmation; an example is 70% score against the frozen pool. A candidate practical threshold is a 20% time reduction. These are provisional choices, not findings. Non-attainment or wide uncertainty leaves superiority unconfirmed.

For budget T, save and evaluate at the first safe update boundary after each 0.1T increment of collection plus training time. Use actual checkpoint times and report overshoot. Detected attainment is the second of two consecutive tuning evaluations at or above target. Match schedules across groups and do not select models on the final set.

Do not average attainment times only over successful seeds. Report attainment rates and the all-seed average of min(detected attainment time, T), interpreting the latter as a restricted-time metric within T. The candidate adoption rule requires no decrease in attainment rate, a point estimate of at least 20% improvement in this metric, and a 95% interval for improvement excluding zero. This does not statistically guarantee at least 20% savings. Insufficient seed precision means unconfirmed results. If both groups frequently miss the target, report fixed-budget scores and curve area without claiming a target-attainment cost advantage.

## 7. Execution stages and stopping rules

### Stage 0: Semantic correctness

Check terminal outcomes, passes, color exchange, legal-only probabilities, zero relative advantage for equal sibling returns, and increased probability for winning branches in controlled cases. Verify equal loss/path weights with shared execution enabled and disabled, and identical per-position network depth and recurrent iteration counts during training and evaluation.

Check schedule reproducibility and bounds, location versus action sampling, no resampling after early termination, H=0 and H=1, and move-range coverage.

In an enumerable game or endgame, compare Monte Carlo returns with exact expectations under the frozen policy, not minimax values. Freeze the opponent and differentiate only one player's policy; compare the conditional policy gradient and leave-one-out baseline with enumeration. Repeat for both colors.

### Stage 1: System viability

Start with a common small network (candidate: four residual blocks, width 64), AdamW, and batch size 32. Measure memory and time. Defer URM and Muon comparisons. Batch small trees and choose inference batch size/concurrency from measurements.

Before long runs, demonstrate recovery after forced termination, exclusion of incomplete trees, and no duplicate consumption of completed data.

### Stage 2: Small learning-signal comparison

Compare B/C on identical trees and initial models, measuring legal probabilities, KL, clipping fraction, and expected terminal returns. Freeze the opponent at π_old and replace only the evaluated player with the updated model. Evaluate Black and White separately; self-play win rate with both players changing is not an improvement metric. Then run A/B/C with 3 seeds and equal short budgets. Beating random is a smoke check, not research success.

### Stage 3: Efficiency confirmation

Freeze settings and thresholds after the pilot; compare A/B/C with at least 5 seeds and equal total budgets. Compare no branching, one random level, and two random levels. Hold width and level count fixed for location ablations. Avoid unrestricted width/depth searches. If C does not beat B, report no confirmed additional benefit from relative comparison. If B/C lose to A, report tree generation's disadvantages, including correlation and diversity loss.

### Stage 4: External comparison and extensions

Add validated AlphaZero-style D. Subsequently investigate adaptive branching, historical-opponent pools, normalization closer to the original multiple-tree formulation, or URM, one change at a time. Do not simultaneously change search and architecture.

## 8. Persistence and recovery

Separate collection, training, and evaluation into explicit phases on one machine. A distributed system is not initially required.

- Persist each completed tree with parent IDs, positions, players, actions, old log probabilities/values, model generation, schedule, path weights, terminal results, randomness metadata, and compute cost.
- Publish complete trees using temporary files plus atomic rename or SQLite transactions. Never train on partially written trees.
- Save model, optimizer, scheduler, RNG, model generation, consumed-data position, phase, accumulated compute, and configuration as a consistent checkpoint.
- Tie model state and consumption to the same manifest. After interruption, reuse completed trees and discard only incomplete small trees.
- Finalize a checkpoint before evaluation so evaluation failure cannot erase updates. Also save periodically during training by time or update count.
- Retain old trees for reproducibility and analysis. Storage availability and eligibility for current-policy updates are separate concerns.

## 9. Settings to freeze and reporting

The primary baseline is independent self-play. Primary tree experiments draw random locations per tree; fixed locations are an ablation only. Preserve terminal-reward supervision, frozen evaluation opponents, and accounting for compute and correlation.

Before confirmation, freeze network/optimizer settings, collection target, total budget, branching conditions, seeds, opponents/openings, evaluation counts, attainment thresholds, and statistical procedure in machine-readable configuration. Distinguish pilot from confirmation results, and do not retrospectively select favorable seeds or checkpoints.

No confirmation results for this plan are available yet. Report hypotheses separately, including H1 alone succeeding, B helping without additional benefit from C, or improvement over independent self-play but not over AlphaZero.

## References

- [Tree Search for LLM Agent Reinforcement Learning, §3](https://arxiv.org/html/2509.21240v1#S3): shared trajectories and relative learning from terminal rewards.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347): the policy-update surrogate objective.
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815): the additional AlphaZero-style comparison.
