# MCTS-GRPO: Efficient Self-Play Reinforcement Learning via Relative Policy Optimization using Monte Carlo Tree Search

[**日本語版はこちら (Click here for the Japanese version)**](README_ja.md)

## Abstract

This research introduces MCTS-GRPO, a novel method designed to enhance the sample efficiency of self-play reinforcement learning frameworks such as AlphaZero. In conventional AlphaZero, the training signal relies primarily on the final game outcome, resulting in sparse rewards and requiring an enormous number of self-play games for an agent to learn effectively. To address this challenge, we leverage the rich information generated during the Monte Carlo Tree Search (MCTS) process to create a denser training signal. Specifically, we incorporate a Group Relative Policy Optimization (GRPO) loss term to update the policy head. This is achieved by comparing the action-values (Q-values) of multiple moves evaluated from the same board state by MCTS and directly training the network on their relative superiority. This allows the model to learn not just from the final outcome but also to develop a more nuanced understanding of the quality of moves at each state, thereby accelerating the learning process. We aim to validate this approach by applying it to the game of Othello and measuring the improvement in learning speed compared to a standard AlphaZero baseline.

## Introduction

Since the advent of AlphaGo, the combination of self-play reinforcement learning and Monte Carlo Tree Search (MCTS) has become the de facto standard for achieving superhuman performance in perfect information games like Go, chess, and shogi. At the core of this framework, AlphaZero employs a neural network to provide policy and value guidance to the MCTS engine. The search results from MCTS are then used as training data to continuously improve the network.

However, a significant challenge remains in this powerful learning loop: sample efficiency. The primary supervisory signal for the value network is the singular outcome of the game—a win, loss, or draw. This single label is applied to every state visited throughout the game's dozens of moves, leading to a substantial loss of information and slowing down the convergence of the learning process.

This problem motivates our work, which draws inspiration from the concepts presented in Tree-GRPO. Tree-GRPO is a technique that improves sample efficiency by generating multiple simulation trajectories from a given state and using the relative difference in their final rewards as a training signal. We adapt this idea to the AlphaZero MCTS framework. The MCTS process is, in essence, an evaluation of countless partial trajectories branching from a single board state. The Q-value for each action calculated by MCTS can be seen as a condensed, high-quality evaluation of the promise of that trajectory.

The MCTS-GRPO method proposed in this paper uses these MCTS search results (specifically, the Q-values) as training data, teaching the policy network to reproduce the relative evaluations between promising and suboptimal moves. We hypothesize that this richer supervisory signal will enable faster learning compared to the standard AlphaZero approach.

## MCTS-GRPO

MCTS-GRPO is built upon the standard AlphaZero learning cycle (self-play and training) and is implemented as an extension to the loss function used in the training phase.

### 1. Self-Play and Storing Search Data

As in AlphaZero, self-play games are generated using the latest neural network. At each turn, MCTS is executed to determine the next move. When storing training data, in addition to the standard tuple `(Board State S, MCTS Search Probabilities π, Final Game Outcome Z)`, we also save the set of **Q-values for all actions `{Q(S, a)}`** at the root node to a replay buffer.

### 2. The Extended Loss Function

During training, mini-batches are sampled from the replay buffer. For each sample `(S, π, Z, {Q})`, the following three loss components are calculated.

#### a. Value Loss

Consistent with standard AlphaZero, we compute the mean squared error between the network's value output `v(S)` and the actual game outcome `Z`.
$$ L_{value} = (v(S) - Z)^2 $$

#### b. Policy Loss

Also consistent with standard AlphaZero, we compute the cross-entropy between the network's policy output `p(S)` and the MCTS visit count distribution `π`.
$$ L_{policy} = - \pi \cdot \log(p(S)) $$

#### c. MCTS-GRPO Loss

This is the core of our proposed method. We use the stored set of Q-values `{Q(S, a)}` to compute a relative policy loss. First, we identify the action `a_best` with the highest Q-value. Next, we sample multiple suboptimal actions `a_suboptimal` from the remaining legal moves. For each pair `(a_best, a_suboptimal)`, we compute the following loss:

$$ L_{GRPO\_pair} = \log(1 + \exp(-(\log p(S, a_{best}) - \log p(S, a_{suboptimal})))) $$

This loss encourages the network's policy output to assign a higher probability to `a_best` than to `a_suboptimal`. The final MCTS-GRPO loss is the average loss over all sampled pairs.

### 3. Total Loss

The final loss function is a weighted sum of these three components:
$$ L_{total} = L_{value} + L_{policy} + \lambda_{GRPO} \cdot L_{GRPO} $$
where `λ_GRPO` is a hyperparameter that balances the contribution of the GRPO loss.

## Experiments

### TODO

- **Game**: Othello (8x8)
- **Model Architecture**: ResNet architecture compliant with AlphaZero.
- **Baseline**: Standard AlphaZero implementation (i.e., λ_GRPO = 0).
- **Evaluation Metric**: The agent's strength (win rate against a benchmark AI) as a function of training steps.
- **Hyperparameters**:
  - Number of MCTS simulations
  - Batch size
  - Learning rate
  - λ_GRPO
  - ...

## Results and Discussion

### TODO

- Present a graph comparing the learning curves of the baseline and the proposed model.
- Provide a table showing the final win rates of each model against the benchmark AI.
- Discuss whether the MCTS-GRPO loss was particularly effective in the early stages of training or if it improved final convergence.
- Include a graph demonstrating the suppression of the periodic loss spike phenomenon.

## Conclusion

### TODO

- Based on the experimental results, state whether the hypothesis that MCTS-GRPO improves the sample efficiency of AlphaZero is supported.
- Discuss the limitations of the current study and future work, such as applying the method to other games.

## References

 (TODO: Add formal citation for the original Tree-GRPO paper)
