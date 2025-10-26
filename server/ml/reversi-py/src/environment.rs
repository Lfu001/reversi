use common::{Action, BitPosition, DiskColor, PutConfig, Table};
use ndarray::{s, Array};
use numpy::{prelude::*, PyArray, PyArray1, PyArray4, PyReadonlyArray3};
use pyo3::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use reversi_core::{
    action::{get_puttable_positions, PuttablePositions},
    controller::Controller,
};

/// A PyClass that represents a batch of parallel Reversi games for reinforcement learning.
///
/// This environment is designed to be highly efficient, processing multiple games
/// simultaneously using the Rayon library for parallel computation. It's suitable for
/// training machine learning agents that require high throughput.
#[pyclass]
struct ReversiEnvironment {
    /// A vector of `Table`s, each representing the state of a single Reversi game.
    tables: Vec<Table>,
    /// A vector of booleans, where `true` indicates that the game at the corresponding index has finished.
    dones: Vec<bool>,
    /// The number of games in the batch.
    batch_size: usize,
}

// Private implementation block for core logic not exposed to Python.
impl ReversiEnvironment {
    /// A private core function that handles the common logic for stepping the batch environment.
    ///
    /// This function iterates over all games in parallel. For each game, it determines the
    /// next action by calling the `action_selector` closure, applies the action,
    /// and updates the game state.
    ///
    /// # Arguments
    /// * `py` - The Python GIL token.
    /// * `actions` - A 3D numpy array of shape `(batch_size, 8, 8)` containing action probabilities.
    /// * `action_selector` - A closure that defines the policy for selecting an action. It takes a
    ///   slice of legal positions and the corresponding 2D action probabilities for a single game,
    ///   and returns the chosen `Position`.
    ///
    /// # Type Parameters
    /// * `F` - The type of the `action_selector` closure. It must be `Sync` and `Send` to be
    ///   safely used across multiple threads by Rayon.
    fn _step_batch_core<F>(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray3<f32>,
        action_selector: F,
    ) -> PyResult<StepBatchResult>
    where
        F: Fn(PuttablePositions, &ndarray::ArrayView2<'_, f32>) -> BitPosition + Sync + Send,
    {
        let actions_array = actions.as_array();

        // Process each game state in parallel using Rayon.
        let results: Vec<_> = self
            .tables
            .par_iter_mut()
            .zip(self.dones.par_iter_mut())
            .enumerate()
            .map(|(i, (table, done))| {
                // If the game is already done, skip processing and return its done status.
                if *done {
                    return Ok(true);
                }

                let turn = table.turn();
                let puttable_positions = get_puttable_positions(table.board(), turn);

                let action = if puttable_positions.is_empty() {
                    // If there are no legal moves, the only valid action is to pass the turn.
                    Action::PassTurn(turn)
                } else {
                    // Get the 2D action probabilities for the current game.
                    let game_actions = actions_array.slice(s![i, .., ..]);
                    // Use the provided strategy (closure) to select the next move.
                    let chosen_pos = action_selector(puttable_positions, &game_actions);
                    Action::PutDisk(PutConfig::new(turn, chosen_pos.into()))
                };

                // Apply the chosen action to the game board.
                match Controller::step(table, action) {
                    Ok(step_result) => {
                        // Check if the game has ended and update the `done` flag.
                        if step_result.judge_result.is_some() {
                            *done = true;
                        }
                        Ok(*done)
                    }
                    Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err)),
                }
            })
            .collect();

        // Collect results from all threads. If any thread returned an error, propagate it.
        let current_dones: Vec<bool> = results.into_iter().collect::<Result<_, _>>()?;

        // Get the new state of all games and return the results to Python.
        let next_states = self.get_state(py)?;
        let dones_array = Array::from_vec(current_dones).into_pyarray(py);

        Ok((next_states, dones_array.into()))
    }
}

/// A tuple containing the results of a batch step of the environment.
///
/// The tuple contains two elements:
/// - The next state of the environment as a `Py<PyArray4<f32>>`.
/// - A boolean array indicating whether each game in the batch has finished as a `Py<PyArray1<bool>>`.
type StepBatchResult = (Py<PyArray4<f32>>, Py<PyArray1<bool>>);

#[pymethods]
impl ReversiEnvironment {
    /// Creates a new `ReversiEnvironment` with a specified batch size.
    ///
    /// # Arguments
    /// * `batch_size` - The number of parallel games to simulate in this environment.
    #[new]
    fn new(batch_size: usize) -> Self {
        ReversiEnvironment {
            tables: (0..batch_size).map(|_| Table::default()).collect(),
            dones: vec![false; batch_size],
            batch_size,
        }
    }

    /// The number of games in the batch.
    #[getter]
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Resets all games in the environment to their initial state.
    ///
    /// This is typically called at the beginning of a new training episode.
    ///
    /// # Returns
    /// A `Py<PyArray4<f32>>` representing the initial state of the batch.
    fn reset(&mut self, py: Python<'_>) -> PyResult<Py<PyArray4<f32>>> {
        self.tables = (0..self.batch_size).map(|_| Table::default()).collect();
        self.dones = vec![false; self.batch_size];
        self.get_state(py)
    }

    /// Returns the current state of all games in the batch.
    ///
    /// The state is represented as a 4-dimensional numpy array with shape
    /// `(batch_size, 4, 8, 8)`. The four channels are:
    /// - Channel 0: Positions of the dark disks (1.0 if disk exists, 0.0 otherwise).
    /// - Channel 1: Positions of the light disks (1.0 if disk exists, 0.0 otherwise).
    /// - Channel 2: A plane indicating the current turn (1.0 for Dark, -1.0 for Light).
    /// - Channel 3: A plane indicating all legal moves for the current player (1.0 if move is legal, 0.0 otherwise).
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyArray4<f32>>> {
        // Pre-allocate a vector with the exact required capacity for performance.
        let mut state_vec = Vec::with_capacity(self.batch_size * 4 * 8 * 8);
        state_vec.resize(self.batch_size * 4 * 8 * 8, 0.0);

        for (batch_idx, table) in self.tables.iter().enumerate() {
            let turn = table.turn();
            let board = table.board();
            let puttable_positions = get_puttable_positions(board, turn);
            let turn_val = if turn == DiskColor::Dark { 1.0 } else { -1.0 };

            let idx_offset = batch_idx * 256;

            // Populate disk positions and legal moves.
            let dark_plane = board.dark_plane();
            let light_plane = board.light_plane();
            for i in 0..64 {
                let idx = idx_offset + i;
                let mask = 1u64 << (63 - i);
                if dark_plane & mask != 0 {
                    state_vec[idx] = 1.0;
                }
                if light_plane & mask != 0 {
                    state_vec[idx + 64] = 1.0;
                }
                state_vec[idx + 128] = turn_val;
                if puttable_positions.0 & mask != 0 {
                    state_vec[idx + 192] = 1.0;
                }
            }
        }

        let array = PyArray::from_vec(py, state_vec);
        Ok(array.reshape((self.batch_size, 4, 8, 8))?.into())
    }

    /// Steps the environment forward by one move for each game, selecting actions stochastically.
    ///
    /// This method is intended for **training**, as it promotes exploration. It treats the
    /// `actions` array as a probability distribution and samples a move for each game
    /// according to the weights of the legal moves.
    ///
    /// # Arguments
    /// * `actions`: A `numpy.ndarray` of shape `(batch_size, 8, 8)` and `dtype=float32`.
    ///
    /// # Returns
    /// A tuple `(next_states, dones)`.
    #[pyo3(name = "step_batch_stochastic")]
    fn step_batch_stochastic(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray3<f32>,
    ) -> PyResult<StepBatchResult> {
        // Define the action selection strategy: sample from the probability distribution.
        let selector = |puttable_positions: PuttablePositions,
                        game_actions: &ndarray::ArrayView2<'_, f32>| {
            let puttable_positions_vec = puttable_positions.to_vec();
            let weights: Vec<f32> = puttable_positions_vec
                .iter()
                .map(|pos| game_actions[[pos.row() as usize, pos.column() as usize]])
                .collect();

            // Create a weighted distribution for sampling.
            match WeightedIndex::new(&weights) {
                Ok(dist) => {
                    // Create a thread-local random number generator.
                    let mut rng = rand::rng();
                    // Sample an index from the distribution and return the corresponding position.
                    let idx = dist.sample(&mut rng);
                    BitPosition::from(puttable_positions_vec[idx])
                }
                // Fallback: if all legal moves have zero probability, just pick the first one.
                Err(_) => BitPosition::from(puttable_positions_vec[0]),
            }
        };

        // Delegate the core logic to the private batch processing function.
        self._step_batch_core(py, actions, selector)
    }

    /// Steps the environment forward by one move for each game, selecting actions deterministically.
    ///
    /// This method is intended for **inference or evaluation**. It always chooses the
    /// action with the highest probability from the legal moves for each game.
    ///
    /// # Arguments
    /// * `actions`: A `numpy.ndarray` of shape `(batch_size, 8, 8)` and `dtype=float32`.
    ///
    /// # Returns
    /// A tuple `(next_states, dones)`.
    #[pyo3(name = "step_batch_deterministic")]
    fn step_batch_deterministic(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray3<f32>,
    ) -> PyResult<StepBatchResult> {
        // Define the action selection strategy: always pick the best move (greedy).
        let selector = |puttable_positions: PuttablePositions,
                        game_actions: &ndarray::ArrayView2<'_, f32>| {
            // Find the position with the maximum probability among all legal moves.
            // The `unwrap()` calls are safe because this closure is only called when `puttable_positions` is not empty.
            let position = *puttable_positions
                .to_vec()
                .iter()
                .max_by(|&a, &b| {
                    let prob_a = game_actions[[a.row() as usize, a.column() as usize]];
                    let prob_b = game_actions[[b.row() as usize, b.column() as usize]];
                    prob_a.partial_cmp(&prob_b).unwrap()
                })
                .unwrap();
            BitPosition::from(position)
        };

        // Delegate the core logic to the private batch processing function.
        self._step_batch_core(py, actions, selector)
    }
}

#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<ReversiEnvironment>()?;
    Ok(())
}
