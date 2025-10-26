use common::{
    position, Action, BitPosition, Bitboard, Column, DiskColor, Position, PutConfig, Row, Table,
};
use ndarray::{s, Array};
use num_traits::FromPrimitive;
use numpy::{prelude::*, PyArray, PyArray1, PyArray3, PyArray4, PyReadonlyArray3};
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
    /// A seed value used to determine actions stochastically.
    seed: u64,
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

                // Get the 2D action probabilities for the current game.
                let game_actions = actions_array.slice(s![i, .., ..]);
                // Use the provided strategy (closure) to select the next move.
                let chosen_pos = action_selector(puttable_positions, &game_actions);
                let action = Action::PutDisk(PutConfig::new(turn, chosen_pos.into()));

                // Apply the chosen action to the game board.
                match Controller::step(table, action) {
                    Ok(step_result) => {
                        // Check if the game has ended and update the `done` flag.
                        if step_result.judge_result.is_some() {
                            *done = true;
                        }
                        if step_result.puttable_positions.is_empty() {
                            match Controller::step(table, Action::PassTurn(table.turn())) {
                                Ok(step_result) => {
                                    if step_result.judge_result.is_some() {
                                        *done = true;
                                    }
                                }
                                Err(err) => {
                                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        err,
                                    ))
                                }
                            }
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

    /// Convert the current state of the environment to a 1D vector.
    fn get_state_vec(tables: &[Table]) -> Vec<f32> {
        // Pre-allocate a vector with the exact required capacity for performance.
        let mut state_vec = Vec::with_capacity(tables.len() * 4 * 8 * 8);
        state_vec.resize(tables.len() * 4 * 8 * 8, 0.0);

        for (batch_idx, table) in tables.iter().enumerate() {
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

        state_vec
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
    /// * `seed` - The seed value used to determine actions stochastically.
    #[new]
    fn new(batch_size: usize, seed: u64) -> Self {
        ReversiEnvironment {
            tables: (0..batch_size).map(|_| Table::default()).collect(),
            dones: vec![false; batch_size],
            batch_size,
            seed,
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

    /// Resets games specified by `indices` to their initial state.
    ///
    /// # Arguments
    /// * `indices` - A slice of indices of the games to reset.
    fn reset_indices(
        &mut self,
        py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<Py<PyArray4<f32>>> {
        for index in indices {
            if index >= self.batch_size {
                continue;
            }
            self.tables[index] = Table::default();
            self.dones[index] = false;
        }
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
        let state_vec = ReversiEnvironment::get_state_vec(&self.tables);
        let array = PyArray::from_vec(py, state_vec);
        Ok(array.reshape((self.batch_size, 4, 8, 8))?.into())
    }

    /// Compute the next state from a given state and action.
    ///
    /// This is a single-game version when processing individual games.
    #[staticmethod]
    fn get_next_state(
        py: Python<'_>,
        state: PyReadonlyArray3<f32>,
        action: usize,
    ) -> PyResult<Py<PyArray3<f32>>> {
        let state_array = state.as_array();

        if state_array.shape() != [4, 8, 8] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "State must have shape (4, 8, 8)",
            ));
        }

        // Create a new table with the initial position
        let mut table = Table::default();

        // Convert the state to the internal representation
        table.set_turn(if state_array[[2, 0, 0]] > 0.0 {
            DiskColor::Dark
        } else {
            DiskColor::Light
        });
        let mut dark_plane = 0u64;
        let mut light_plane = 0u64;
        for row in 0..8 {
            for col in 0..8 {
                let dark = state_array[[0, row as usize, col as usize]];
                let light = state_array[[1, row as usize, col as usize]];
                let bit = 1 << (63 - (row * 8 + col));
                if dark > 0.5 {
                    dark_plane |= bit;
                }
                if light > 0.5 {
                    light_plane |= bit;
                }
            }
        }
        table.set_board(Bitboard::new(dark_plane, light_plane));

        // Validate the action is within bounds (0-63)
        if !(0..64).contains(&action) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Action must be between 0 and 63",
            ));
        }

        // Apply the action
        let action = Action::PutDisk(PutConfig::new(
            table.turn(),
            position!(
                Row::from_usize(action / 8).unwrap(),
                Column::from_usize(action % 8).unwrap()
            ),
        ));

        match Controller::step(&mut table, action) {
            Ok(step_result) => {
                if step_result.puttable_positions.is_empty() {
                    let turn = table.turn();
                    match Controller::step(&mut table, Action::PassTurn(turn)) {
                        Ok(_) => {}
                        Err(err) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err))
                        }
                    }
                }
            }
            Err(err) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err)),
        }

        // Convert the resulting state back to a numpy array
        let result = ReversiEnvironment::get_state_vec(&[table]);
        let array = PyArray::from_vec(py, result);
        Ok(array.reshape((4, 8, 8))?.into())
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
        let seed = self.seed;
        self.seed += 1;
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
                    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
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
