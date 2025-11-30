use crate::{
    inference::ModelEvaluator,
    inference::worker::{InferenceRequest, Worker},
    node::Node,
    policy::PolicyEvaluation,
    selection::PuctConfig,
    state::State,
    transposition_table::TranspositionTable,
    tree::{SearchResults, Tree},
};
use common::DiskColor;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray4, ndarray::Array2};
use pyo3::prelude::*;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use tokio::sync::{mpsc, oneshot};

struct PythonModel {
    sender: mpsc::Sender<InferenceRequest>,
}

impl ModelEvaluator for PythonModel {
    fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation> {
        let (response_tx, response_rx) = oneshot::channel();
        let request = InferenceRequest::new(states.to_vec(), response_tx);

        if let Err(_) = self.sender.blocking_send(request) {
            panic!("Failed to send inference request to worker");
        }

        match response_rx.blocking_recv() {
            Ok(result) => result,
            Err(_) => panic!("Failed to receive inference response from worker"),
        }
    }
}

#[pyclass(unsendable)]
pub struct MCTS {
    transposition_table: TranspositionTable,
    max_inference_batch_size: usize,
    states_per_inference: usize,
}

#[pymethods]
impl MCTS {
    /// Creates a new MCTS instance.
    ///
    /// # Arguments
    ///
    /// * `max_inference_batch_size` - The maximum batch size for inference requests.
    /// * `states_per_inference` - The number of states to process per inference call.
    #[new]
    fn new(max_inference_batch_size: usize, states_per_inference: usize) -> Self {
        MCTS {
            transposition_table: TranspositionTable::new(),
            max_inference_batch_size,
            states_per_inference,
        }
    }

    /// Runs MCTS simulations for a batch of states.
    ///
    /// # Arguments
    ///
    /// * `py` - The Python GIL token.
    /// * `states` - A 4D numpy array of shape (batch_size, 2, 8, 8) representing the game states.
    /// * `callback` - A Python callable for inference.
    /// * `num_simulations` - Number of MCTS simulations per state.
    /// * `dirichlet_epsilon` - Dirichlet noise epsilon.
    /// * `dirichlet_alpha` - Dirichlet noise alpha.
    /// * `c_puct` - PUCT constant.
    /// * `seed` - Optional random seed.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `pi` - Policy distribution of shape (batch_size, 64).
    /// * `q_values` - Q-values of shape (batch_size, 64).
    fn run(
        &self,
        py: Python<'_>,
        states: PyReadonlyArray4<f32>,
        callback: Py<PyAny>,
        num_simulations: usize,
        dirichlet_epsilon: f64,
        dirichlet_alpha: f64,
        c_puct: f64,
        seed: Option<u64>,
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
        let states_array = states.as_array();
        let batch_size = states_array.shape()[0];

        let rust_states = Self::convert_to_rust_states(states_array);

        let puct_config = PuctConfig {
            c_puct,
            dirichlet_epsilon,
            dirichlet_alpha,
        };

        let results = self.run_mcts_batch(
            py,
            rust_states,
            callback,
            num_simulations,
            puct_config,
            seed,
        );

        Self::process_results(py, results, batch_size)
    }
}

impl MCTS {
    /// Converts a Python numpy array of states to a vector of Rust [`State`] objects.
    ///
    /// # Arguments
    ///
    /// * `states_array` - A 4D numpy array view of shape (batch_size, 2, 8, 8).
    ///
    /// # Returns
    ///
    /// A vector of [`State`] objects.
    fn convert_to_rust_states(states_array: numpy::ndarray::ArrayView4<f32>) -> Vec<State> {
        let batch_size = states_array.shape()[0];
        let mut rust_states = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            // Check turn from Channel 2 (index 2)
            // state[i, 2, 0, 0]
            let turn_val = states_array[[i, 2, 0, 0]];
            let turn = if turn_val > 0.0 {
                DiskColor::Dark
            } else {
                DiskColor::Light
            };

            let mut dark_plane = 0u64;
            let mut light_plane = 0u64;

            for r in 0..8 {
                for c in 0..8 {
                    let dark = states_array[[i, 0, r, c]];
                    let light = states_array[[i, 1, r, c]];
                    let bit = 1u64 << (63 - (r * 8 + c));

                    if dark > 0.5 {
                        dark_plane |= bit;
                    }
                    if light > 0.5 {
                        light_plane |= bit;
                    }
                }
            }

            let board = common::Bitboard::new(dark_plane, light_plane);
            let state = State::new(board, turn);
            rust_states.push(state);
        }
        rust_states
    }

    /// Runs MCTS simulations for a batch of states in parallel.
    ///
    /// Returns a vector of SearchResults, one for each input state.
    fn run_mcts_batch(
        &self,
        py: Python<'_>,
        rust_states: Vec<State>,
        callback: Py<PyAny>,
        num_simulations: usize,
        puct_config: PuctConfig,
        seed: Option<u64>,
    ) -> Vec<SearchResults> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        let (mut worker, sender) = Worker::new(self.max_inference_batch_size, callback);

        runtime.block_on(async {
            worker.start();
        });

        let tt = &self.transposition_table;

        py.detach(|| {
            rust_states
                .par_iter()
                .enumerate()
                .map(|(batch_idx, state)| {
                    // Create RNG for this tree
                    // If seed is provided, use seed + batch_idx for diversity
                    // Otherwise, use rand::rng() (ThreadRng) directly
                    let mut rng: Box<dyn RngCore> = if let Some(base_seed) = seed {
                        Box::new(StdRng::seed_from_u64(
                            base_seed.wrapping_add(batch_idx as u64),
                        ))
                    } else {
                        Box::new(rand::rng())
                    };

                    let py_model = PythonModel {
                        sender: sender.clone(),
                    };

                    let root = Node::new(*state, None);
                    let tree = Tree::new(root);

                    let num_inferences = (num_simulations + self.states_per_inference - 1)
                        / self.states_per_inference;

                    tree.search(
                        num_inferences,
                        self.states_per_inference,
                        &py_model,
                        tt,
                        puct_config,
                        &mut rng,
                    )
                })
                .collect()
        })
    }

    /// Processes the MCTS results into numpy arrays.
    ///
    /// Converts search results into policy distributions (pi) and Q-values as numpy arrays.
    fn process_results(
        py: Python<'_>,
        results: Vec<SearchResults>,
        batch_size: usize,
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
        // Convert results into numpy arrays and normalize visit counts to pi
        // pi: [batch_size, 64], q_values: [batch_size, 64]
        let mut pi = vec![0.0f64; batch_size * 64];
        let mut q_values = vec![0.0f64; batch_size * 64];

        for (i, result) in results.iter().enumerate() {
            // Calculate total visits for normalization
            let total_visits: u32 = result.visit_counts.iter().sum();

            for j in 0..64 {
                // Normalize visit counts to get policy distribution
                if total_visits > 0 {
                    pi[i * 64 + j] = result.visit_counts[j] as f64 / total_visits as f64;
                } else {
                    // If no visits, uniform distribution
                    pi[i * 64 + j] = 1.0 / 64.0;
                }
                q_values[i * 64 + j] = result.q_values[j];
            }
        }

        // Convert to numpy arrays
        let pi_array = Array2::from_shape_vec((batch_size, 64), pi)
            .expect("Failed to create pi array")
            .into_pyarray(py)
            .unbind();

        let q_values_array = Array2::from_shape_vec((batch_size, 64), q_values)
            .expect("Failed to create q values array")
            .into_pyarray(py)
            .unbind();

        Ok((pi_array, q_values_array))
    }
}
