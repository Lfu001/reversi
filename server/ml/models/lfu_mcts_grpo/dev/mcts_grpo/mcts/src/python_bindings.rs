use crate::{
    inference::ModelEvaluator,
    inference::worker::{InferenceRequest, Worker},
    policy::PolicyEvaluation,
    search::{SearchParams, SearchResults, get_move_second},
    selection::PuctConfig,
    state::State,
    transposition_table::TranspositionTable,
    tree::Tree,
};
use common::DiskColor;
use indicatif::ProgressBar;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray4, ndarray::Array2};
use pyo3::prelude::*;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, oneshot};

/// Type alias for MCTS run results: (policy distribution, Q-values)
type MctsResult = PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)>;

struct PythonModel {
    sender: mpsc::Sender<InferenceRequest>,
}

impl ModelEvaluator for PythonModel {
    fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation> {
        let (response_tx, response_rx) = oneshot::channel();
        let request = InferenceRequest::new(states.to_vec(), response_tx);

        if self.sender.blocking_send(request).is_err() {
            panic!("Failed to send inference request to worker");
        }

        match response_rx.blocking_recv() {
            Ok(result) => result,
            Err(_) => panic!("Failed to receive inference response from worker"),
        }
    }
}

#[pyclass(unsendable)]
pub struct Mcts {
    transposition_table: TranspositionTable,
    max_inference_batch_size: usize,
    states_per_inference: usize,
    runtime: tokio::runtime::Runtime,
    thread_pool: rayon::ThreadPool,
    simulation_count: AtomicUsize,
}

#[pymethods]
impl Mcts {
    /// Creates a new MCTS instance.
    ///
    /// # Arguments
    ///
    /// * `max_inference_batch_size` - The maximum batch size for inference requests.
    /// * `states_per_inference` - The number of states to process per inference call.
    #[new]
    fn new(max_inference_batch_size: usize, states_per_inference: usize) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .build()
            .expect("Failed to create Rayon thread pool");

        Mcts {
            transposition_table: TranspositionTable::new(),
            max_inference_batch_size,
            states_per_inference,
            runtime,
            thread_pool,
            simulation_count: AtomicUsize::new(0),
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
    #[allow(clippy::too_many_arguments)]
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
    ) -> MctsResult {
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

impl Mcts {
    /// Converts a Python numpy array of states to a vector of Rust [`State`] objects.
    fn convert_to_rust_states(states_array: numpy::ndarray::ArrayView4<f32>) -> Vec<State> {
        let batch_size = states_array.shape()[0];
        let mut rust_states = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
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
    fn run_mcts_batch(
        &self,
        py: Python<'_>,
        rust_states: Vec<State>,
        callback: Py<PyAny>,
        num_simulations: usize,
        puct_config: PuctConfig,
        seed: Option<u64>,
    ) -> Vec<SearchResults> {
        let _guard = self.runtime.enter();

        let (mut worker, sender) = Worker::new(self.max_inference_batch_size, callback);
        worker.start();

        let tt = &self.transposition_table;
        let num_inferences = num_simulations.div_ceil(self.states_per_inference);
        let current_sim = self.simulation_count.fetch_add(1, Ordering::SeqCst);

        let pbar = ProgressBar::new((rust_states.len() * num_inferences) as u64);
        pbar.set_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(5));
        pbar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .expect("Failed to create progress style")
                .progress_chars("#>-"),
        );
        pbar.set_message(format!("MCTS Batch Search (Step {})", current_sim));

        let results = py.detach(|| {
            self.thread_pool.install(|| {
                rust_states
                    .par_iter()
                    .enumerate()
                    .map(|(batch_idx, state)| {
                        let pbar = pbar.clone();
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

                        let mut tree = Tree::new(*state);

                        let params = SearchParams {
                            num_batches: num_inferences,
                            batch_size: self.states_per_inference,
                            config: puct_config,
                            pbar: Some(pbar),
                        };
                        get_move_second(&mut tree, params, &py_model, tt, &mut rng)
                    })
                    .collect()
            })
        });

        pbar.finish_with_message(format!("MCTS Batch Search (Step {}) Finished", current_sim));

        drop(sender);

        self.runtime.block_on(async {
            let _ = worker.stop().await;
        });

        results
    }

    /// Processes the MCTS results into numpy arrays.
    fn process_results(
        py: Python<'_>,
        results: Vec<SearchResults>,
        batch_size: usize,
    ) -> MctsResult {
        let mut pi = vec![0.0f64; batch_size * 64];
        let mut q_values = vec![0.0f64; batch_size * 64];

        for (i, result) in results.iter().enumerate() {
            let total_visits: u32 = result.visit_counts.iter().sum();

            for j in 0..64 {
                if total_visits > 0 {
                    pi[i * 64 + j] = result.visit_counts[j] as f64 / total_visits as f64;
                } else {
                    pi[i * 64 + j] = 1.0 / 64.0;
                }
                q_values[i * 64 + j] = result.q_values[j];
            }
        }

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
