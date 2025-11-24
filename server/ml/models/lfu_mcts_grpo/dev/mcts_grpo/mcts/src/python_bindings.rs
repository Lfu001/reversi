use crate::{
    inference::ModelEvaluator,
    inference::worker::{InferenceRequest, Worker},
    node::Node,
    policy::PolicyEvaluation,
    selection::PuctConfig,
    state::State,
    transposition_table::TranspositionTable,
    tree::Tree,
};
use common::DiskColor;
use numpy::PyReadonlyArray4;
use pyo3::prelude::*;
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
}

#[pymethods]
impl MCTS {
    #[new]
    fn new() -> Self {
        MCTS {
            transposition_table: TranspositionTable::new(),
        }
    }

    fn run(
        &self,
        py: Python<'_>,
        states: PyReadonlyArray4<f32>,
        callback: Py<PyAny>,
        num_simulations: usize,
        dirichlet_epsilon: f64,
        dirichlet_alpha: f64,
        c_puct: f64,
    ) -> PyResult<Vec<Option<usize>>> {
        let states_array = states.as_array();
        let batch_size = states_array.shape()[0];

        // Convert Tensor to States
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

        let puct_config = PuctConfig {
            c_puct,
            dirichlet_epsilon,
            dirichlet_alpha,
        };

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        let worker_batch_size = 32;
        let (mut worker, sender) = Worker::new(worker_batch_size, callback);

        runtime.block_on(async {
            worker.start();
        });

        let tt = &self.transposition_table;

        let results: Vec<Option<usize>> = py.detach(|| {
            rust_states
                .par_iter()
                .map(|state| {
                    let py_model = PythonModel {
                        sender: sender.clone(),
                    };

                    let root = Node::new(*state, None);
                    let tree = Tree::new(root);

                    let internal_batch_size = 8;
                    let num_batches =
                        (num_simulations + internal_batch_size - 1) / internal_batch_size;

                    tree.search(num_batches, internal_batch_size, &py_model, tt, puct_config)
                })
                .collect()
        });

        Ok(results)
    }
}
