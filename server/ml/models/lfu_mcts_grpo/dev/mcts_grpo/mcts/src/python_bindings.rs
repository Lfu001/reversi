use crate::{
    inference::ModelEvaluator,
    node::Node,
    policy::{Policy, PolicyEvaluation, Value},
    selection::PuctConfig,
    state::State,
    transposition_table::TranspositionTable,
    tree::Tree,
};
use common::BitPosition;
use common::{Column, DiskColor, Position, Row};
use num_traits::FromPrimitive;
use numpy::{PyArray, PyArrayMethods, PyReadonlyArray4};
use pyo3::prelude::*;
use std::cell::RefCell;

struct PythonModel {
    callback: Py<PyAny>,
}

impl ModelEvaluator for PythonModel {
    fn infer(&self, states: &[State]) -> Vec<PolicyEvaluation> {
        Python::attach(|py| {
            let batch_size = states.len();
            // Create input array [B, 4, 8, 8]
            let mut input_data = Vec::with_capacity(batch_size * 4 * 8 * 8);

            for state in states {
                let board = state.board();
                let turn = state.turn();
                let legal_actions = state.legal_actions(); // indices 0-63, 64 for pass

                // Channel 0: Dark disks
                for r in 0..8 {
                    for c in 0..8 {
                        let pos =
                            Position::new(Row::from_u8(r).unwrap(), Column::from_u8(c).unwrap());
                        let bit_pos = BitPosition::from(pos).0;
                        let val = if (board.dark_plane() & bit_pos) != 0 {
                            1.0
                        } else {
                            0.0
                        };
                        input_data.push(val);
                    }
                }

                // Channel 1: Light disks
                for r in 0..8 {
                    for c in 0..8 {
                        let pos =
                            Position::new(Row::from_u8(r).unwrap(), Column::from_u8(c).unwrap());
                        let bit_pos = BitPosition::from(pos).0;
                        let val = if (board.light_plane() & bit_pos) != 0 {
                            1.0
                        } else {
                            0.0
                        };
                        input_data.push(val);
                    }
                }

                // Channel 2: Turn (1.0 for Dark, -1.0 for Light)
                let turn_val = if turn == DiskColor::Dark { 1.0 } else { -1.0 };
                for _ in 0..64 {
                    input_data.push(turn_val);
                }

                // Channel 3: Legal moves
                let mut legal_map = [0.0; 64];
                for &action in &legal_actions {
                    if action < 64 {
                        legal_map[action] = 1.0;
                    }
                }
                input_data.extend_from_slice(&legal_map);
            }

            let input_array = PyArray::from_vec(py, input_data)
                .reshape((batch_size, 4, 8, 8))
                .expect("Failed to reshape input array");

            // Call callback(input_array)
            let result = self
                .callback
                .call1(py, (input_array,))
                .expect("Failed to call callback");

            // Extract result: (policy_batch, value_batch)
            let result_tuple: (Py<PyAny>, Py<PyAny>) =
                result.extract(py).expect("Failed to extract tuple");
            let policy_obj = result_tuple.0;
            let value_obj = result_tuple.1;

            let policy_array: &Bound<'_, PyArray<f64, numpy::Ix2>> = policy_obj
                .downcast_bound(py)
                .expect("Failed to downcast policy array");
            let value_array: &Bound<'_, PyArray<f64, numpy::Ix2>> = value_obj
                .downcast_bound(py)
                .expect("Failed to downcast value array");

            let policy_data = unsafe { policy_array.as_array() };
            let value_data = unsafe { value_array.as_array() };

            let mut evaluations = Vec::with_capacity(batch_size);

            for i in 0..batch_size {
                let mut policy_arr = [0.0; 64];
                for j in 0..64 {
                    policy_arr[j] = policy_data[[i, j]];
                }
                let value = value_data[[i, 0]];

                evaluations.push(PolicyEvaluation::new(Policy(policy_arr), Value(value)));
            }

            evaluations
        })
    }
}

#[pyclass(unsendable)]
pub struct MCTS {
    transposition_table: RefCell<TranspositionTable>,
}

#[pymethods]
impl MCTS {
    #[new]
    fn new() -> Self {
        MCTS {
            transposition_table: RefCell::new(TranspositionTable::new()),
        }
    }

    fn run(
        &self,
        _py: Python<'_>,
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

        let py_model = PythonModel { callback };
        let puct_config = PuctConfig {
            c_puct,
            dirichlet_epsilon,
            dirichlet_alpha,
        };

        // Run MCTS for each state
        // Note: This is sequential execution.
        // To optimize, we should parallelize this or batch the inference across trees.
        // For now, we use the existing Tree::search which batches internally for a single tree.

        let mut results = Vec::with_capacity(batch_size);
        let mut tt = self.transposition_table.borrow_mut();

        for state in rust_states {
            let root = Node::new(state, None);
            let tree = Tree::new(root);

            // Assuming batch_size for Tree::search is fixed or derived?
            // User passed num_simulations.
            // We need to decide num_batches and batch_size for search.
            // Let's assume batch_size = 8 (common for MCTS batching)
            // num_batches = num_simulations / 8
            let internal_batch_size = 8;
            let num_batches = (num_simulations + internal_batch_size - 1) / internal_batch_size;

            let action = tree.search(
                num_batches,
                internal_batch_size,
                &py_model,
                &mut *tt,
                puct_config,
            );
            results.push(action);
        }

        Ok(results)
    }
}
