use common::{Action, Column, DiskColor, Position, PutConfig, Row, Table};
use ndarray::Array;
use num_traits::FromPrimitive;
use numpy::{prelude::*, PyArray, PyArray1, PyArray4, PyReadonlyArray3};
use pyo3::prelude::*;
use reversi_core::{action::get_puttable_positions, controller::Controller, state::BoardExt};

/// A PyClass that represents a batch of Reversi games.
#[pyclass]
struct ReversiEnvironment {
    /// A vector of Tables, each representing a game of Reversi.
    tables: Vec<Table>,
    /// A vector of booleans, each indicating whether a game has ended or not.
    dones: Vec<bool>,
    /// The number of games in the batch.
    batch_size: usize,
}

/// A tuple containing the results of a batch step of the environment.
///
/// The tuple contains two elements:
/// - The next state of the environment as a `Py<PyArray4<f32>>`.
/// - A boolean array indicating whether each game in the batch has finished or not as a `Py<PyArray1<bool>>`.
type StepBatchResult = (Py<PyArray4<f32>>, Py<PyArray1<bool>>);

#[pymethods]
impl ReversiEnvironment {
    /// Creates a new [`ReversiEnvironment`] with the given `batch_size`.
    ///
    /// The environment is initialized with a vector of `Table`s, each
    /// representing a game of Reversi. A vector of booleans is also
    /// initialized to keep track of whether each game has ended or not.
    #[new]
    fn new(batch_size: usize) -> Self {
        ReversiEnvironment {
            tables: (0..batch_size).map(|_| Table::default()).collect(),
            dones: vec![false; batch_size],
            batch_size,
        }
    }

    /// Returns the number of games in the batch.
    #[getter]
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Resets the environment to its initial state.
    ///
    /// This function resets the `tables` field to a vector of default [`Table`]s,
    /// and resets the `dones` field to a vector of `false` values.
    ///
    /// The function then returns the current state of the environment as a
    /// `Py<PyArray4<f32>>`, which indicates [batch_size, channels, 8, 8] where
    /// channels are:
    /// - 0: Dark disks
    /// - 1: Light disks
    /// - 2: Turn
    /// - 3: Legal moves
    fn reset(&mut self, py: Python<'_>) -> PyResult<Py<PyArray4<f32>>> {
        self.tables = (0..self.batch_size).map(|_| Table::default()).collect();
        self.dones = vec![false; self.batch_size];
        self.get_state(py)
    }

    /// Returns the current state of the environment.
    ///
    /// Returns a `Py<PyArray4<f32>>`, which indicates \[batch_size, channels, 8, 8\]
    /// where channels are:
    /// - 0: Dark disks
    /// - 1: Light disks
    /// - 2: Turn
    /// - 3: Legal moves
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyArray4<f32>>> {
        let mut state_vec = Vec::with_capacity(self.batch_size * 4 * 8 * 8);

        for table in &self.tables {
            let turn = table.turn();
            let board = table.board();
            let puttable_positions = get_puttable_positions(board, turn);

            let mut c0_dark_disks = [0.0f32; 64];
            let mut c1_light_disks = [0.0f32; 64];
            let c2_turn = match turn {
                DiskColor::Dark => [1.0f32; 64],
                DiskColor::Light => [0.0f32; 64],
            };
            let mut c3_legal_moves = [0.0f32; 64];

            for r in 0..8 {
                for c in 0..8 {
                    let pos = Position::new(Row::from_u8(r).unwrap(), Column::from_u8(c).unwrap());
                    let idx = (r * 8 + c) as usize;
                    if let Some(disk) = board.get_disk(&pos) {
                        if disk == turn {
                            c0_dark_disks[idx] = 1.0;
                        } else {
                            c1_light_disks[idx] = 1.0;
                        }
                    }
                }
            }

            for pos in puttable_positions {
                let idx = (pos.row() as u8 * 8 + pos.column() as u8) as usize;
                c3_legal_moves[idx] = 1.0;
            }

            state_vec.extend_from_slice(&c0_dark_disks);
            state_vec.extend_from_slice(&c1_light_disks);
            state_vec.extend_from_slice(&c2_turn);
            state_vec.extend_from_slice(&c3_legal_moves);
        }

        let array = PyArray::from_vec(py, state_vec);
        Ok(array.reshape((self.batch_size, 4, 8, 8))?.into())
    }

    /// Steps the environment forward by one step.
    ///
    /// This function takes in a 3D array of `actions` of shape
    /// (batch_size, 8, 8) where each element is a float between 0 and
    /// 1 representing the probability of taking a certain action.
    /// Each element of the array is accessed as `actions[i, r, c]` where
    /// `i` is the batch index, `r` is the row index and `c` is the column
    /// index.
    fn step_batch(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray3<f32>,
    ) -> PyResult<StepBatchResult> {
        let actions_array = actions.as_array();
        let mut current_dones = vec![false; self.batch_size];

        for i in 0..self.batch_size {
            if self.dones[i] {
                current_dones[i] = true;
                continue;
            }

            let table = &mut self.tables[i];
            let turn = table.turn();
            let puttable_positions = get_puttable_positions(table.board(), turn);

            let action = if puttable_positions.is_empty() {
                Action::PassTurn(turn)
            } else {
                let mut best_pos = puttable_positions[0];
                let mut max_prob = -1.0;

                for pos in &puttable_positions {
                    let r = pos.row() as usize;
                    let c = pos.column() as usize;
                    let prob = actions_array[[i, r, c]];
                    if prob > max_prob {
                        max_prob = prob;
                        best_pos = *pos;
                    }
                }
                Action::PutDisk(PutConfig::new(turn, best_pos))
            };

            match Controller::step(table, action) {
                Ok(step_result) => {
                    if step_result.judge_result.is_some() {
                        self.dones[i] = true;
                        current_dones[i] = true;
                    }
                }
                Err(_) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Invalid action executed internally.",
                    ));
                }
            }
        }

        let next_states = self.get_state(py)?;
        let dones_array = Array::from_vec(current_dones).into_pyarray(py);

        Ok((next_states, dones_array.into()))
    }
}

#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<ReversiEnvironment>()?;
    Ok(())
}
