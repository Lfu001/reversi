mod dirichlet;
mod inference;
mod node;
mod policy;
mod python_bindings;
mod selection;
mod state;
mod transposition_table;
mod tree;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python_bindings::MCTS>()?;
    Ok(())
}
