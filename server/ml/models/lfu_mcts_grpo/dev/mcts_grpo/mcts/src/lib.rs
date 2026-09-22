mod game;
mod inference;
mod python_bindings;
mod search;
mod tree;

use pyo3::prelude::*;
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python_bindings::Mcts>()?;
    Ok(())
}
