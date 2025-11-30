use crate::policy::{Policy, PolicyEvaluation, Value};
use crate::state::State;
use common::{BitPosition, Column, DiskColor, Position, Row};
use num_traits::FromPrimitive;
use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// Data structure for an inference request.
#[derive(Debug)]
pub struct InferenceRequest {
    /// Game states to be evaluated by the neural network.
    states: Vec<State>,
    /// Channel to send back the evaluation results.
    response_tx: oneshot::Sender<Vec<PolicyEvaluation>>,
}

impl InferenceRequest {
    /// Creates a new inference request with the given `states` and response channel.
    ///
    /// The `states` will be batched for neural network evaluation, and results
    /// will be sent back through `response_tx`.
    pub fn new(states: Vec<State>, response_tx: oneshot::Sender<Vec<PolicyEvaluation>>) -> Self {
        Self {
            states,
            response_tx,
        }
    }
}

/// Manages and executes inference worker tasks.
///
/// The worker batches inference requests to improve GPU/CPU utilization efficiency.
pub struct Worker {
    /// Maximum number of inference requests the worker can process at once.
    /// Batching up to this size improves GPU throughput.
    max_inference_batch_size: usize,
    /// Receiving channel for incoming requests.
    /// Set to `None` after the worker starts to prevent restarting.
    rx: Option<mpsc::Receiver<InferenceRequest>>,
    /// Handle to the asynchronous worker task.
    /// Used to await worker completion when stopping.
    handle: Option<tokio::task::JoinHandle<()>>,
    /// Python callback function that performs neural network inference.
    /// Called with batched game states.
    callback: Py<PyAny>,
}

impl Worker {
    /// Creates a new Worker instance with the specified batch size and callback.
    ///
    /// Takes `max_inference_batch_size` to configure batching behavior and a Python
    /// `callback` function for neural network inference. Returns a tuple of the
    /// [`Worker`] and a sender channel for submitting inference requests.
    pub fn new(
        max_inference_batch_size: usize,
        callback: Py<PyAny>,
    ) -> (Self, mpsc::Sender<InferenceRequest>) {
        let (tx, rx) = mpsc::channel(32);

        (
            Self {
                max_inference_batch_size,
                rx: Some(rx),
                handle: None,
                callback,
            },
            tx,
        )
    }

    /// Starts the worker task.
    pub fn start(&mut self) {
        let rx = self.rx.take().expect("Worker already started");
        let max_inference_batch_size = self.max_inference_batch_size;
        let callback = Python::attach(|py| self.callback.clone_ref(py));

        self.handle = Some(tokio::spawn(async move {
            let mut batch_requests = Vec::with_capacity(max_inference_batch_size);
            let mut rx = rx; // Move rx into the task

            loop {
                Self::collect_requests_until_batch_size(
                    &mut rx,
                    &mut batch_requests,
                    max_inference_batch_size,
                )
                .await;

                if !batch_requests.is_empty() {
                    Self::process_batch(&mut batch_requests, &callback).await;
                } else if rx.is_closed() {
                    break; // All senders have been dropped
                }
            }
        }));
    }

    /// Collects inference requests until the batch size is reached.
    ///
    /// Waits for the first request, then collects additional requests within a timeout
    /// window to fill the batch. The `rx` channel receives requests, `batch_requests`
    /// accumulates them, and `max_inference_batch_size` sets the target batch size.
    async fn collect_requests_until_batch_size(
        rx: &mut mpsc::Receiver<InferenceRequest>,
        batch_requests: &mut Vec<InferenceRequest>,
        max_inference_batch_size: usize,
    ) {
        // 1. First, wait for the initial request (blocking)
        if batch_requests.is_empty() {
            match rx.recv().await {
                Some(req) => batch_requests.push(req),
                None => return, // All senders have been dropped
            }
        }

        // 2. Collect remaining requests with timeout
        // Requests already in the channel are immediately retrieved,
        // so no explicit non-blocking collection is needed.
        //
        // In MCTS search, multiple threads send inference requests concurrently.
        // Filling max_inference_batch_size(32) improves GPU/CPU inference efficiency.
        // However, waiting too long when requests don't arrive degrades latency,
        // reducing overall search speed (N/sec).
        //
        // Benchmark results (16 concurrent MCTS, 100 simulations/tree):
        // - Tree::search's states_per_inference=8 results in requests of ~8 states
        // - 100μs timeout allows immediate aggregation when requests are continuous,
        //   and starts processing without waiting when requests taper off
        // - Throughput: ~30,000 req/sec, Latency: ~3ms/state
        //
        // Compared to Python function call and data conversion overhead (several ms),
        // 100 microseconds (0.1ms) is negligible, but provides sufficient opportunity
        // to fill the batch.
        let timeout = std::time::Duration::from_micros(100);
        let deadline = tokio::time::Instant::now() + timeout;

        while batch_requests.len() < max_inference_batch_size {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(req)) => batch_requests.push(req),
                Ok(None) => return, // Channel disconnected
                Err(_) => break,    // Timeout
            }
        }
    }

    /// Processes a batch of inference requests by running inference and distributing results.
    ///
    /// Flattens states from all `batch_requests`, calls the Python `callback` for inference,
    /// then distributes results back to each request's response channel.
    async fn process_batch(batch_requests: &mut Vec<InferenceRequest>, callback: &Py<PyAny>) {
        // Flatten all states from all requests
        let mut all_states = Vec::new();
        let mut request_sizes = Vec::with_capacity(batch_requests.len());

        for req in batch_requests.iter() {
            request_sizes.push(req.states.len());
            all_states.extend_from_slice(&req.states);
        }

        // Run inference on the combined batch
        let all_results = Self::run_inference(&all_states, callback);

        // Distribute results back to requests
        let mut start_idx = 0;
        for (req, size) in batch_requests.drain(..).zip(request_sizes) {
            let end_idx = start_idx + size;
            let req_results = all_results[start_idx..end_idx].to_vec();
            if let Err(_) = req.response_tx.send(req_results) {
                eprintln!("⚠️ Warning: Client dropped the response channel.");
            }
            start_idx = end_idx;
        }
    }

    /// Runs neural network inference on a batch of states using the Python callback.
    ///
    /// Converts the `states` into a 4-channel input tensor, invokes the Python `callback`,
    /// and parses the results into [`PolicyEvaluation`] objects.
    fn run_inference(states: &[State], callback: &Py<PyAny>) -> Vec<PolicyEvaluation> {
        Python::attach(|py| {
            let batch_size = states.len();
            if batch_size == 0 {
                return Vec::new();
            }

            let input_data = Self::build_input_tensor(states);
            let input_array = PyArray::from_vec(py, input_data)
                .reshape((batch_size, 4, 8, 8))
                .expect("Failed to reshape input array");

            let result = callback
                .call1(py, (input_array,))
                .expect("Failed to call callback");

            Self::parse_inference_results(py, result, batch_size)
        })
    }

    /// Builds a 4-channel input tensor from game states.
    ///
    /// Constructs a flat vector representing a \[B, 4, 8, 8\] tensor where:
    /// - Channel 0: Dark disk positions
    /// - Channel 1: Light disk positions
    /// - Channel 2: Current turn (1.0 for Dark, -1.0 for Light)
    /// - Channel 3: Legal move positions
    fn build_input_tensor(states: &[State]) -> Vec<f64> {
        let batch_size = states.len();
        let mut input_data = Vec::with_capacity(batch_size * 4 * 8 * 8);

        for state in states {
            let board = state.board();
            let turn = state.turn();
            let legal_actions = state.legal_actions();

            // Channel 0: Dark disks
            for r in 0..8 {
                for c in 0..8 {
                    let pos = Position::new(Row::from_u8(r).unwrap(), Column::from_u8(c).unwrap());
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
                    let pos = Position::new(Row::from_u8(r).unwrap(), Column::from_u8(c).unwrap());
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

        input_data
    }

    /// Parses Python inference results into Rust PolicyEvaluation objects.
    ///
    /// Extracts policy and value arrays from the Python `result` tuple and converts
    /// them into a vector of [`PolicyEvaluation`] objects, one per state in the batch.
    fn parse_inference_results(
        py: Python<'_>,
        result: Py<PyAny>,
        batch_size: usize,
    ) -> Vec<PolicyEvaluation> {
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
    }

    /// Stops the worker and waits for it to complete.
    ///
    /// Awaits the worker task handle, ensuring clean shutdown.
    pub async fn stop(self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(handle) = self.handle {
            handle.await?;
        }
        Ok(())
    }
}
