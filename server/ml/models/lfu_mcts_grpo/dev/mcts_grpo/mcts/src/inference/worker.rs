use crate::policy::{Policy, PolicyEvaluation, Value};
use crate::state::State;
use common::{BitPosition, Column, DiskColor, Position, Row};
use num_traits::FromPrimitive;
use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// 推論リクエストのデータ構造
#[derive(Debug)]
pub struct InferenceRequest {
    states: Vec<State>,
    response_tx: oneshot::Sender<Vec<PolicyEvaluation>>,
}

impl InferenceRequest {
    pub fn new(states: Vec<State>, response_tx: oneshot::Sender<Vec<PolicyEvaluation>>) -> Self {
        Self {
            states,
            response_tx,
        }
    }
}

/// 推論ワーカーの状態管理と実行を担当する構造体
pub struct Worker {
    /// バッチサイズ
    batch_size: usize,
    /// リクエスト受信用の受信チャンネル
    rx: Option<mpsc::Receiver<InferenceRequest>>,
    /// ワーカーハンドル
    handle: Option<tokio::task::JoinHandle<()>>,
    /// Python callback for inference
    callback: Py<PyAny>,
}

impl Worker {
    /// 新しいWorkerを生成する
    pub fn new(batch_size: usize, callback: Py<PyAny>) -> (Self, mpsc::Sender<InferenceRequest>) {
        let (tx, rx) = mpsc::channel(32);

        (
            Self {
                batch_size,
                rx: Some(rx),
                handle: None,
                callback,
            },
            tx,
        )
    }

    /// ワーカーを起動する
    pub fn start(&mut self) {
        let rx = self.rx.take().expect("Worker already started");
        let batch_size = self.batch_size;
        let callback = Python::attach(|py| self.callback.clone_ref(py));

        self.handle = Some(tokio::spawn(async move {
            let mut batch_requests = Vec::with_capacity(batch_size);
            let mut rx = rx; // Move rx into the task

            loop {
                Self::collect_requests_until_batch_size(&mut rx, &mut batch_requests, batch_size)
                    .await;

                if !batch_requests.is_empty() {
                    Self::process_batch(&mut batch_requests, &callback).await;
                } else if rx.is_closed() {
                    break; // すべての送信者がドロップされた場合
                }
            }
        }));
    }

    /// バッチサイズに達するまでリクエストを収集する
    async fn collect_requests_until_batch_size(
        rx: &mut mpsc::Receiver<InferenceRequest>,
        batch_requests: &mut Vec<InferenceRequest>,
        batch_size: usize,
    ) {
        // 1. まず1つ目のリクエストを待つ (ブロッキング)
        if batch_requests.is_empty() {
            match rx.recv().await {
                Some(req) => batch_requests.push(req),
                None => return, // 送信者がすべてドロップされた
            }
        }

        // 2. 残りのリクエストを収集 (タイムアウト付き)
        // 既にチャンネルにあるリクエストは即座に取得されるため、
        // 明示的なノンブロッキング収集は不要。
        //
        // MCTSの探索では、複数のスレッドが並行して推論リクエストを送る。
        // バッチサイズ(32)を埋めることでGPU/CPUの推論効率が向上する。
        // しかし、リクエストが揃わない場合に待ちすぎるとレイテンシが悪化し、
        // 全体の探索速度(N/sec)が低下する。
        //
        // ベンチマーク結果 (16並行MCTS, 100シミュレーション/ツリー):
        // - Tree::searchのinternal_batch_size=8により、バッチサイズは主に8前後
        // - 100μsのタイムアウトで、リクエストが連続する場合は即座に集約され、
        //   端数やリクエストが途切れた場合は待たずに処理開始
        // - スループット: ~30,000 req/sec, レイテンシ: ~3ms/state
        //
        // Pythonの関数呼び出しやデータ変換のオーバーヘッド(数ms程度)と比較して、
        // 100マイクロ秒(0.1ms)は無視できる程度だが、バッチ充填のチャンスを
        // 確保できる時間として適切。
        let timeout = std::time::Duration::from_micros(100);
        let deadline = tokio::time::Instant::now() + timeout;

        while batch_requests.len() < batch_size {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(req)) => batch_requests.push(req),
                Ok(None) => return, // チャンネル切断
                Err(_) => break,    // タイムアウト
            }
        }
    }

    /// バッチ処理を実行する
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

    /// 推論を実行する
    fn run_inference(states: &[State], callback: &Py<PyAny>) -> Vec<PolicyEvaluation> {
        Python::attach(|py| {
            let batch_size = states.len();
            if batch_size == 0 {
                return Vec::new();
            }

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
            let result = callback
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

    /// ワーカーを停止する
    pub async fn stop(self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(handle) = self.handle {
            handle.await?;
        }
        Ok(())
    }
}
