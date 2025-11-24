use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// 推論リクエストのデータ構造
#[derive(Debug)]
struct InferenceRequest {
    data: i32,
    response_tx: oneshot::Sender<i32>,
}

/// 推論ワーカーの状態管理と実行を担当する構造体
pub struct Worker {
    /// バッチサイズ
    batch_size: usize,
    /// リクエスト受信用の受信チャンネル
    rx: Option<mpsc::Receiver<InferenceRequest>>,
    /// ワーカーハンドル
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl Worker {
    /// 新しいWorkerを生成する
    pub fn new(batch_size: usize) -> (Self, mpsc::Sender<InferenceRequest>) {
        let (tx, rx) = mpsc::channel(32);

        (
            Self {
                batch_size,
                rx: Some(rx),
                handle: None,
            },
            tx,
        )
    }

    /// ワーカーを起動する
    pub fn start(&mut self) {
        let rx = self.rx.take().expect("Worker already started");
        let batch_size = self.batch_size;

        self.handle = Some(tokio::spawn(async move {
            let mut batch_requests = Vec::with_capacity(batch_size);
            let mut rx = rx; // Move rx into the task

            loop {
                Self::collect_requests_until_batch_size(&mut rx, &mut batch_requests, batch_size)
                    .await;

                if !batch_requests.is_empty() {
                    Self::process_batch(&mut batch_requests).await;
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
        // バッチサイズに達するまでブロッキングで待機
        while batch_requests.len() < batch_size {
            match rx.recv().await {
                Some(req) => batch_requests.push(req),
                None => return, // 送信者がすべてドロップされた
            }
        }

        // ノンブロッキングで追加のリクエストを収集
        Self::collect_additional_requests_non_blocking(rx, batch_requests, batch_size).await;
    }

    /// ノンブロッキングで追加のリクエストを収集する
    async fn collect_additional_requests_non_blocking(
        rx: &mut mpsc::Receiver<InferenceRequest>,
        batch_requests: &mut Vec<InferenceRequest>,
        batch_size: usize,
    ) {
        while batch_requests.len() < batch_size {
            match rx.try_recv() {
                Ok(req) => batch_requests.push(req),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
    }

    /// バッチ処理を実行する
    async fn process_batch(batch_requests: &mut Vec<InferenceRequest>) {
        let batch_size = batch_requests.len();
        Self::log_batch_processing_start(batch_size);

        let results = Self::run_inference(batch_requests).await;
        Self::send_responses(batch_requests.drain(..), results);

        Self::log_batch_processing_complete();
    }

    /// 推論を実行する
    async fn run_inference(batch: &[InferenceRequest]) -> Vec<i32> {
        // 実際の推論処理をシミュレート
        let input_sum: i32 = batch.iter().map(|req| req.data).sum();
        vec![input_sum; batch.len()]
    }

    /// 結果を各クライアントに送信する
    fn send_responses(requests: impl Iterator<Item = InferenceRequest>, results: Vec<i32>) {
        for (req, result) in requests.zip(results) {
            if let Err(_) = req.response_tx.send(result) {
                Self::log_response_dropped();
            }
        }
    }

    // ログ関連のメソッド
    fn log_batch_processing_start(batch_size: usize) {
        println!("\n📦 Worker: バッチ推論処理を開始 (サイズ: {})", batch_size);
    }

    fn log_batch_processing_complete() {
        println!("✅ Worker: バッチ処理完了。");
    }

    fn log_response_dropped() {
        eprintln!("⚠️ Warning: Client dropped the response channel.");
    }

    /// ワーカーを停止する
    pub async fn stop(self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(handle) = self.handle {
            handle.await?;
        }
        Ok(())
    }
}

/// クライアントのサンプル実装
pub async fn client_producer(id: i32, tx: mpsc::Sender<InferenceRequest>) {
    let data = id * 10;
    let request = create_inference_request(id, data);

    if let Err(e) = tx.send(request).await {
        log_send_error(id, e);
        return;
    }

    log_request_sent(id);
}

/// 推論リクエストを作成する
fn create_inference_request(id: i32, data: i32) -> InferenceRequest {
    let (response_tx, response_rx) = oneshot::channel();
    let request = InferenceRequest { data, response_tx };

    // レスポンスの非同期処理を開始
    tokio::spawn(handle_response(id, response_rx));
    request
}

/// レスポンスを処理する
async fn handle_response(id: i32, response_rx: oneshot::Receiver<i32>) {
    match response_rx.await {
        Ok(result) => log_success_response(id, result),
        Err(_) => log_response_error(id),
    }
}

// ログ関連の関数
fn log_send_error(id: i32, error: mpsc::error::SendError<InferenceRequest>) {
    eprintln!("[Client {}] リクエスト送信エラー: {}", id, error);
}

fn log_request_sent(id: i32) {
    println!("[Client {}] リクエスト投入完了。結果を待機中...", id);
}

fn log_success_response(id: i32, result: i32) {
    println!("[Client {}] 結果受信: {}", id, result);
}

fn log_response_error(id: i32) {
    eprintln!("[Client {}] 結果受信エラー", id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker() {
        let (mut worker, tx) = Worker::new(3);
        worker.start();

        let client_handles: Vec<_> = (0..5)
            .map(|i| {
                let tx = tx.clone();
                tokio::spawn(async move {
                    client_producer(i, tx).await;
                })
            })
            .collect();

        for handle in client_handles {
            handle.await.unwrap();
        }

        drop(tx);
        worker.stop().await.unwrap();
    }
}
