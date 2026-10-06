# G1 保存基盤の検証

T10 は [test_recovery.py](../tests/test_recovery.py) による fixture 検証です。
T06 の共通 validator、T07 の return 集約、T08 の `DatasetStore`、T09 の
`CheckpointStore` を実際に使用します。collector、trainer、対戦 evaluator の実装ではありません。

## 停止・再開

別 interpreter の subprocess に `os._exit(73)` を注入し、finally や通常終了の
cleanup に依存せず停止します。子プロセスの期限は 90 秒。subprocess の起動は exec を
使い、Torch の複数 thread を持つ親から fork した Python をそのまま実行しません。

| 停止位置 | 停止直後に検証する状態 |
| --- | --- |
| 収集途中 / 完成後公開前 | scratch は残り、dataset に未公開単位は見えない |
| dataset rename 前 / 後 | staging のみ / 完成した単位だけが見える |
| synthetic optimizer step 後 | genesis が確定状態で、入力は未消費 |
| checkpoint runtime 書込後 / rename 前 | staging は残り、genesis と未消費位置を保持 |
| checkpoint rename 後 | モデル generation と入力消費が一体で確定 |
| 評価計算後 / result commit rename 前 | 未完了評価は manifest にない |
| result commit rename 後 | 原結果と identity が確定し、再開時に再確定しない |

11 境界ごとに空の保存先から停止させ、実ストアから二回再開します。別の試験では同じ run を
収集途中、公開後、学習途中、checkpoint 確定後、最初の評価確定後、次の評価計算後で
順に停止します。完了済み開始局面ペアを保持し、未確定ペアだけを再試行します。

無停止 run と再開 run の dataset ID・checksum・全 JSON 意味内容、固定入力 manifest、
確定消費 ID、model tensor、Adam state、scheduler、module ごとの train/eval mode、
global Python/NumPy/Torch RNG と四つの stream RNG、generation、stage、cost、
評価 identity **と原結果 payload** を比較します。固定入力への model の次の出力も一致します。
確定履歴は genesis、学習 1 回、評価ペア 2 件の四つだけで、両入力は一度だけ消費されます。
dataset は generation 0、入力は generation 0、更新モデルは generation 1 を保持します。
未確定計算の再実行は許容し、二重学習・二重評価の判定は実 manifest の確定効果に対して行います。

## G1 条件との対応

| 条件 | fixture と証拠 |
| --- | --- |
| H=0 / H=1 | 一着で終局する Reversi 局面の空分岐予定 / 根で一段分岐。1 辺 / 2 独立辺を保存・読込 |
| 早期終局・子孫数差 | strict enumerated game の根の一標本は早期 +1 終局、他方は -1 と 0 の子標本。再帰平均は 0.25 |
| 同一手の重複 | 同じ action の別 edge・child ID と継続乱数情報を保存。子標本の重みは各 0.25 |
| 強制パス | 実 Reversi 局面で連続した白手番を保持し、depth parity に依存しない target を確認 |
| 経路重み | 保存・読込した全兄弟群で子の重み合計が親の重みに一致 |
| 完全 round-trip | 上記四 fixture の record と全 target が読込前後で一致 |
| 処理境界の再開 | 11 境界と連続停止の subprocess 試験で無停止 run と一致 |

この結果は **M1 保存基盤の G1 fixture 検証** に限定します。任意の production collector の
分岐予定実行、実学習、対戦評価、停電耐性、network filesystem、異なる device への移行を
証明しません。T08/T09 の POSIX local filesystem 契約を前提とし、production 接続は
後続 M2/M3 の実装・検証で扱います。

## 再現コマンド

package directory で `PYTHONPATH=src python -m pytest tests/test_recovery.py -q`、
`PYTHONPATH=src python -m pytest -q` を実行します。repo root の CI 相当 lint は
`uvx --offline --from ruff==0.16.8 ruff check --target-version py312 server/ml/models` と
`uvx --offline --from ruff==0.16.8 ruff format --check --target-version py312 server/ml/models`。
実行結果は T10 report に記録します。

2026-10-06 の実行結果：T10 の 16 cases を含む package 全体は **267 passed**
（176.60 秒）。4 warnings は既存 dataset-store tests の multithreaded fork に対する
DeprecationWarning。root の Ruff check は `All checks passed!`、format check は
`58 files already formatted`。T10 の subprocess は fresh exec を使用します。
