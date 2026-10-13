# 分岐位置の抽選と実行記録

`schedule.py` は木の生成前に分岐位置を決め、経路ごとの実行履歴を管理します。方策や勝敗を見て位置を選び直しません。collector の処理ループは呼出側が用意します。

## 木ごとの予定を作る

```python
import numpy as np

from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.rng import create_rng_streams
from lfu_tree_self_play.schedule import BranchPath, sample_branch_plan

game = ReversiPyGame()
root = game.initial_state()
planes = root.observation
empty_squares = int(np.count_nonzero(planes[0] + planes[1] == 0))
streams = create_rng_streams(7)
plan = sample_branch_plan(
    empty_squares,
    streams.branch_position,
    levels=2,
    strategy="uniform",
)
path = BranchPath(plan)
```

`empty_squares` は根局面の空きマス数 H（0〜64 の Python int）、`levels` は 0・1・2 です。bool や数値文字列は受け付けません。戻り値は既存の `BranchPlan` で、幅は 4、`positions` は昇順・重複なしの tuple です。位置 d は根から d 回石を置いた後、次の石を置く直前を意味します。位置 0 は根です。

| `strategy` | 抽選規則 |
| --- | --- |
| `uniform` | `{0, …, H−1}` から `min(levels, H)` 個を一様に非復元抽出 |
| `stratified` | 候補を連続したほぼ等しい 3 区間に分け、空でない区間から最大 `levels` 区間を等確率で非復元抽出し、各区間から 1 位置を一様抽選 |
| `fixed` | `levels=1` は位置 20、`levels=2` は位置 20・40。範囲外の位置は除外し、別の位置で補充しない |

層化区間の余りは前の区間から 1 個ずつ配ります。H=8 なら `[0, 3)`、`[3, 6)`、`[6, 8)`、H=2 なら `[0, 1)`、`[1, 2)`、空区間です。区間の選択確率は同じですが、区間長が違えば個々の位置の選択確率は異なります。

`levels=0` はどの strategy でも分岐なしです。H=0 は予定位置なし、H=1 のランダム条件は最大 1 位置になります。分岐なし・候補なし・固定条件では RNG を消費しません。入力の誤りも抽選前に拒否します。

位置抽選には `streams.branch_position` を渡し、行動抽出や兄弟の継続試行と RNG を分離します。同じ seed と抽選順序なら同じ予定を再現できます。木ごとに新しい予定を一度だけ作り、その木の全経路へ同じ予定を渡してください。同じ seed の RNG を木ごとに作り直すと、同じ予定を繰り返してしまいます。

## 経路ごとに分岐を判定する

パスと終局を解決した `GameState` を、石を置く直前に渡します。

```python
branch, next_path = path.visit(root, placement_depth=0)
```

`branch=True` なら呼出側が `plan.width` 個の行動標本を復元抽出します。同じ手を引いた標本も別の子として扱い、各子へ同じ `next_path` を渡します。履歴は immutable なので、一方の子で分岐しても他方の履歴は変わりません。`branch=False` の非終局局面では単一の継続経路を進めます。

`visit` は `(分岐するか, 更新後の履歴)` を返します。呼出側は必ず返された履歴を保持します。分岐を実行した位置へ再び `visit` しても分岐しません。`placement_depth` は根からの実際の着手数で、石を置いたときだけ 1 増やします。既存 `Game.step` は強制パスを解決するため、同じ手番が続いても着手の増分は 1 です。

終局では位置が一致しても分岐しません。未到達の予定は `plan.positions` に残り、別の位置への移動・再抽選・補充をしません。根局面や子局面には、検証済み game adapter がパスを解決した状態を渡す必要があります。

`BranchPath.planned_count` は経路上の予定位置数、`executed_count` はその経路で実行した位置数です。`executed_positions` は実行履歴、`unexecuted_positions` はまだ実行していない位置です。終局前の `unexecuted_positions` には今後到達しうる位置も含まれます。

## 完了 record の予定・実行を集計する

予定は `CollectionRecord.branch_plan`、実行した分岐は `sibling_groups` と親ノードの `placement_depth` に保存します。[既存の保存 API](dataset-store.md) で両方を保存でき、schema version は 1 のままです。経路の `BranchPath` は収集中の補助情報で、保存 record には含めません。

```python
from lfu_tree_self_play.schedule import summarize_branch_execution

summary = summarize_branch_execution(record, game)
for depth, planned, executed in zip(
    summary.planned_positions,
    summary.planned_by_position,
    summary.executions_by_position,
    strict=True,
):
    print(depth, planned, executed)
```

`summarize_branch_execution` はまず全 record を `validate_record` で検証します。未完了の木や、本来分岐すべき非終局ノードに兄弟群がない record を、実行数 0 として扱うことはありません。

| field / property | 木全体での意味 |
| --- | --- |
| `planned_positions` | 全経路で共有する予定位置 |
| `planned_by_position` | 全経路が各予定位置に到達する場合の分岐群数。予定位置の順に `1, width, …` |
| `executions_by_position` | 各予定位置で実際に生成された兄弟群数。未到達なら 0 |
| `planned_count` | `planned_by_position` の合計 |
| `executed_count` | `executions_by_position` の合計 |

幅 4・2 段なら予定分岐数は `1 + 4 = 5` 群です。全経路が到達すれば実行数も 5、最初の分岐の全標本が次の予定位置より前に終局すれば実行数は 1 です。これは分岐位置の数（2）、辺の数、終局の数とは異なる計数です。早期終局の影響は予定位置と各位置の実行数を一緒に報告してください。

## 検証

パッケージのディレクトリで実行します。

```sh
uv run --locked pytest tests/test_schedule.py
```

H=0・1、候補範囲外、分岐なし・1 段・2 段、一様・層化の抽選頻度、独立した RNG、実際のオセロの早期終局・強制パス、全経路への予定の共有、保存後の集計を確認します。要件の正本は [T13](research/milestones/02-collection.md#t13-一様層化固定の分岐位置schedule) です。
