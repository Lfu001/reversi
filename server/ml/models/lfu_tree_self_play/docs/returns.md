# 終局リターン集約と経路重み

[T07 正本](milestones/01-data-storage.md#t07-終局リターン集約と経路重み) と [実験計画 §4](experiment-plan.md#4-勝敗の集約と学習目的) の集約を、[T06 record](records.md) に適用します。trainer、advantage、loss は追加していません。G1 通過前に保存データを trainer へ接続しません。

## API

```python
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.returns import ReturnTargets, aggregate_returns

# record: CollectionRecord
result: ReturnTargets = aggregate_returns(record, ReversiPyGame())
black_return = result.node_black_returns[record.root_node_id]
```

[aggregate_returns](../src/lfu_tree_self_play/returns.py) は最初に `validate_record(record, game)` を実行します。不正な完了単位は T06 と同じ `RecordValidationError` で拒否し、入力を補正・変更しません。ノード・辺 tuple の順序には依存しません。返り値は frozen dataclass で、各 mapping はコピーした辞書の読み取り専用 view です。ID は元 record の node / edge ID をそのまま使います。

| `ReturnTargets` field | キー | 意味 |
| --- | --- | --- |
| `node_black_returns` | node ID | 終局の黒視点結果、または子標本の再帰的等重み平均 |
| `edge_black_returns` | edge ID | 子 node の黒視点リターン |
| `node_weights` | node ID | 根で 1、子で親の重み / 実際の出辺数 |
| `edge_weights` | edge ID | その辺の子 node と同じ経路重み |
| `node_player_targets` | node ID | 黒視点リターン × その node の `state.to_play` |
| `edge_parent_targets` | edge ID | 辺の黒視点リターン × 親 node の `state.to_play` |

手番は記録された局面から取得します。強制パス後に親子で同じ手番になっても、深さの偶奇や無条件の符号反転を使いません。終局 node にも node target を提供しますが、学習対象にするかどうかは後工程の責任です。

## 集約と重み

走査は反復的に行い、根から重みを伝播した後、逆順でリターンを集約します。終局結果を黒視点のまま保持し、出辺が K 本なら子のリターンの和 / K を親のリターンにします。同じ action の標本も別の ID のまま集計します。通常の 1 本の辺は重みをそのまま伝え、分岐の K 本の辺の重みの合計は親の重みになります。早期終局、未到達の予定分岐、実際の終局手数による追加の正規化は行いません。固定定数 60 や根数による loss の正規化はこの API の責任外です。

手計算例：根の一方の子は早期終局で +1、もう一方はさらに 2 標本に分岐して -1 と 0 を得ます。後者の条件付き平均は -0.5、根は `(1 + (-0.5)) / 2 = 0.25` です。葉全件の単純平均 0 とは異なります。終局葉の経路重みは 0.5、0.25、0.25 で合計 1 です。

H=0 の終局根は node return と重み 1 を持ち、edge mapping は空です。H=1 の分岐でも重複 action を K 個の標本として維持します。

## 検証

[test_returns.py](../tests/test_returns.py) は小さな列挙可能ゲームで上記の手計算、重複 action の異なる後続結果、非交代手番、早期終局、順序非依存、入力保持、出力不変性、不正 record 拒否を確認します。実際の Reversi adapter で幅 2/3/4、H=0/H=1、強制パスを確認します。この T07 の検証だけでは G1 全体の完了を意味しません。
