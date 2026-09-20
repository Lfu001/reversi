# M3：A/B/C学習

[← M2](02-collection.md) · [一覧](../README.md) · [検証ゲート](../verification-gates.md) · [次：M4](04-evaluation.md)

BとCでデータを共通化し、兄弟比較の有無だけを変更できる学習基盤を作ります。

## T18 通常advantageとleave-one-out advantage

- 依存：T05、T07
- 領域：教師生成、B/C差分
- 成果：非分岐辺にはR−V_old、分岐標本には兄弟leave-one-outを適用できる。
- 合格：同結果群で0、既知の勝敗差で手計算値と一致し、異なる局面や手番を混ぜて正規化しない。

## T19 経路重み付きPPO・value loss

- 依存：T04、T07、T11、T18
- 領域：方策loss、value loss、entropy、clip
- 成果：保存した行動だけにPPO surrogateを適用し、固定定数60と根数で正規化できる。
- 合格：θ=π_oldでratioが1、教師と旧確率へ勾配が流れず、共有辺を葉数だけ重複計上しない。

## T20 1 epoch trainerと更新量メトリクス

- 依存：T09、T19
- 領域：学習loop、AdamW、checkpoint
- 成果：一つの収集世代を1 epochだけ学習し、KL、clip率、勾配norm、policy entropy、value誤差を記録する。
- 合格：同じmanifestを二度消費せず、停止・再開後も同じ確定更新へ到達する。

## T21 同一保存木・同一初期値によるB/C比較

- 依存：T15、T18、T19、T20
- 領域：offline比較、差分管理
- 成果：同じ木と初期checkpointから、通常advantageのBと兄弟比較のCを別runとして更新できる。
- 合格：B/C間で変更される要因がadvantage構成だけであることをmanifestで検証できる。

## T22 黒・白別の条件付き方策改善試験

- 依存：T05、T21
- 領域：凍結相手、片側更新評価
- 成果：相手をπ_oldへ固定し、更新モデルを黒または白の片側だけに適用して期待リターン変化を測れる。
- 合格：両者を同時更新した自己対戦を改善指標に使わず、黒・白の結果を別々に保存する。

## 完了条件

同一入力を使ったB/C比較が可能で、[G3](../verification-gates.md#g3学習信号の小規模比較)に必要な学習指標を出力できること。
