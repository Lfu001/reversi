# 根局面の生成と用途別集合の固定

初期盤面から0〜8回石を置いた局面を、学習・調整・最終評価の根として使います。
`ReversiPyGame` の合法手から各着手を一様に抽出します。パスの処理はゲームAPIが行い、石を置いた回数にパスを数えません。

## 一つの根局面を生成する

```python
from lfu_tree_self_play.roots import generate_root_position

root = generate_root_position(seed=42)
state = root.state
placements = root.placements
actions = root.actions
key = root.key
```

手数は0〜8から一様に抽出します。`generate_root_position(42, placements=8)` のように手数を指定することもできます。
同じseedと手数指定から同じ着手列・盤面が得られます。根局面専用の `root_position` RNGを内部で作るため、global RNGや方策・分岐位置・評価用RNGを消費しません。

`RootPosition` はseed、着手回数、着手列、`GameState` を保持する不変のrecordです。
`state.observation` はコピーを返します。着手列にはパスを含めず、各要素は0〜63の行優先のマス番号です。

## 学習・調整・最終評価の集合を固定する

```python
from lfu_tree_self_play.config import load_config
from lfu_tree_self_play.roots import RootRegistry, generate_root_registry

config = load_config("configs/example.toml")
registry = generate_root_registry(config, roots_per_seed=100)
registry.save("roots.json")

fixed = RootRegistry.load("roots.json", config)
training_roots = fixed.positions("training")
tuning_roots = fixed.positions("tuning")
final_roots = fixed.positions("final_evaluation")
registry_id = fixed.registry_id
```

既存の設定の `seeds.training`、`seeds.tuning`、`seeds.final_evaluation` を使います。
`roots_per_seed` は各seedに必要な根の数です。例の設定は学習seedが3個、調整と最終評価が各1個なので、上記では300・100・100個を作ります。既定値は各seedに1個です。

用途順は `training` → `tuning` → `final_evaluation`、各用途のseed順は昇順で固定します。
候補seedはrun seed、根のindex、試行番号から導出し、他の根の乱数消費に依存しません。
集合内と集合間の両方で、既に採用した根と同じ識別子の候補を除外します。
`RootEntry` の `purpose`、`run_seed`、`index`、`attempt`、`root` で採用した候補を追跡できます。

`max_attempts` は根一つあたりの試行上限で、既定値は1000です。
全seedの必要数を揃えられなければ `ValueError` となり、部分集合を返しません。
単体の生成では手数と各合法手を一様に選びますが、registryでは重複を除くため、採用された手数・盤面の分布は一様ではありません。特に0手・1手などの浅い局面は早い用途で採用されやすくなります。根集合を比較条件の一部として固定し、A/B/Cで同じ学習集合を使います。

## 重複と固定の意味

`position_key(state)` は黒石・白石・実際の手番を使い、盤面の回転4通りと反転を含む8対称から同一局面を判定します。
合法手planeは局面から導出できるため識別子に含めません。黒白交換と手番の変更は別局面です。
seedが異なっても、対称な盤面と同じ手番なら重複として扱います。

このAPIは共有する定番局面の例外を設けず、初期盤面も一つの用途だけに採用します。
用途間のroot共有がないことを保証するもので、学習中に到達する全局面との非重複や、未見盤面への一般化を保証するものではありません。
最終評価のrootをモデル選択に使う実行境界はT24・T28の対象です。このAPIだけで呼出側の用途を制限することはできません。

## 保存と読込の契約

JSONにはversion、experiment ID、各seedの根数、用途・run seed・index・試行番号、着手回数、着手列、局面識別子を保存します。
`registry_id` はその正規化JSONのSHA256です。設定、根数、用途、着手列などが変わるとIDも変わります。
生成規則を変える場合は保存versionも更新する必要があります。

`save` は既存ファイルを上書きせず、存在すれば `FileExistsError` を返します。
`load` は指定された設定とのexperiment ID・seed所属を確認し、候補seedから実際のゲーム遷移を再生して全recordを検証します。
重複、欠けたseed・index、着手列・識別子の不一致、未知のfield・version、重複JSON fieldは拒否します。

保存はオフラインで集合を固定するための通常のファイル書込です。停止・電源断時のatomicityを保証するAPIではありません。
読込では不完全なファイルを拒否します。確定したファイルと `registry_id` を実験成果物として保管してください。
