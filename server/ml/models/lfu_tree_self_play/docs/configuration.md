# 設定と乱数の管理

## 設定ファイル

```toml
[experiment]
name = "tree-comparison"

[seeds]
training = [1, 2, 3]
tuning = [101]
final_evaluation = [201]
```

`training` は学習、`tuning` は設定調整、`final_evaluation` は最終評価の seed です。モデル構成や学習条件を指定する field はありません。

## 検証する

パッケージのディレクトリで実行します。

```sh
uv run --locked lfu-tree-self-play check-config configs/example.toml
```

`check-config` は TOML の型と seed 集合を検証し、experiment ID を表示します。
ファイル不在、読込失敗、文字コード・構文エラー、設定エラーは標準エラー出力に表示し、終了コード 2 で終了します。
`uv run --locked python -m lfu_tree_self_play check-config configs/example.toml` でも実行できます。

設定は `[experiment]` の `name` と、`[seeds]` の `training`、`tuning`、`final_evaluation` を必須とします。
各 seed 集合は空にできず、0 以上の整数を重複なく指定します。学習・調整・最終評価の集合間でも seed を共有できません。
読み込んだ設定は読み取り専用の OmegaConf `DictConfig` として扱います。
seed 集合の記載順と TOML の整形は experiment ID に影響しません。設定値を変更すると ID が変わるため、dataset と checkpoint は別の保存先を使います。

## Python で設定と乱数を使う

```python
from pathlib import Path

from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.rng import create_rng_streams, derive_seed

config = load_config(Path("configs/example.toml"))
identifier = experiment_id(config)
run_seed = config.seeds.training[0]
streams = create_rng_streams(run_seed)
action_rng = streams.action_sampling
action_seed = derive_seed(run_seed, "action_sampling")
```

`create_rng_streams(run_seed)` は位置抽選、行動抽出、兄弟の継続試行、評価用の独立した `random.Random` を返します。
別の RNG 実装が必要な場合は `derive_seed(run_seed, purpose)` で同じ用途別 seed を取得できます。

| stream / purpose | 用途 |
| --- | --- |
| `branch_position` | 分岐位置の抽選 |
| `action_sampling` | 方策からの行動抽出 |
| `sibling_rollout` | 兄弟標本の継続試行 |
| `evaluation` | 評価 |

同じ run seed と用途からは、呼出し順に依存せず同じ seed を導出します。再開時に途中から乱数列を継続するには、seed から作り直さず [チェックポイント](checkpoints.md) に保存した RNG state を復元します。
