# Policy/value モデル

[← ドキュメント一覧](README.md)

`PolicyValueModel` は `GameState.observation` と同じ4面（黒石、白石、手番、合法手）を `(batch, 4, 8, 8)` の
float32 tensor で受け取り、`policy_logits` `(batch, 64)` と着手者視点の
`value` `(batch,)` を返します。残差ブロックは Transformers の
`ResNetBasicLayer` を既定で4個使い、盤面の8×8を維持します。`PolicyValueConfig(width=64, num_blocks=4)` で幅とブロック数を指定できます。value は黒手番なら黒、白手番なら白の視点で、範囲は `[-1, 1]` です。

## 局面を推論する

```python
import torch

from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueModel, masked_policy_logits

state = ReversiPyGame().initial_state()
observations = torch.from_numpy(state.observation).unsqueeze(0)
model = PolicyValueModel()
model.eval()
with torch.no_grad():
    output = model(observations)
    legal_mask = observations[:, 3].flatten(start_dim=1).bool()
    probabilities = masked_policy_logits(output.policy_logits, legal_mask).softmax(
        dim=-1
    )
    value = output.value
```

policy の index は盤面の行優先順で、`index = row * 8 + column` です。上の例は重みを初期化したモデルを使います。保存済みモデルを使う場合は、`PolicyValueModel.from_pretrained(directory)` で置き換えてください。

合法手確率が必要なときは、観測の第4面を `(batch, 64)` の bool mask に変換し、
`masked_policy_logits(policy_logits, legal_mask).softmax(dim=-1)` を使います。
終局局面には合法手がないため、この関数は `ValueError` を返します。

収集ラウンドのモデルを固定して行動を復元抽出する場合は、[FrozenPolicySampler](sampling.md) を使います。

## 保存する

モデル重みと構成だけを保存する場合は Transformers の `save_pretrained(directory)` を使います。optimizer、乱数、消費位置を含む学習の再開には [CheckpointStore](checkpoints.md) を使ってください。
