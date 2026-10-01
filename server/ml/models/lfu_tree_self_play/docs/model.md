# Policy/value モデル

[← ドキュメント一覧](README.md)

`PolicyValueModel` は `GameState.observation` と同じ4面を `(batch, 4, 8, 8)` の
float32 tensor で受け取り、`policy_logits` `(batch, 64)` と着手者視点の
`value` `(batch,)` を返します。残差ブロックは Transformers の
`ResNetBasicLayer` を4個使い、盤面の8×8を維持します。

合法手確率が必要なときは、観測の第4面を `(batch, 64)` の bool mask に変換し、
`masked_policy_logits(policy_logits, legal_mask).softmax(dim=-1)` を使います。
終局局面には合法手がないため、この関数は `ValueError` を返します。

モデル重みは Transformers の `save_pretrained(directory)` で設定とともに保存し、
`PolicyValueModel.from_pretrained(directory)` で読み込みます。
