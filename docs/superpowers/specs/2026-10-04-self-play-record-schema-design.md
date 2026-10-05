# T06: 自己対戦データの共通 record schema

対象: [issue #70](https://github.com/Lfu001/reversi/issues/70)

正本: [M1 / T06](../../../server/ml/models/lfu_tree_self_play/docs/milestones/01-data-storage.md)、[実験計画](../../../server/ml/models/lfu_tree_self_play/docs/experiment-plan.md)、[実行モデル](../../../server/ml/models/lfu_tree_self_play/docs/execution-model.md)

## 目的と範囲

A の完了した独立対局と B/C の全枝が完了した木を、同じ型と検証器で扱う。親子関係、局面、手番、行動、旧 log probability、旧 value、モデル世代、分岐構成、乱数情報、終局結果、計算コストを欠落なく表現する。不正な親子関係とモデル世代混在を拒否する。

B/C は同じ収集方式を共有するため、データの収集方式は `independent` または `tree` とし、学習方式 B/C をスキーマへ持ち込まない。A の各対局は一本道の木として表現する。盤面が同じでも標本の ID を統合しない。

T07 の条件付き平均・経路重み計算、T08 のファイル形式・atomic 公開、T09 の checkpoint・manifest、T10 の停止・再開、収集器と trainer はこの変更に含めない。T06 では不変の意味モデルとその検証契約を作る。G1 通過前に保存データを trainer へ接続しない。

## 採用する構成

ユーザーが許可した Pydantic v2 (`>=2.12,<3`) を直接依存に追加し、`BaseModel` と tuple を使用する。Python の対象は既存パッケージと同じ `>=3.14,<3.15`。共通基底に `strict=True`、`frozen=True`、`extra="forbid"`、`allow_inf_nan=False`、`validate_default=True`、`revalidate_instances="always"` を設定する。整数・文字列・tuple の暗黙変換を拒否し、未知 field と属性の再代入も拒否する。時間や旧 value 等の実数 field では Python の int / float の数値を受け付けるが、bool・文字列・非有限値を拒否する。

既存の `GameState` は `InstanceOf[GameState]` として保持し、Pydantic に NumPy の配列変換や既存 dataclass の再構築をさせない。既存 `Game` 契約も維持する。`frozen` の保証は属性の再代入に対するもので、内部の NumPy 配列まで深い不変性を保証するとは扱わない。局面の公開操作には既存 `GameState.observation` のコピーを使う。

群別のスキーマは検証規則の重複を招くため採用しない。未型付けの辞書のみを公開する方式は、ID 参照と標本の意味を利用側へ押し付けるため採用しない。保存形式は T08 で、この意味モデルを維持して選定する。

追加ファイルはパッケージの `records.py`、`record_validation.py`、`tests/test_record_types.py`、`tests/test_records.py`、`docs/records.md`。Pydantic の直接依存を `pyproject.toml` と `uv.lock` に反映し、README の詳細一覧と T06 正本へ説明・検証結果の参照を追加する。

## 型とデータの配置

ID は空でない文字列で、木内のノード・辺・兄弟群それぞれで一意とする。収集単位 ID は experiment と round の文脈で呼出側が発行する。T06 は ID の発行器、dataset 全体の重複防止、時刻順の意味を定義しない。

| 型 | 必須情報と意味 |
| --- | --- |
| `CollectionRecord` | `schema_version=1`、unit ID、experiment ID、round ID、固定モデルの世代、収集方式、root node ID、分岐予定、乱数情報、nodes / edges / sibling groups、コスト |
| `NodeRecord` | node ID、親から入る edge ID（根だけ `None`）、`GameState`、根からの着手数、モデル世代、旧 value（非終局のみ）、黒視点の終局結果（終局のみ） |
| `EdgeRecord` | edge ID、parent / child node ID、着手 action、旧 log probability、モデル世代、sampling RNG の参照、兄弟群 ID と sample index（通常辺は両方 `None`） |
| `SiblingGroupRecord` | group ID、親 node ID、順序付き edge ID tuple、実際に異なる action の数 |
| `BranchPlan` | 分岐幅（2 以上）、予定着手位置の昇順 tuple。分岐なしでは位置 tuple は空 |
| `RNGRecord` | アルゴリズム識別子、run seed、用途別 stream seed。各抽出について stream の参照と非負の抽出番号を保持し、兄弟標本には独立した continuation seed を保持 |
| `RNGStreamRecord` | stream ID、用途 (`branch_position` / `action_sampling` / `sibling_rollout`)、非負 seed |
| `SamplingRecord` | edge が使った stream ID、stream 内の抽出番号、分岐標本のみの continuation seed |
| `CostRecord` | 実際の環境遷移数、モデルで評価した局面数、推論 batch 数、収集経過秒、推論秒、環境遷移秒、木管理秒、ピークメモリ bytes |

実際の推論件数や遷移数は共有・キャッシュ・再計算の方式で変わるため、辺数との等値を強制しない。計数は非負整数、時間は有限の非負値とする。木管理時間などは collector が計測する値で、T06 は計測器を実装しない。保存時間・学習時間はまだ完了していない別段階なので、この収集コストには混ぜない。

旧 value は親局面の着手者視点で `[-1, 1]`、旧 log probability は合法手マスク・温度を適用した抽出時の有限値で `<= 0`。初期実験は温度 1 とし、この契約をドキュメントに明示する。検証器は元のモデルを実行しないため、保存確率と保存 value が当時のモデル出力だったこと自体を証明しない。

乱数情報は抽出系列と標本の追跡用で、途中再開の完全な RNG state は T09 の責任とする。用途は既存 `rng.py` と一致させ、位置抽選、行動抽出、兄弟の継続試行を区別する。評価 RNG は収集 record に不要。

`RNGRecord` は `algorithm="python.random.MT19937-v1"`、`run_seed`、stream record の tuple を保持する。主要 stream ID を `branch` / `action` / `sibling` とし、用途と seed をそれぞれ既存 `derive_seed(run_seed, purpose)` と照合する。追加 stream は `sibling_rollout` 用とし、分岐標本の continuation seed で開始する系列を記録する。辺の sampling は `action_sampling` または `sibling_rollout` 用 stream を参照し、位置抽選用 stream の行動抽出への流用を拒否する。兄弟 continuation seed は対応する追加 stream の seed と一致し、同一群では重複しない。これは seed と stream の追跡契約で、抽出値の確率的独立性の証明ではない。

## 局面と標本の意味

手番・合法手・終局は `GameState` を正本とする。ノードに別の手番フィールドを重複保存せず、`state.to_play` で取得する。観測は既存の黒石・白石・手番・合法手の 4 plane を保持する。

根からの着手数は辺を 1 本進むごとに 1 増える。強制パスは `Game.step` が解決するため、パス辺を保存しない。親子で手番が同じになる遷移も正常であり、深さの偶奇から符号を推測しない。

同じ action を引いた兄弟には別の edge ID、child node ID、sample index、continuation seed を割り当てる。直後の局面が同じでもノードを共有しない。これにより木は 1 親の構造を保ち、各標本の独立した終局結果を残せる。

終局結果は黒勝ち `+1`、引き分け `0`、黒負け `-1`。非終局ノードには終局結果を設定しない。終局ノードには旧 value や出辺を設定しない。完了 record なので、出辺のない非終局ノードは拒否する。

## 共通検証器

個々の record の構築時の型・範囲違反は Pydantic の `ValidationError` とする。`validate_record(record, game)` は正常な場合 `None` を返し、不正な場合は `RecordValidationError(ValueError)` を発生させる。入口で `CollectionRecord.model_validate(record)` を実行し、ネストされた既存インスタンスも再検証する。`model_copy(update=...)` 等の未検証更新で作られた型違反も、この入口で `ValidationError` を `RecordValidationError` に包んで拒否する。入口以降は型・範囲チェックを重複実装せず、木とゲームの整合性を検証する。メッセージに不正な型または field、該当する ID を含める。入力は補正・並び替え・書換えず、Pydantic の型エラー詳細、または最初に見つけた意味違反を報告する。

検証を以下の順に行う。

1. schema version、収集方式、ID、モデル世代、数値の型と範囲を確認する。整数としての bool、NaN、Inf、欠落した必要値を拒否する。モデル世代は非負整数とし、単位・全ノード・全辺で一致させる。
2. node / edge / group の ID 索引を作り、重複と参照切れを拒否する。根は存在し incoming edge はない。根以外は incoming edge がちょうど 1 本で、その edge の child ID と一致する。複数親、自己参照、循環、根から到達不能なノードを拒否する。
3. 各観測の shape、有限性、黒石・白石・合法手の binary 値、石の重複、合法手と占有マスの重複、全マスで統一された `+1/-1` の手番 plane を確認する。各辺の action が親の合法手に含まれ、`game.step(parent.state, action)` が保存した子の局面と完全に一致することを確認する。根局面の合法手 mask 自体の規則再計算は既存 `Game` インターフェースではできないため、根の生成は検証済み adapter を使うことを収集側の契約とする。
4. 保存した着手数が根 0、子は親 +1 になっていることを確認する。全葉の終局、黒視点結果と盤面結果の一致、終局ノードに子がないことを確認する。
5. `independent` では全非終局に出辺が 1 本、兄弟群なし、予定分岐位置は空とする。`tree` では予定位置が根の空きマス数 H の範囲内で、重複なく昇順であることを確認する。H=0 の終局根はノード 1 個・辺 0 本を許す。予定位置に達した非終局ノードは分岐幅どおりの兄弟群を持ち、それ以外は出辺が 1 本。予定位置より早く終局した経路は未実行の分岐を持たなくてよい。幅と段数は固定値 4 / 2 にハードコードせず、分岐なし・1 段・2 段等の計画を表現できる。
6. 各兄弟群の親と member edge の親が一致し、その親の全出辺と群の edge ID が一致することを確認する。sample index は `0..K-1` を重複なく使用し、edge の順序と一致する。同じ edge を別群に入れたり、通常辺に群の一部の情報だけを持たせたりすることを拒否する。異なる action の数は実データから照合し、同一 action の重複標本は受理する。
7. RNG の用途別参照と抽出番号を確認する。同じ行動抽出イベントの重複使用と、兄弟内の continuation seed の重複を拒否する。標本間の乱数独立性そのものを統計検定することや RNG の replay はここでは行わない。

## 合格条件と試験

| 試験 | 期待する結果 |
| --- | --- |
| A の完了した一本道、B/C の分岐木 | 同じ `validate_record` が受理する |
| 同じ action の兄弟、子孫数の違い、早期終局 | 標本 ID と終局結果を保持して受理する |
| H=0、H=1、強制パス | 終局根、候補位置、着手数、手番の契約を維持する |
| ID 重複、参照切れ、複数親、循環、孤立ノード | `RecordValidationError` で拒否する |
| ノード・辺のモデル世代だけを変更 | 共通検証器が世代混在として拒否する |
| 不正 action、保存した子の局面の改変 | 遷移不一致として拒否する |
| 未完成葉、終局結果の欠落・符号違い、終局に子を追加 | 完了単位の意味に反するため拒否する |
| 予定分岐の欠落・移動、群の過不足、重複 index | 分岐構成不一致として拒否する |
| 同一 action の標本を 1 本に統合 | 分岐幅・兄弟群の不一致として拒否する |
| bool を整数 field に使用、数値文字列、NaN / Inf、負コスト、未知 field | 構築時に Pydantic が拒否し、未検証更新を使っても共通検証器が拒否する |
| collection / node / edge の変更、検証前後の比較 | frozen / tuple の契約と非破壊検証を確認する |

合法な fixture は `ReversiPyGame` の実遷移で構築し、終局まで完了させる。既知の強制パス局面も使い、深さからの交互手番という誤実装を検出する。不正例は正常 fixture の必要箇所だけを置換し、失敗理由を照合する。

実装時にはまず現行 package のテストを実行して baseline を確認する。追加試験を失敗させてから実装し、追加試験・package 全体の pytest・CI と同じ Ruff lint / format を実行する。文書では T06 の検証結果のみを報告し、G1 全体の通過を主張しない。

## レビュー状態

2026-10-04: 共通型・共通検証器を採用する会話上の設計案をユーザーが承認。

2026-10-05: 書面確認への応答でユーザーが Pydantic の導入を許可。型制約を Pydantic の strict / frozen model に置き換え、共通検証器と T06 の範囲は維持する。

Pydantic の参照: [strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)、[設定](https://docs.pydantic.dev/latest/api/config/)、[モデルと未検証コピー](https://docs.pydantic.dev/latest/concepts/models/)。これらは型とインスタンスの検証用であり、保存形式の決定は T08 に残す。
