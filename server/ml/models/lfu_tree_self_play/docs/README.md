# 木状自己対戦学習・実験ドキュメント

このディレクトリは、[改訂済み実験計画](experiment-plan.md)を実行可能な研究開発タスクへ分割した入口です。詳細は必要な段階の文書だけを参照してください。

現在は計画段階です。確認実験の結果や、検証済みの新実装はまだありません。

## 最初に読む文書

1. [研究設計](research-design.md)：研究の問い、比較群、主張の範囲
2. [実行モデル](execution-model.md)：収集、学習、評価、再開の流れ
3. [検証ゲート](verification-gates.md)：次工程へ進む条件

研究設計の原典と数値を確認する場合は、[実験計画（日本語）](experiment-plan.md)または[Experimental plan (English)](experiment-plan.en.md)を参照してください。

## マイルストーン

| 段階 | 内容 | タスク | 開始条件 |
|---|---|---:|---|
| [M0](milestones/00-foundation.md) | 新規基盤と意味仕様 | T01〜T05 | なし |
| [M1](milestones/01-data-storage.md) | データモデルと永続化 | T06〜T10 | M0の関連成果 |
| [M2](milestones/02-collection.md) | A/B/Cデータ収集 | T11〜T17 | G0、M1の関連成果 |
| [M3](milestones/03-learning.md) | A/B/C学習 | T18〜T22 | G0、G1 |
| [M4](milestones/04-evaluation.md) | 評価と統計 | T23〜T28 | M0のモデル・ゲーム契約 |
| [M5](milestones/05-pilot.md) | 実行制御とPilot | T29〜T35 | G1、M2〜M4 |
| [M6](milestones/06-abc-study.md) | A/B/C本実験 | T36〜T39 | G4 |
| [M7](milestones/07-alphazero.md) | AlphaZero型D | T40〜T45 | G0、D本比較はM6完了後 |
| [M8](milestones/08-extensions.md) | 結果依存の拡張 | T46〜T51 | 基礎実験の結果 |
| [M9](milestones/09-reproducibility.md) | 再現と報告 | T52〜T54 | 実施対象の実験完了 |

基本順序は `M0 → M1 → M2/M3 → M4 → M5 → M6 → M7 → M8 → M9` です。M2とM3の一部は保存済みfixtureを使って並行できますが、実データとの接続はG1通過後に限ります。M7の参照MCTSはG0通過後に着手できますが、Dの本比較はM6とG6の完了後に行います。

## 文書の使い方

- 全体判断をする場合は、このREADMEと検証ゲートだけを確認します。
- 実装やレビューを始める場合は、対象マイルストーンだけを開きます。
- 個別タスクの設計時に型、関数、ファイル、デザインパターンを決めます。この文書群では固定しません。
- M8の全項目を一覧へ含めますが、複数の拡張を一つのrunへ同時投入しません。

## 全体完了の定義

- G0〜G8の結果が保存されている。
- A/B/C/Dの本実験が、成功、未到達、不成立のいずれかとして完結している。
- 実施対象となった拡張が、一要因ずつ基礎条件と比較されている。
- 任意の結果を設定、dataset、checkpoint、評価対局、集計入力まで追跡できる。
- 代表runと図表を保存済みmanifestから再生成できる。
- 仮説ごとの結論と主張できない範囲が最終報告に明記されている。
