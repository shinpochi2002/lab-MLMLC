# 公開ドキュメント

公開対象は [public_files.txt](public_files.txt) に、リポジトリルートからの相対パスで登録する。この一覧には一覧自身と本ページも含める。

| 文書 | 内容 |
|---|---|
| [分割とモデル選定](validation_test_split.md) | validation 100件・test 236件の固定分割と評価手順 |
| [階層モデルの命名](hierarchy_model_naming.md) | 構成識別名、モデル保存先、評価データの対応 |
| [Baseline実行ガイド](baselines/README.md) | 入出力、依存関係、実行例 |
| [API名マスク仕様](baselines/official_api_name_masking_method_for_llm.md) | 正式API名のみを使う前処理 |
| [過去のBaseline結果](baselines/BASELINE_RESULTS_SUMMARY.md) | 旧336件・マスクなしの実験記録 |
| [Early Commitment Scoreの監査](early_commitment_score_e_audit.md) | 実装式、再計算、定義上の注意点 |
| [データ件数の監査](wi2026_dataset_22180_audit.md) | 22,180件の生成過程と除外・衝突処理 |

## 実験記録の読み方

現在の評価手順はvalidation 100件・test 236件を使用する。これは過去に構成比較を行った336件の事後分割であり、test236件を一度も参照していない独立データとは扱わない。過去の336件の成績を現在のtest236件の成績として引用しない。

監査文書は保存済み成果物・コードに基づく調査記録である。数式の定義の相違やデータ生成上の制約も含めて公開する。参照するデータ・重み・生成結果の一部はGit管理対象外であり、コードと文書のcloneだけで全実験を再実行できるわけではない。

## 公開対象の管理

作業計画、未確定の方針、査読返信案、探索用レポートはローカルに保管する。一覧にないdocs配下のファイルはルートの`.gitignore`で除外する。

公開文書を追加するときは`public_files.txt`にパスを追加し、`.gitignore`にも対応する例外を追加する。次のコマンドで一覧のファイルをステージする（リポジトリ直下で実行）。

```bash
while IFS= read -r path; do
  git add -- "$path"
done < docs/public_files.txt
git add .gitignore
git diff --cached --stat
```

既に追跡している内部文書を除外する場合は、ファイルを保持して`git rm --cached -- <path>`で追跡を外す。過去のコミットに含まれる内容は、この設定では削除されない。
