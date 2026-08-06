# Masked Test Data Creation Conditions

> 2026-07-08 update: 現行の実装は `official-api-name-only-v1` に変更した。テスト入力文と API カタログ上の正式 API 名へ同一の決定的正規化（HTML escape 復元、NFKC、小文字化、非英数字の空白化、空白統一、trim）を適用し、正規化後の連続トークン完全一致だけを `<API_NAME>` に置換する。手作業 alias 辞書や意味的補完は使用しない。


作成日: 2026-07-08

この資料は、別のLLMに masked テストデータの作成条件を渡すための引き継ぎ資料である。対象は API 名を `[API_NAME]` に置換したテストデータであり、学習データや新しい train/test split の作成条件ではない。

## 1. LLM向け要約

- `masked` は、テスト入力文中の候補 API 名または確認済み alias を `[API_NAME]` に置換することを指す。
- マスク対象は `description` のみである。`categories`、正解APIリスト `apis`、`id`、階層ラベル、JSONキーは変更しない。
- 正解APIリストはマスク判定に使わない。全336サンプルに対して、909候補APIから作った固定辞書を同一に適用する。
- 現行実装は `simple-v1` である。API名辞書との正規化済み whole-phrase 完全一致だけを採用し、意味推論やサンプルごとの正解参照は行わない。
- layer1 のマスク済みデータを source とし、階層パターン別 masked データでは各 `category_hierarchy_*.json` から `gt_layer*` を作り直す。
- 条件分析サブセットを masked データに適用する場合は、既存の sample_id 分割をそのまま投影する。マスク後の description で `word_count` を再計算しない。

## 2. 主要ファイル

プロジェクトルート: `/home/nakagawa/lab-MLMLC`

| 役割 | パス |
| --- | --- |
| unmasked layer1 test data | `codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json` |
| masked layer1 test data | `codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json` |
| layer1 masking script | `codex/baseline/mask_test_api_names.py` |
| masking policy | `codex/baseline/API_NAME_MASKING_POLICY.md` |
| dictionary config | `codex/baseline/api_name_masking_dictionary.json` |
| resolved dictionary | `codex/baseline/api_name_masking_resolved_dictionary.json` |
| masking audit | `codex/baseline/api_name_masking_audit.json` |
| hierarchy-pattern masked generation script | `codex/scripts/create_masked_hierarchy_pattern_testdata.py` |
| condition split source CSV | `codex/analysis_test_subset_conditions/sample_condition_flags.csv` |
| condition subset report | `codex/test_subsets/condition_analysis/reports/subset_creation_report.md` |

## 3. layer1 masked データの入力と出力

入力は既存の336件テストデータである。JSONルートは dict で、各キーが sample_id 相当の mashup 名、値は次の list 形式である。

```text
[categories, description, apis, id]
```

出力は同じキー順と同じ list 形式を保つ。ただし `description` だけがマスク済み文字列になる。

検証済み件数:

| 項目 | 値 |
| --- | --- |
| total_samples | 336 |
| masked_samples | 178 |
| unmasked_samples | 158 |
| total_replacements | 317 |
| resolved_dictionary_entries | 944 |
| case_sensitive_replacements | 14 |

監査ログ上の validation はすべて pass している。

| validation | 値 |
| --- | --- |
| sample_count_is_336 | true |
| same_keys_and_order | true |
| non_description_fields_unchanged | true |
| idempotent | true |
| remaining_unmasked_dictionary_matches | 0 |
| all_passed | true |

## 4. マスク辞書の作成条件

候補APIカタログは `api_dataset_like_manifest.json` の909 APIである。補助的に `new_api_data/api_data_new.json` を参照する。辞書設定の version は `simple-v1`。

辞書は次の要素から作る。

- 909候補APIの canonical name。ハイフンは空白相当として扱う。
- `api_name_masking_dictionary.json` に固定登録された確認済み alias。
- `case_sensitive_aliases` に固定登録された大小文字区別 alias。
- 一般語と衝突する canonical name は除外する。

辞書設定の内訳:

| 項目 | 値 |
| --- | --- |
| excluded_canonical_names | 14 |
| aliases | 47 |
| case_sensitive_aliases | 6 |

除外 canonical name:

```text
23, 2, active, box, compete, domain, food, images, indeed, numbers, photo, sms, vast, vk
```

注意: `API_NAME_MASKING_POLICY.md` の後半には contextual alias などの調査記録があるが、現行実装で優先されるのは同ファイル冒頭の「採用する簡易基準」である。つまり、現行スクリプトは文脈を広く推論せず、固定辞書との正規化済み完全一致を使う。

## 4.1 alias を機械的に採用する基準案

今後 alias の基準を人手判断ではなく機械的に揃える場合は、次のルールに固定する。目的は、担当者ごとの判断差をなくし、同じ入力カタログから常に同じ alias 辞書を生成できるようにすることである。

### 採用する alias

次のいずれかに該当する表記だけを alias として採用する。

1. **canonical name 由来の表記**
   - 候補APIキーを正規化して作る。
   - 例: `google-maps` -> `google maps`
   - 処理: lowercase、NFKC、HTML entity decode、`-` / `_` / 連続空白を単語区切りへ正規化。

2. **full catalog の正式 name/title 由来の表記**
   - `new_api_data/api_data_new.json` など、APIカタログ内の name/title 相当フィールドにある表記だけを使う。
   - そのAPIの候補キーと1対1または明示的に対応づけられる場合だけ採用する。

3. **カタログ説明文内の明示パターン由来の表記**
   - 説明文から抽出する場合は、固定正規表現に一致したものだけを採用する。
   - 例: `also known as X`、`formerly X`、`X API`、`X REST API` のようなパターン。
   - 抽出した `X` は候補APIの canonical name または正式 name/title と高い文字列類似度を持つ場合だけ採用する。

4. **大小文字で固有名と判定できる表記**
   - `Box`、`Last.fm`、`Context.IO` のように、大文字・記号・ドメイン風表記がカタログ上で確認できる場合だけ採用する。
   - 小文字化すると一般語になるものは case-sensitive alias としてのみ採用する。

### 採用しない alias

次は機械的に不採用にする。

- 正解APIリスト `apis` だけを根拠にした表記。
- テストデータの description に出てきたという理由だけの表記。
- 3文字以下の一般語、数字のみ、または stopword 相当の表記。
- denylist に入っている一般語。
- canonical name の一部だけを切り出した表記。
- 会社名・プラットフォーム名だけで、複数APIを指し得る表記。
- 意味推論で対応づけた表記。

### 機械的な採用フロー

alias 辞書生成は次の順に行う。

1. 909候補APIの canonical name から base alias を作る。
2. full catalog の name/title から catalog alias を作る。
3. catalog description に固定正規表現を適用し、明示的な別名候補だけを抽出する。
4. 全 alias 候補を同じ正規化関数で normalized alias に変換する。
5. normalized alias が denylist、短すぎる語、数字のみ、一般語に該当する場合は除外する。
6. 同じ normalized alias が複数APIに対応する場合は、固有サービス名として安全なものだけ残す。安全判定できない場合は除外する。
7. case-sensitive にしないと一般語と衝突するものは、元表記の大小文字・記号が一致する場合だけ採用する。
8. 生成結果を `api_name_masking_resolved_dictionary.json` として保存し、entry_count と除外理由別件数を監査ログに出す。

この機械的基準を採用する場合、人手で追加した alias は原則使わない。どうしても追加する場合も、`manual_override` として分離し、通常の実験条件とは別扱いにする。

## 5. 文字列一致と置換条件

一致判定は `mask_test_api_names.py` で行う。

- Unicode NFKC、HTML entity、curly apostrophe、句読点、空白を比較用に正規化する。
- `+` は比較用に `plus` として扱う。
- 通常 alias は case-insensitive、`case_sensitive_aliases` は case-sensitive で照合する。
- 正規化後の whole-phrase 完全一致だけを採用する。別単語の部分文字列は採用しない。
- 一致範囲が重なる場合は最長一致を採用する。
- 同じ範囲に複数の候補APIが対応しても、置換は1回だけ行う。
- 既にある `[API_NAME]` は保護し、再実行でさらに置換しない。
- 採用された全範囲を `[API_NAME]` に置換する。

マスク判定に使わないもの:

- 各サンプルの正解APIリスト `apis`
- 正解カテゴリ `categories`
- 階層ラベル
- モデルの予測結果
- 意味推論、類義語推論、外部検索

## 6. マスクするもの、しないもの

マスクする:

- 909候補APIの canonical name に由来する固有API名。
- 固定辞書に登録済みの surface alias。
- 正解APIでなくても、候補API辞書に含まれる名前が description に出た場合。

マスクしない:

- `API`、`APIs`、`web service` などの種類名。
- 機能、カテゴリ、通信方式、データ型、一般的な技術語。
- 候補API辞書にないサービス名や製品名。
- 正解APIリストだけを根拠にした表記。

## 7. hierarchy-pattern masked データの作成条件

階層パターン別 masked データは `create_masked_hierarchy_pattern_testdata.py` で作る。source は layer1 のマスク済み336件である。

デフォルト source:

```text
codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json
```

デフォルト pattern dir:

```text
codex/output/by_pattern
```

各 pattern directory での処理条件:

1. `category_hierarchy_{pattern}.json` を優先して読む。なければ `category_hierarchy_*.json` を探す。複数候補があり、`max_depth` と `category_paths` が一致しない場合はエラー。
2. `max_depth` を階層深さとして扱う。
3. `*_drop_unknown_samples.json` のうち、既存 masked 出力を除外し、`gt_layer` 数が `max_depth + 1` のものを full-depth template として1つだけ選ぶ。該当が0件または複数件ならエラー。
4. source と template の sample_id 集合が完全一致することを要求する。
5. source の各 entry は `[categories, description, apis, id]` でなければならない。
6. 各 category は pattern の `category_paths` に存在し、path 長が `max_depth` と一致しなければならない。未知カテゴリや不完全 path があればエラー。
7. `gt_layer1` から `gt_layer{max_depth}` は、正解カテゴリの path prefix を `-` 連結したラベルとして作る。
8. `gt_layer{max_depth + 1}` は、正解カテゴリ名の sorted unique list として作る。
9. 出力フィールド順は template の順序を保持する。

出力ファイル名は template stem に次の suffix を付ける。

```text
_categories_from_collaboration_api_name_masked.json
```

現在確認できる hierarchy-pattern masked 出力は19ファイルである。各ファイルは336件を持つ。`condition_analysis_by_pattern` の既存レポートは unmasked の `*_drop_unknown_samples.json` を22ソース処理しているが、masked 生成は各 pattern directory の full-depth template を1つ選ぶため、intermediate layer1 用の追加ファイルは対象外である。

## 8. 条件分析サブセットとの関係

既存の condition-analysis 9分割は、336件の sample_id を以下の条件で切ったものである。新しい split ではない。

| subset | condition | count |
| --- | --- | --- |
| overall | all 336 existing test samples | 336 |
| shorter_desc | `word_count <= 24` | 171 |
| longer_desc | `word_count >= 25` | 165 |
| fewer_candidates | `sum_oracle_category_candidate_size <= 52` | 195 |
| more_candidates | `sum_oracle_category_candidate_size >= 53` | 141 |
| shorter_desc_fewer_candidates | `word_count <= 24 and sum_oracle_category_candidate_size <= 52` | 107 |
| shorter_desc_more_candidates | `word_count <= 24 and sum_oracle_category_candidate_size >= 53` | 64 |
| longer_desc_fewer_candidates | `word_count >= 25 and sum_oracle_category_candidate_size <= 52` | 88 |
| longer_desc_more_candidates | `word_count >= 25 and sum_oracle_category_candidate_size >= 53` | 77 |

masked データへこの条件分析を適用する場合は、既存の sample_id 分割をそのまま投影する。マスク後の description は `[API_NAME]` を含むため、`word_count` を再計算すると元の実験条件とずれる可能性がある。

`sum_oracle_category_candidate_size` は、正解カテゴリパスが分かっていると仮定した後に残るAPI候補数である。カテゴリ予測自体の難しさを直接表す指標ではない。

## 9. 再生成コマンド

以下は `/home/nakagawa/lab-MLMLC` から実行する想定である。

layer1 masked データを再生成する:

```bash
python3 codex/baseline/mask_test_api_names.py
```

階層パターン別 masked データを再生成する。既存出力を置き換える場合は `--overwrite` が必要である。

```bash
python3 codex/scripts/create_masked_hierarchy_pattern_testdata.py --overwrite
```

masked layer1 に既存 condition split を投影する場合の例:

```bash
python3 codex/scripts/create_condition_analysis_subsets.py \
  --test-data codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json \
  --output-dir codex/test_subsets/condition_analysis_api_name_masked
```

masked hierarchy-pattern データに既存 condition split を投影する場合の例:

```bash
python3 codex/scripts/create_condition_analysis_hierarchy_pattern_subsets.py \
  --source-glob "*/*_api_name_masked.json" \
  --output-dir codex/test_subsets/condition_analysis_by_pattern_api_name_masked
```

## 10. 検証観点

再生成後は次を確認する。

- layer1 masked JSON は336件である。
- unmasked と masked の key 順が同じである。
- `description` 以外のフィールドが変わっていない。
- 2回目のマスク実行で追加置換が発生しない。
- `api_name_masking_audit.json` の `validation.all_passed` が true である。
- hierarchy-pattern masked JSON は各336件である。
- hierarchy-pattern masked JSON の sample_id 集合は source と template と完全一致する。
- condition split を作る場合、9分割の件数は既存 report と一致する。

## 11. やってはいけないこと

- 正解APIリスト `apis` を使って description をマスクしない。
- マスク済みJSONを手編集しない。alias や除外語を変える場合は辞書設定を更新して全件再生成する。
- `description` 以外の正解ラベルや metadata をマスクしない。
- マスク後の `word_count` で condition-analysis の閾値を再計算しない。
- 外部検索や意味推論で alias を追加しない。採用する alias は固定辞書に根拠付きで追加する。

## 12. 現在の監査ハッシュ

`api_name_masking_audit.json` に記録されている SHA-256:

| 対象 | sha256 |
| --- | --- |
| input | `6c78aad686e8ea78d6ea6937ef933da36c18e1c341ff8ed54bbadbe1e242bdb3` |
| output | `8e168381d7e6c386f4ebd202fe4a2d5c496bc2c7e79aaa372a29cff486ad2baf` |
| candidate_catalog | `c5d3bd19c19026e8380e24f01896650badf749ad5c57a5f6cc3a5b91aa6b2df9` |
| dictionary_configuration | `34a17f159a742b7e575962c2bc3623203c375751afb45d70d4d967efd63a7e3b` |
| resolved_dictionary | `0bdfda8805b22eac9eada959fdeaea89f25860ba45d0e6b7fbf1eac76a5283e1` |
