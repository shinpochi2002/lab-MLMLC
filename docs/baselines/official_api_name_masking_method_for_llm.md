# Official API Name Masking Method for Test Inputs

作成日: 2026-07-09
対象実装: `scripts/baselines/mask_test_api_names.py`
方式名: `official-api-name-only-v1`

この文書は、実験の再現のために、テスト入力文中のAPI名マスク処理を自己完結的に説明するものである。ここで説明する現行方式は、APIカタログに登録された正式API名のみを用いる。手作業のalias辞書、意味的補完、略称展開、stemming、lemmatization、単複変換、サンプルごとの正解APIラベルは使用しない。

## 1. 目的

本処理の目的は、テスト入力文中に正解または候補APIの正式名が直接出現することによるラベルリークを抑制することである。入力文にAPI名そのものが含まれている場合、モデルは要求内容の理解ではなく文字列一致によってAPIを推定できてしまう可能性がある。そのため、APIカタログ上の正式API名と一致する表現を、共通のプレースホルダ `<API_NAME>` に置換する。

この処理は評価用テストデータに対する前処理であり、マスク判定に正解APIリストを参照しない。すべてのテスト入力文に対して、同じAPIカタログ由来の正式API名集合を適用する。

## 2. 入力と出力

### 入力

既定の入力ファイルは次である。

```text
results/hierarchy/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json
```

既定のAPIカタログは次である。

```text
data/processed/root_json/api_dataset_like_manifest.json
```

このAPIカタログはJSON objectであり、キーをAPIカタログ上の正式API名として扱う。今回の実験では909 APIが登録されている。

### 出力

既定のマスク済みデータ出力は次である。

```text
results/hierarchy/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json
```

マスクログCSVは次である。

```text
scripts/baselines/api_name_masking_log.csv
```

監査結果JSONと、カタログから生成した正規化済み辞書JSONも出力する。

```text
scripts/baselines/api_name_masking_audit.json
scripts/baselines/api_name_masking_resolved_dictionary.json
```

過去の集計CSVは実験記録であり、現行スクリプトの既定出力には含まれない。

## 3. マスク対象

マスク対象は、APIカタログに登録されている正式API名のみである。APIカタログにない表記は、たとえ正解APIを示しているように見えてもマスクしない。

マスクするもの:

- APIカタログ上の正式API名と正規化後に完全一致する連続トークン列。
- 正解APIであるかどうかに関係なく、カタログ上の正式API名として入力文に出現したもの。

マスクしないもの:

- 手作業で追加したalias。
- カタログ正式名の一部だけの表記。
- 会社名、サービス名、略称、同義語、単複変換、意味推論による対応づけ。
- 正解APIリストだけを根拠にした表記。
- 正規化後に連続トークン完全一致しない表記。

例として、API名 `Box` は正規化後に `box` という独立トークンとして出現した場合のみマスクする。`sandbox`、`checkbox`、`boxer` の一部としてはマスクしない。

## 4. 正規化処理

APIカタログ上の正式API名と入力文の両方に、同一の決定的な正規化処理を適用する。正規化は次の順に行う。

1. HTMLエスケープを復元する。
2. Unicode正規化 NFKC を適用する。
3. 小文字化する。
4. ASCII英数字 `[a-z0-9]` 以外の文字を区切りとして扱う。
5. 連続する区切りを1つの空白に相当するものとして扱う。
6. 先頭と末尾の空白を除去する。

実装上は、正規化文字列そのものだけでなく、正規化後の各トークンが元入力文のどの文字範囲に対応するかも保持する。これにより、HTMLエスケープやNFKC正規化によって文字数が変化しても、実際に置換した元入力文上の `start_char` と `end_char` を正しく記録できる。

例:

| 元表記 | 正規化後 |
| --- | --- |
| `Google Maps` | `google maps` |
| `Google-Maps` | `google maps` |
| `Sales &amp; Invoices` | `sales invoices` |
| `ＡＰＩ １２３` | `api 123` |

## 5. 一致判定

正規化後のAPI正式名をトークン列として扱い、正規化後の入力文中に同じ連続トークン列がある場合のみ一致とする。部分文字列一致は認めない。

一致条件:

- API正式名と入力文側の表現を同じ正規化関数で正規化する。
- 正規化後API名のトークン列が、正規化後入力文の連続トークン列と完全一致する。
- トークン境界をまたがない部分一致は不可。
- 重複する一致範囲がある場合、長いAPI名を優先する。
- 同じ範囲が複数の正式API名に対応する場合、置換は1回だけ行う。

不採用の処理:

- `Amazon API` を `Amazon Product Advertising` とみなす処理。
- `Twitter API` から `API` を除去して `Twitter` と一致させる処理。
- `Google Map` を `Google Maps` とみなす単複変換。
- 略称展開、同義語展開、stemming、lemmatization。
- サンプルごとの正解APIリストを参照したマスク判定。

## 6. 置換処理

一致した元入力文上の範囲を、共通プレースホルダ `<API_NAME>` に置換する。

```text
Google Maps is used.
```

は次のようになる。

```text
<API_NAME> is used.
```

置換範囲は元入力文上の文字位置で記録する。`end_char` はPythonのスライスと同様に、置換範囲の終端の次の位置である。

すでに `<API_NAME>` が含まれている範囲は保護し、再実行時に再マスクしない。

## 7. ログ仕様

マスクログCSVは、1つの置換につき1行を保存する。最低限、次の列を持つ。

| 列名 | 意味 |
| --- | --- |
| `row_index` | 入力データ中の行番号または順序番号。0始まり。 |
| `text_id` | 入力テキストのID。JSON objectの場合はキーを使う。 |
| `official_api_id` | API ID列が存在する場合のID。今回の既定カタログでは空欄。 |
| `official_api_name` | APIカタログ上の正式API名。 |
| `normalized_api_name` | 正式API名を正規化した文字列。 |
| `matched_surface_form` | 入力文中で実際に置換された元表記。 |
| `normalized_matched_surface_form` | 置換された元表記を正規化した文字列。 |
| `start_char` | 元入力文上の置換開始位置。 |
| `end_char` | 元入力文上の置換終了位置の次の位置。 |

`matched_surface_form` と `normalized_matched_surface_form` を両方保存することで、実際の入力文にどのような表記揺れが存在し、それがどの正規化形としてマスクされたかを追跡できる。

## 8. マスク済みデータの追加情報

通常出力では、各入力レコードに少なくとも次の情報を追加する。

| 列名またはフィールド名 | 意味 |
| --- | --- |
| `original_text` | マスク前の元入力文。 |
| `masked_text` | `<API_NAME>` に置換した後の入力文。 |
| `num_masked_api_mentions` | その入力文でマスクされたAPI名出現数。 |

元の入力文列は既定では上書きしない。ただし、既存の階層評価スクリプトが従来のlist形式データを期待するため、評価実行用には次のオプションを使って元の説明文スロットを上書きし、list形式を維持する。

```bash
python3 scripts/baselines/mask_test_api_names.py --overwrite-text-column --legacy-list-output
```

## 9. 正規化後API名の重複

複数の正式API名が同じ正規化後文字列になる場合、それらは同一のマスク対象として扱う。つまり、同じ入力文上の範囲を複数回置換しない。

ただし、正規化後API名の重複は警告として出力し、監査ログにも保存する。API ID列が存在する場合、該当するIDもログに保存する。

今回の既定カタログでは、正規化後API名の重複は確認されていない。

## 10. 実験データでの集計結果

現行方式を336件のテストデータに適用した結果は次のとおりである。

| 項目 | 値 |
| --- | ---: |
| 入力サンプル数 | 336 |
| APIカタログ登録数 | 909 |
| マスクされたサンプル数 | 168 |
| マスクされなかったサンプル数 | 168 |
| マスクされたAPI名出現数 | 305 |
| マスクされたユニークAPI数 | 76 |
| マスクされた正規化表現数 | 76 |
| 正規化後API名の重複数 | 0 |

マスクされた正規化表現の例:

```text
google maps
twitter
youtube
facebook
flickr
images
box
photo
food
bing
foursquare
dropbox
google earth
indeed
kissmetrics
sms
```

`images`、`photo`、`food`、`box` などの一般語にも見える表現が含まれるのは、これらがAPIカタログ上の正式API名であり、正規化後に独立トークンとして入力文中に出現したためである。現行方式では、手作業の一般語denylistは使用しない。

## 11. 階層構造パターン評価用データの生成

以下は旧336件全体を再現する手順です。現在の構成選定・評価は[validation/test分割](../validation_test_split.md)に従って100件と236件を分離してください。第10節の集計も旧336件に対する記録です。

階層構造パターンごとの評価データは、layer1のマスク済みデータをもとに再生成する。

```bash
python3 scripts/data_prep/create_masked_hierarchy_pattern_testdata.py --overwrite
```

この処理では、各階層パターンの既存テンプレートからフィールド構造と階層ラベルを維持しつつ、descriptionだけをマスク済み入力文に置き換える。各パターンの出力は336サンプルを持つ。

全階層パターンで評価を実行するコマンドは次である。

```bash
python3 scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked --split full \
  --results-dir results/hierarchy/hierarchy_existing_models_official_api_name_masked \
  --continue-on-error
```

途中結果が存在する場合に再開するには、`--skip-existing` を付ける。

```bash
python3 scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked --split full \
  --results-dir results/hierarchy/hierarchy_existing_models_official_api_name_masked \
  --continue-on-error \
  --skip-existing
```

## 12. 論文記載用の短い説明例

本研究では、テスト入力文中にAPIカタログ上の正式API名が直接出現することによるラベルリークを抑制するため、評価用テストデータに対してAPI名マスク処理を適用した。マスク対象はAPIカタログに登録された正式API名のみに限定し、手作業のalias辞書、略称展開、同義語展開、stemming、lemmatization、単複変換、意味的補完、およびサンプルごとの正解APIラベルは使用しなかった。API正式名と入力文の双方に、HTMLエスケープの復元、Unicode NFKC正規化、小文字化、非英数字の空白化、空白の統一、前後空白の削除からなる同一の決定的正規化を適用した。そのうえで、正規化後のAPI正式名が正規化後入力文中の連続トークン列と完全一致した場合のみ、元入力文上の該当範囲を `<API_NAME>` に置換した。部分文字列一致や意味推論による一致は認めていない。各置換について、元入力文中の実表記、正規化後表記、対応する正式API名、開始・終了文字位置をログとして保存した。

## 13. 運用上の注意事項

- 古い資料には `[API_NAME]` やalias辞書方式の記述が残っている場合があるが、現行方式は `<API_NAME>` を用いる `official-api-name-only-v1` である。
- カタログ正式名のみを用いるため、`Dog CEO` のように人間には対応APIが推測できる表記でも、正式API名 `dog-ceo-dog` と正規化後に完全一致しなければマスクしない。
- `Box` は独立トークンとして出現した場合のみマスクする。`sandbox`、`checkbox`、`boxer` の部分文字列としてはマスクしない。
- マスク処理は正解APIラベルを使わない。正解APIリストは評価ラベルとして保持されるが、マスク判定には関与しない。
- 既存の階層評価パイプラインに流す場合は、`--overwrite-text-column --legacy-list-output` を使って従来のlist形式JSONを生成する。
