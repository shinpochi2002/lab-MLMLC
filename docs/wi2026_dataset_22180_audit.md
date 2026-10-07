# WI 2026 Dataset Count Audit: 22,180 Instances

## 1. 結論

`22,180` は保存済み成果物から再現できた。

ただし、`4,284 + 21,526 = 25,810` にならない理由は、raw API data `21,526` 件を全件結合していないためである。実際には、category prediction 用の expansion では raw API data から `17,961` 件だけが API 側追加候補として使われている。さらに、その `17,961` 件のうち `65` 件は filtered composite service data `4,284` 件と同名 key で衝突しており、最終件数の増加には寄与していない。

したがって、`22,180` の再現式は次である。

```text
22,180 = 4,284 + 17,961 - 65
```

差分 `3,630` の正体は、API 側のカテゴリ条件による除外 `3,565` 件と、mashup/API 間の同名 key collision `65` 件である。論文中の `22,180` という数値自体は保存済みデータと一致するが、「composite service data と API data を単純に結合した」という説明は不正確で、API filtering と key collision の説明が必要である。

## 2. 件数の生成経路

件数は JSON を parse して top-level 要素数で確認した。データファイルの行番号は代表レコードの開始行、コードの行番号は該当処理行を示す。

| 処理段階 | 件数 | 処理内容 | 根拠ファイル | 行番号 |
|---|---:|---|---|---:|
| raw mashup data | 6,184 | 元 mashup/composite service data | `data/raw/mashup_data/mashup_data_new.json` | 1 |
| excluded mashups | 1,900 | raw API catalog に存在しない API を 1 つ以上含む mashup を除外 | `data/api_category_list/softapilist.json` | N/A |
| filtered composite service data | 4,284 | `6,184 - 1,900`。ID を 1 から振り直し | `scripts/data_prep/create_filted_mashup.py` | 16-29 |
| filtered mashup artifact | 4,284 | filtered composite service data の保存済み成果物 | `data/raw/mashup_data/updated_mashup_new_data.json` | 1 |
| mashup-to-category data | 4,284 | 使用 API 名をカテゴリ名に置換 | `scripts/data_prep/cre_mash_apicate.py` | 20-40 |
| raw API data | 21,526 | API catalog の top-level key 数 | `data/raw/new_api_data/api_data_new.json` | 1 |
| processed API data for expansion | 17,961 | raw API の先頭カテゴリが対象カテゴリ集合に入る API のみ保持 | `scripts/data_prep/create_4cate_data_inc_nomash.py` | 47-54 |
| processed API artifact | 17,961 | API expansion 候補の保存済み成果物 | `data/raw/new_api_data/new_api_data_to_4category.json` | 1 |
| mashup/API key collision | -65 | processed API と filtered mashup の同名 key。API 側が後勝ち | `scripts/data_prep/create_4cate_data_inc_nomash.py` | 87 |
| expanded service dataset | 22,180 | `4,284 + 17,961 - 65` | `data/processed/softfill_mashup_data_new_api_inclu_nomash/nocategory/softfill_mashup_data_new_api_category_inclu_nomash.json` | 1 |
| service-level study split | 16,635 | `22,180` の service-level train/study 側 | `data/processed/softfill_mashup_data_new_api_inclu_nomash/nocategory/softfill_mashup_data_new_api_category_inclu_nomash_study_data.json` | 1 |
| service-level test split | 5,545 | `22,180` の service-level test 側 | `data/processed/softfill_mashup_data_new_api_inclu_nomash/nocategory/softfill_mashup_data_new_api_category_inclu_nomash_test_data.json` | 1 |

## 3. 4,284 の生成根拠

`4,284` は raw mashup data `6,184` 件から、raw API catalog に存在しない API を使用する mashup `1,900` 件を除外した結果である。

該当コードは `scripts/data_prep/create_filted_mashup.py` である。

- line 4: 入力 `mashup_data_new.json`
- line 5: 除外リスト `softapilist.json`
- line 16-19: `softapilist.json` に含まれる key を mashup data から削除
- line 21-25: ID を 1 から振り直し
- line 27-29: filtered data を保存

実データで確認した結果は次の通り。

| 項目 | 件数 |
|---|---:|
| raw mashup data | 6,184 |
| `softapilist.json` | 1,900 |
| raw mashup key と `softapilist.json` の一致 | 1,900 |
| raw API catalog に存在しない API を 1 つ以上含む mashup | 1,900 |
| raw API catalog の全 API が存在する mashup | 4,284 |
| `updated_mashup_new_data.json` | 4,284 |

`updated_mashup_new_data.json` の key set は、`mashup_data_new.json` から `softapilist.json` の 1,900 key を除いた集合と完全一致した。したがって、論文の "removed composite services that consisted of APIs not included in the API data" という説明は、`4,284` の説明としては事実に合っている。ただし、より正確には「API data に含まれない API を少なくとも 1 つ使用する composite service を除外した」と書く方がよい。

## 4. 21,526 の意味

`21,526` は `data/raw/new_api_data/api_data_new.json` の top-level key 数であり、raw API catalog の件数である。

raw API data の構造例:

```text
api_name -> [categories, description, providers, related_mashups, id]
```

確認結果:

| 項目 | 件数 |
|---|---:|
| raw API records | 21,526 |
| bad structure | 0 |
| empty category list | 0 |
| empty description | 0 |
| unique IDs | 21,526 |
| raw API categories, all category labels | 495 |
| raw API first categories | 436 |

重要なのは、raw API data `21,526` 件全体のカテゴリ数は `152` ではないという点である。raw API data 全体では全カテゴリで `495`、先頭カテゴリだけでも `436` が確認された。`152` は category prediction 用に使われる最終 label set のカテゴリ数である。

したがって、論文の "The API data contained 21,526 instances, and the API data had 152 categories." は、raw API data に対する説明として読むと不正確である。

## 5. 22,180 の詳細計算

最終成果物は次のファイルである。

```text
data/processed/softfill_mashup_data_new_api_inclu_nomash/nocategory/softfill_mashup_data_new_api_category_inclu_nomash.json
```

このファイルの top-level 要素数は `22,180` であり、ID も `1` から `22,180` まで重複なく振られていた。

再現式:

```text
22,180 = 4,284 + 17,961 - 65
```

検証結果:

| 検証項目 | 件数 |
|---|---:|
| filtered mashup category data | 4,284 |
| processed API data | 17,961 |
| mashup/API processed key overlap | 65 |
| union key count | 22,180 |
| final expanded dataset | 22,180 |
| final keys == mashup/API union | true |

`create_4cate_data_inc_nomash.py` line 87 では、次のように dict 展開で mashup 側と API 側を結合している。

```python
for api_name, data in {**processed_mashupapi_data, **processed_api_data}.items():
```

Python の dict 展開では、同じ key が存在する場合、後ろの dict が前の dict を上書きする。このため、processed API 側と mashup 側で同名だった `65` 件は、最終的に API 側のレコードとして残り、件数は増えない。

実データでも、同名衝突した 65 件について final record の description はすべて API 側と一致し、mashup 側とは一致しなかった。

## 6. 3,630 件の差の正体

単純計算との差分:

```text
21,526 - (22,180 - 4,284) = 3,630
```

実際の内訳:

```text
3,630 = 3,565 + 65
```

| 差分要因 | 件数 | 内容 |
|---|---:|---|
| API category filtering | 3,565 | raw API `21,526` 件のうち、先頭カテゴリが final 152 category set に含まれないため expansion 候補に入らなかった API |
| key collision | 65 | processed API `17,961` 件と filtered mashup `4,284` 件の同名 key。API 側が後勝ちで上書き |
| 合計 | 3,630 | `21,526` 全件が追加されない理由 |

除外された `3,565` 件について、カテゴリ配列欠損や description 欠損は確認されなかった。したがって、これは欠損除去ではなくカテゴリ条件による除外である。

除外 API の first category 上位例:

| first category | 除外件数 |
|---|---:|
| banking | 249 |
| artificial-intelligence | 88 |
| software-as-a-service | 82 |
| customer-relationship-management | 80 |
| content-management | 77 |
| mobile | 65 |
| human-resources | 53 |
| insurance | 52 |
| agriculture | 47 |
| hotels | 47 |

同名衝突 65 件の例:

```text
allogarage
alpinaut
antideo
appfollow
ask-amee
basecamp
clickup
contentful
facebook
mapmyrun
myfitnesspal
twitpic
ups-tracking
venmo
wikimapia
```

なお、raw API と filtered mashup の同名 key は合計 `70` 件あった。そのうち `65` 件は API processing を通過して上書き衝突になった。残り `5` 件は API 側ではカテゴリ条件により expansion 候補から落ちたため、mashup 側として final dataset に残っている。

API 側では落ちたが mashup 側として残った 5 件:

```text
binary-canary
mashit
perfecto
rebate-bus
talkdesk
```

## 7. 重複・欠損処理の有無

この `22,180` 生成経路では、pandas の `concat`、`merge`、`drop_duplicates`、`dropna` は使われていない。

実質的な除外・重複処理は次の 2 つである。

| 処理 | 根拠 | 内容 |
|---|---|---|
| `if new_elements` | `scripts/data_prep/create_4cate_data_inc_nomash.py` line 30, 54 | cluster/category mapping に該当するカテゴリがある record のみ保持 |
| dict key overwrite | `scripts/data_prep/create_4cate_data_inc_nomash.py` line 87 | `{**processed_mashupapi_data, **processed_api_data}` により同名 key は API 側が後勝ち |

raw API data にはカテゴリ配列欠損、description 欠損、ID 重複は確認されなかった。final expanded dataset でも top-level key と ID は `22,180` 件で重複なしだった。

## 8. Category Prediction 用データとの関係

`22,180` は category prediction 用に API data を追加した service-level dataset である。

ただし、実際の学習スクリプトはこの `22,180` 件ファイルをそのまま直接読むのではなく、後続処理で category membership ごとの flat list や階層ラベル付き study data を読む。

確認した後続処理:

| 処理 | 根拠 | 内容 |
|---|---|---|
| service-level split | `softfill_mashup_data_new_api_category_inclu_nomash_study_data.json`, `..._test_data.json` | `16,635 + 5,545 = 22,180`。key overlap なし |
| category flat list 化 | `scripts/data_prep/create_split_data.py` line 13-34 | service の category list を category ごとの row に展開 |
| category membership count | 実データ parse | `22,180` service records から `24,668` category-membership rows |
| saved original study/test flat data | `data/processed/studydata/original/...study_data.json`, `...test_data.json` | `21,463 + 3,205 = 24,668` |

したがって、論文で `22,180 instances` と書く場合は、これは「category expansion 後の service-level dataset」であると明記する必要がある。category membership row 数や階層学習用の study/test row 数とは別である。

## 9. 909 APIs との関係

`909` は `22,180` の生成原因ではない。

`909` は filtered mashup `4,284` 件に実際に出現する unique API 数であり、API recommendation の candidate API catalog である。

検証結果:

| 項目 | 件数 |
|---|---:|
| filtered mashups | 4,284 |
| unique APIs used in filtered mashups | 909 |
| `api_dataset_like_manifest.json` | 909 |
| raw API catalog hit for these APIs | 909 |
| `study_data_mashup_apis.json` + `test_data_mashup_apis.json` の used API union | 909 |

既存文書 `docs/baselines/README.md` line 12 でも、`api_dataset_like_manifest.json` の 909 件は `study_data_mashup_apis.json` と `test_data_mashup_apis.json` に実際に出現する API の union と説明されている。

`909` は API recommendation task の候補集合であり、category prediction 用の `17,961` API expansion records とは別の数である。ただし、`909` 件は `17,961` 件の processed API set には含まれていた。

## 10. 152 Categories との関係

`152` は raw API data 全体のカテゴリ数ではなく、category prediction dataset の最終カテゴリ集合である。

検証結果:

| 対象 | ユニークカテゴリ数 |
|---|---:|
| raw API data, all category labels | 495 |
| raw API data, first category only | 436 |
| filtered mashup category data | 152 |
| processed API first categories | 152 |
| final expanded dataset | 152 |

つまり、`152 categories` は `21,526` raw API records 全体の性質ではなく、`4,284` filtered mashup data と category prediction expansion に使われる `17,961` API records に共通する label set として説明するべきである。

## 11. 論文記述との照合

| 論文記述 | 判定 | 根拠 | 必要修正 |
|---|---|---|---|
| We used composite service data and API data to build the service recommendation model. | 説明不足 | API recommendation 用 candidate 909 と category prediction 用 expansion 17,961 が別系統 | recommendation task と category prediction task のデータを分けて説明する |
| After filtering, the composite service data contained 4,284 instances. | 正しい | raw mashup 6,184 から API catalog に存在しない API を使う 1,900 mashup を除外 | 除外条件を「少なくとも 1 つ API catalog 外 API を含む mashup」と具体化 |
| The API data contained 21,526 instances. | 正しい | `api_data_new.json` の top-level key 数 | raw API catalog の件数であると明記 |
| The API data had 152 categories. | 不正確 | raw API は all category 495、first category 436。152 は final label set | 「filtered/retained API records covered the 152-category label set」へ修正 |
| Because the number of composite service instances was small, we expanded the data for the category prediction task. | 正しい | `22,180` service-level dataset が存在 | API 全件ではなく filtered API records を追加したと補足 |
| We combined the composite service data and the API data into one service dataset for this task. | 不正確 | `21,526` 全件ではなく `17,961` 件を使用。さらに 65 key collision | filtered API records と key collision/overwrite を明記 |
| After this expansion, the dataset contained 22,180 instances. | 件数は正しいが説明不足 | final file の top-level len が 22,180 | `4,284 + 17,961 - 65` を説明 |

## 12. 最終判定

最終判定は **Case E** とする。

理由: filtered composite service data `4,284` 件の key は final dataset に全件存在するが、そのうち `65` 件は API 側と同名 key で衝突し、final record の内容は API 側に上書きされている。したがって、結合時に composite service 側の全件がそのまま保持されているわけではない。

補足として、**Case C** の性質もある。`21,526` は raw API data の件数であり、expansion 前に `17,961` 件へ filtering されているためである。

## 13. 論文で必要な修正

### 数値修正の要否

| 項目 | 判定 |
|---|---|
| `4,284` | 数値修正不要 |
| `21,526` | 数値修正不要。ただし raw API catalog 件数と明記が必要 |
| `22,180` | 数値修正不要。ただし計算根拠の説明が必要 |
| `909` | 数値修正不要。API recommendation candidate set として説明 |
| `152` | raw API 全体のカテゴリ数としては修正必要。final label set として説明 |

### 修正案 1: API data 21,526 と 152 categories

現在:

```text
The API data contained 21,526 instances, and the API data had 152 categories.
```

推奨:

```text
The raw API catalog contained 21,526 API records. For the category-prediction expansion, we retained 17,961 API records whose first category belonged to the 152-category label set used in the filtered composite-service data.
```

### 修正案 2: 22,180 expanded dataset

現在:

```text
We combined the composite service data and the API data into one service dataset for this task. After this expansion, the dataset contained 22,180 instances.
```

推奨:

```text
We merged the 4,284 filtered composite-service records with 17,961 filtered API records for the category-prediction task. Because 65 names appeared in both sources and were resolved by keeping the API-side record, the resulting expanded service-level dataset contained 22,180 records.
```

### 修正案 3: 4,284 filtered composite services

現在:

```text
After filtering, the composite service data contained 4,284 instances.
```

推奨:

```text
After removing 1,900 composite services that used at least one API absent from the raw API catalog, the composite-service data contained 4,284 instances.
```

### 修正案 4: 909 candidate APIs

必要に応じて追記:

```text
The 909 candidate APIs used in the API recommendation task were defined as the union of APIs actually used by the filtered composite services, and are distinct from the 17,961 API records used to expand the category-prediction dataset.
```

## 14. 不足・注意事項

次の生成元ファイルは現行 repository 内では見つからなかった。

- `kmean_4_category_cluster.json`
- `api_name_category.json`
- `softfill_mashup_data_new_api_category_inclu_nomash_split.json`

ただし、`22,180` の生成経路そのものは、保存済みの入力・中間・最終 JSON の key set と値の照合で再現できた。特に final dataset の key set は `filtered mashup 4,284` と `processed API 17,961` の union と完全一致し、同名衝突 `65` 件については API 側 record が保持されていることも確認できた。
