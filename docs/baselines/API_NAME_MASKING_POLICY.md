# API Name Masking Policy for the 336-Sample Test Set

> 2026-07-08 update: 現行の `mask_test_api_names.py` は `official-api-name-only-v1` である。マスク対象は API カタログに登録された正式 API 名のみで、手作業 alias 辞書、denylist、意味推論、stemming、lemmatization、単複変換、略称展開は使わない。置換文字列は `<API_NAME>`。後続の alias/contextual alias に関する節は過去方針・調査記録として扱う。


## 0. 採用する簡易基準

実際のマスク生成では、この節を優先する。後続の節は境界事例の調査記録であり、文脈推論規則は実装しない。

1. 909 候補 API の canonical name と、API カタログで確認した表記違いだけを固定辞書に入れる。
2. 正解 API リスト `value[2]` は辞書作成にもマスク判定にも使わない。
3. Unicode、HTML entity、句読点、空白を比較用に正規化する。
4. 辞書との単語境界付き完全一致だけをマスクする。部分一致や意味推論は行わない。
5. 一致範囲が重なる場合は最長一致を採用する。
6. `images`、`photo`、`food`、`domain`、`sms` など一般語と衝突する canonical name は固定除外する。
7. 採用した一致範囲をすべて `[API_NAME]` に置換する。

表記揺れには、大文字・小文字、句読点、ドメイン風表記、同一 API の公式略称を含める。会社名・サービス名だけで特定 API を指さない表記は含めない。

使用ファイル:

- 辞書設定: `codex/baseline/api_name_masking_dictionary.json`
- 生成スクリプト: `codex/baseline/mask_test_api_names.py`
- マスク済みデータ: `codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json`
- 監査ログ: `codex/baseline/api_name_masking_audit.json`
- 解決済み辞書: `codex/baseline/api_name_masking_resolved_dictionary.json`

## 1. 目的

テスト入力文に候補 API の名前が直接書かれていることで、モデルが要求内容ではなく名前の文字列一致だけで API を当てることを防ぐ。

対象データは次の 336 件とする。

`codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json`

候補 API 辞書は次の 909 API とする。

`api_dataset_like_manifest.json`

元データは変更せず、マスク済みデータを別ファイルとして生成する。

## 2. 最重要原則

### P1. 正解 API リストをマスク判定に使わない

各サンプルの `value[2]` にある正解 API 名だけを探してマスクしてはならない。これは前処理が正解ラベルを参照することになり、評価時の情報リークになるためである。

すべての入力文に対して、909 候補 API から作った同一の名前・別名辞書を適用する。文中で見つかった名前がそのサンプルの正解 API かどうかは、マスク判定に影響させない。

alias 辞書の作成にもテストサンプルの正解 API リストを使わない。alias の根拠は API カタログのキー、カタログ説明文、または別の公開・固定メタデータに限定する。正解 API リストとの比較は、辞書と規則を凍結した後の監査にだけ使用する。

### P2. マスク対象は API を識別できる固有表現だけとする

API の機能、カテゴリ、通信方式、一般的な技術用語は残す。たとえば `maps`、`weather`、`images`、`photo`、`food`、`SMS`、`JSON`、`SOAP`、`API` は、それだけでは特定 API の識別名とみなさない。

### P3. 意味から API 名を推測しない

マスクは、登録された名前または別名と実際の文字列範囲が一致した場合だけ行う。「天気予報を取得するから OpenWeatherMap のことだろう」のような意味推論によるマスクは行わない。

## 3. 用語

- **canonical name**: 909 API 辞書のキー。例: `google-maps`、`openweathermap`。
- **surface alias**: 入力文に現れ得る表記。例: `Google Maps`、`OpenWeatherMap`、`Dog CEO`。
- **safe alias**: 単独で現れても API・サービスの固有名として十分に識別できる表記。
- **contextual alias**: 一般語、短い語、企業名など、API 文脈がある場合だけマスクできる表記。
- **API marker**: `API`、`APIs`、`web service`、`web services`。

## 4. 辞書の作成基準

### 4.1 canonical name からの自動生成

次の処理で機械的に alias 候補を作る。

1. Unicode を NFKC 正規化する。
2. HTML entity を復号する。
3. ハイフン、アンダースコア、連続空白を単語区切りとして扱う。
4. ASCII apostrophe、curly apostrophe、全角記号を比較用の共通表現に変換する。
5. 比較時だけ大文字・小文字を同一視する。置換位置は元の文から取得する。
6. 正規化後の各文字が元文字列のどの範囲に対応するかを保持する。HTML entity の復号や NFKC で文字数が変わっても、元文字列の正しい範囲を置換する。
7. API カタログ説明文に正式表記がある場合は、その表記を alias 候補として抽出する。
8. データ重複を表すことが確認できた末尾番号だけを除いた候補も作る。意味を持つ版番号や製品番号は除かない。除去対象は固定リストとして保存する。
9. canonical name 全体を alias とし、一部分だけを勝手に alias にしない。

例:

- `google-maps` -> `Google Maps` 相当
- `google-maps-elevation` -> `Google Maps Elevation` 相当
- `openweathermap` -> `OpenWeatherMap` 相当

`lastfm` -> `Last.fm`、`contextio` -> `Context.IO`、`shoppingcom` -> `Shopping.com`、`dog-ceo-dog` -> `Dog CEO` のように、この変換だけでは作れない表記は明示的な別名辞書に登録する。

句読点をすべて削除して全文を連結した文字列では照合しない。この方法は `Last.fm` のような表記を拾える一方、`domain` と `domains` や、隣接する別単語を誤って一致させるためである。

### 4.2 safe alias の判定

次のいずれかを満たし、かつ一般語 denylist に該当しない alias は辞書作成時に `safe` として固定する。

1. 2 語以上からなり、少なくとも 1 語が一般機能語ではない。
2. 4 文字以上の単語固有名で、一般英単語として頻出しない。
3. 大文字・記号を含む正式表記で識別できるブランド名として、別名辞書に登録されている。

例: `Google Maps`、`OpenWeatherMap`、`YouTube`、`Flickr`、`Dropbox`、`Twilio`。

「一般英単語として頻出するか」「一般機能語か」を実行時に推測してはならない。使用する一般語 denylist、一般機能語リスト、正式大小文字表記はすべてリポジトリ内の版管理されたファイルに保存する。同じ辞書バージョンでは分類結果が必ず同じでなければならない。

### 4.3 contextual alias の判定

次の表記は既定で `contextual` とする。

- 3 文字以下の表記
- 数字だけの表記
- 一般英単語と衝突する表記
- 企業・プラットフォーム全体の名前で、複数 API を指し得る表記
- canonical name の全単語が一般機能語である表記

初期 denylist には、実データで誤検出が確認された次の語を少なくとも含める。

`images`, `photo`, `food`, `numbers`, `domain`, `active`, `vast`, `sms`

`box` や `indeed` のように一般語とブランド名の両方になり得る語は、正式な大小文字・ドメイン表記を登録して区別する。例: `Box`、`Box.net`、`Indeed`。

contextual alias は、次のいずれかの API 文脈に直接接続するときだけマスクする。

- `<alias> API` / `<alias> APIs`
- `<alias>'s API` / `<alias>'s APIs`
- `<alias> web service` / `<alias> web services`
- `API from <alias>` / `API by <alias>` / `API of <alias>`
- `APIs from <alias-list>` / `APIs of <alias-list>`
- `APIs including <alias-list>` / `APIs such as <alias-list>`
- `API services: <alias-list>` / `APIs: <alias-list>`
- `<alias-list> APIs`。alias-list はカンマ、`and`、`or` で接続された確認済み alias の列に限る。

`<alias> Data API`、`<alias> Public API`、`<alias> Open Graph API` のように alias と API marker の間へ製品記述語が入る表記は、文法だけで広く許可しない。カタログで確認した完全な表面形を alias として登録する。途中に未登録の内容語を挟む場合は自動マスクせず、要確認に送る。

alias-list は同じ節内だけを対象とし、文境界を越えない。最大項目数と許可する接続記号を設定ファイルで固定する。

### 4.4 明示的な別名辞書

別名辞書は API カタログを根拠に作成し、テストサンプルの正解 API リストに応じて切り替えない。1 つの alias が複数候補 API に対応してもよい。マスクの目的は候補を一意に決めることではなく、識別名を入力から消すことである。

各 alias には、最低限 `text`、`candidate_api_names`、`alias_type`、`case_sensitive`、`source_type`、`source_reference` を持たせる。`source_type` は `catalog_key`、`catalog_description`、`external_fixed_metadata` のいずれかとし、根拠を追跡できない alias は採用しない。

実データから初期確認が必要な例:

- `Dog CEO` -> `dog-ceo-dog`
- `Last.fm` -> `lastfm`
- `Context.IO` -> `contextio`
- `Microsoft's Emotion` / `Microsoft Emotion` -> `microsoft-cognitive-services-emotion`
- `Spotify` -> `spotify-web`
- `Bol.com` -> `bolcom-open`
- `Shopping.com` -> `shoppingcom`
- `New York Times Bestsellers` -> `new-york-times-best-sellers`
- `Google Search` -> `google-custom-search` の可能性があるため、カタログで確認して登録
- `eBay`、`Amazon` のように複数 API を指す provider alias は `contextual` として登録

カタログで関係を確認できない表記は、正解リストと一致して見えても登録しない。

`Emotion API` のように機能名だけになる短縮形は alias にしない。要求内容を表す一般語まで消してしまうため、provider を含む確認済み表面形だけを登録する。また、現データの `ScentSee API` と正解 `scalarium` のようにカタログで関係を確認できない組は、正解ラベルを根拠とした alias にせず、データ注釈の要確認項目として分離する。

## 5. 文中での一致規則

判定順は固定する。

1. 入力文を正規化するが、元文字列の開始・終了位置を保持する。
2. safe alias の完全一致を探す。
3. contextual alias を探し、API 文脈規則を満たすものだけ採用する。
4. 単語境界を必須とする。別単語の部分文字列は採用しない。
5. 一致範囲が重なる場合は、最長一致を優先する。
6. 同じ長さなら開始位置が早い一致を優先する。
7. 同一範囲が複数 API に対応しても、置換は 1 回だけ行う。
8. 採用された範囲を後ろから置換し、文字位置のずれを防ぐ。

例:

- `Google Maps Elevation` では `Google Maps Elevation` を 1 範囲としてマスクし、内側の `Google Maps` を別に置換しない。
- `Box.net` が登録されている場合は `Box` より `Box.net` を優先する。

## 6. 置換規則

すべての API 名を同じ文字列 `[API_NAME]` に置き換える。API ごとの ID、連番、元の文字数、カテゴリは埋め込まない。

`[API]` は TF-IDF の標準的な token pattern では `API` として読まれ、直後に残す `API` と重複しやすいため使用しない。実装前に、TF-IDF、BERT、SBERT、Cross-Encoder で `[API_NAME]` の tokenization を記録し、全方式で同じ置換済み文字列を入力する。

名前の後ろにある `API` / `APIs` / `web service` は残す。これらは特定 API の名前ではなく、要求文の意味を表すためである。

例:

- `uses the OpenWeatherMap API` -> `uses the [API_NAME] API`
- `including YouTube, Facebook and Vimeo` -> `including [API_NAME], [API_NAME] and [API_NAME]`
- `Dog CEO's API` -> `[API_NAME]'s API`
- `product images` -> 変更しない
- `SMS software` -> 変更しない

置換済みの `[API_NAME]` は再実行時に変更しない。処理は冪等でなければならない。1 つの名前につき 1 つの placeholder を残すため、API 名の個数情報は保持される。この仕様を実験条件として明記する。

## 7. 境界事例の扱い

### マスクする

- 候補辞書の固有 API 名または確認済み alias
- API 名と同名のサービス・製品名が文中に出た場合
- 正解ではない候補 API 名が文中に出た場合
- URL やドメイン表記に含まれる確認済み固有名。例: `Box.net`

API とサービスの名称が同じ場合に利用文脈を判別しようとすると、人手判断が増え再現性が落ちる。そのため、safe alias であれば文脈にかかわらずマスクする。

### マスクしない

- `API`、`web service` という種類名だけ
- 機能、カテゴリ、データ型、プロトコル名
- API カタログにないライブラリ、フレームワーク、製品名
- 意味から推測しただけで、文字列一致がない API
- `Amazon Echo` のようなプラットフォーム名。ただし確認済み API alias として辞書登録された場合を除く

### 要確認に送る

- contextual alias だが、定義済み API 文脈から外れている
- カタログ名と入力表記の対応が不明確
- 1 つの一般的な provider 名が多数の API を指し、API marker から離れている
- 文字化けにより一致範囲が不確かなもの

要確認項目を担当者が直接 JSON 編集してはならない。採用する場合は alias 辞書または denylist を更新し、336 件すべてを再生成する。

## 8. 現データでの予備監査結果

canonical name のハイフンを空白にし、小文字化しただけの粗い完全一致を 336 件に適用した結果は次のとおりだった。

- 候補 API 名が 1 件以上一致したサンプル: 168 / 336
- 一致した候補名の延べ数: 264
- 正解 API 名との一致を監査目的で事後比較したサンプル: 150 / 336
- 正解 API 名との一致延べ数: 212 / 717

正解 API はマスク判定には使用せず、上の後半 2 数値は規則の監査にだけ用いた。

粗い一致では `images` 13 件、`photo` 8 件、`food` 4 件、`sms` 4 件などの一般語も拾った。一方、`Dog CEO`、`Last.fm`、`Amazon API` など canonical name と表面形が異なる名前は漏れる。このため、単純な全置換ではなく safe/contextual/alias の 3 区分が必要である。

追加監査では、API marker を含むサンプルは 55 / 336 件だった。marker 周辺には次のような、初版の直接接続規則だけでは拾えない構文が存在した。

- `Facebook and Twitter APIs` のような複数名で 1 つの `APIs` を共有する表記
- `APIs, including ...`、`APIs from ...`、`API services: ...` のような後置リスト
- `Facebook Open Graph API`、`YouTube Data API`、`Spotify's Public API` のような途中語
- `Last.fm`、`Context.IO`、`Shopping.com`、`Bol.com` のような句読点・ドメイン表記
- `Microsoft’s` のような curly apostrophe

また、句読点と空白をすべて除去する粗い追加照合では 13 件を追加検出したが、`domain` と `domains` の誤一致も発生した。この照合方法は本処理には採用せず、別名候補を作る監査にだけ使う。

## 9. 出力と監査ログ

生成時に次を保存する。

1. マスク済み 336 件の JSON
2. 使用した alias 辞書と denylist
3. 1 一致 1 行の監査ログ
4. 要確認一覧
5. 実行設定と辞書バージョン
6. 入力元 JSON と出力 JSON の SHA-256

監査ログには最低限、次を含める。

- `sample_id`
- `original_span`
- `start`, `end`
- `matched_alias`
- `candidate_api_names`
- `rule_id`
- `alias_type` (`safe` / `contextual`)
- `replacement`
- `alias_source_type`
- `alias_source_reference`

## 10. 生成後の必須検証

次をすべて満たした場合だけ評価に使用する。

1. サンプル数が 336 のままである。
2. サンプルキーと順序が元データと同じである。
3. 各 value の説明文 `value[1]` 以外が完全一致する。
4. safe alias と、文脈条件を満たす contextual alias の未マスク残存が 0 件である。
5. 同じ入力と辞書から同じ出力・監査ログが得られる。
6. 生成処理を 2 回適用しても結果が変わらない。
7. マスク件数、対象サンプル数、rule_id 別件数を報告する。
8. 要確認一覧を固定した後は、人手で出力 JSON を修正せず再生成する。
9. placeholder 以外の説明文が元文字列と完全一致する。
10. API catalog、alias 辞書、denylist、設定ファイルのハッシュを結果へ記録する。

## 11. 採用する運用方針

初回は高精度優先とする。

- safe alias: 自動マスク
- contextual alias + 定義済み API 文脈: 自動マスク
- denylist: 自動的に非マスク
- その他: 要確認

見逃しをゼロにしようとして一般語まで消すと、モデルに必要な要求内容そのものが失われる。したがって、別名漏れは監査ログから辞書を更新して減らし、一般語の一括マスクは行わない。

## 12. 評価データへの反映範囲

元の 336 件だけをマスクし、階層モデルや条件別評価が参照する複製 JSON を未マスクのまま残してはならない。比較対象によって入力文が変わるためである。

運用は次のどちらかに統一する。

1. すべての評価プログラムが 1 つの canonical masked JSON を直接読む。
2. canonical masked JSON から階層別・条件別 JSON を再生成する。

派生 JSON について、同じ `sample_id` の説明文が canonical masked JSON と一致することを検証する。既存の未マスク派生 JSON を直接編集してはならない。

評価は少なくとも次の対応を保つ。

- unmasked 336 件での従来結果
- masked 336 件での同一モデル・同一候補集合・同一評価指標による結果

この 2 条件の差を API 名リークへの依存度として報告する。学習データをマスクしない場合、masked test の結果は「API 名を除いたテストに対する頑健性評価」と明記する。学習データもマスクする実験は別条件とし、結果を混在させない。
