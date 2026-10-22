# API負例3比率の接続確認

**PASS：152分類器の1:1・1:3・1:5保存データを読み込み、既存学習処理に接続できる。WARNING：比率不足が3条件で存在する。GPU学習による比率比較は未実施。**

分類器ごとに保存全FoldのValidation ID順・集合が3条件で同一、正例Train集合が不変、負例が包含関係を満たすことを確認した。比率別のTrain件数、実比率、負例不足、group丸め超過はnegative_ratio_inventory.csvへ全件記録した。データ生成・再抽出・再分割を行っていない。API説明文のTrain固定方針も保持した。

## 代表のfold_01

| 分類器（api_classification/ 以下） | 要求比率 | Train正 | Train負 | 実比率 | Val |
| --- | --- | --- | --- | --- | --- |
| category_booking | 1 | 1 | 1 | 1:1.0000 | 792 |
| category_booking | 3 | 1 | 3 | 1:3.0000 | 792 |
| category_booking | 5 | 1 | 5 | 1:5.0000 | 792 |
| category_content | 1 | 45 | 45 | 1:1.0000 | 788 |
| category_content | 3 | 45 | 135 | 1:3.0000 | 788 |
| category_content | 5 | 45 | 225 | 1:5.0000 | 788 |
| category_internet_things | 1 | 7 | 7 | 1:1.0000 | 1316 |
| category_internet_things | 3 | 7 | 21 | 1:3.0000 | 1316 |
| category_internet_things | 5 | 7 | 35 | 1:5.0000 | 1316 |
| category_mapping | 1 | 1386 | 1386 | 1:1.0000 | 791 |
| category_mapping | 3 | 1386 | 1821 | 1:1.3139 | 791 |
| category_mapping | 5 | 1386 | 1821 | 1:1.3139 | 791 |

bookingは正例1件のため3条件が保存されていても比率・lrを比較しない。固定設定としてratio1・lr2e-5・10epochを適用する既存経路である。ほかの97選択対象は3条件×2lr×全保存Foldのchoose_configへ接続する。3条件すべての本格学習は今回行っていない。

## 負例不足の全件

| 分類器 | 比率 | 用途 | 正例 | 実負例 | 要求負例 | 実比率 |
| --- | --- | --- | --- | --- | --- | --- |
| api_classification/category_mapping | 3 | fold_01 | 1386 | 1821 | 4158 | 1:1.3139 |
| api_classification/category_mapping | 3 | final_refit | 1720 | 2278 | 5160 | 1:1.3244 |
| api_classification/category_mapping | 5 | fold_01 | 1386 | 1821 | 6930 | 1:1.3139 |
| api_classification/category_mapping | 5 | final_refit | 1720 | 2278 | 8600 | 1:1.3244 |
| api_classification/category_social | 5 | fold_01 | 584 | 2621 | 2920 | 1:4.4880 |
| api_classification/category_social | 5 | final_refit | 718 | 3276 | 3590 | 1:4.5627 |

不足は1:1が0分類器、1:3がmappingの1分類器、1:5がmappingとsocialの2分類器。各条件で選択用fold_01とfinal_refitの両方に不足がある。利用可能負例を上限としており、要求比率を満たすための複製・追加生成はしていない。

mappingではratio3とratio5のTrain集合が同じ上限に達しており、2条件を比較しても「異なる実比率の効果」の比較にはならない。一方、現実装の条件選択はこの不足を記録した条件を候補として扱える。ratio名称だけで実比率が3/5に達した実験と報告しないこと。修正ではなく、実比率・不足とともに結果を記述する必要がある。
