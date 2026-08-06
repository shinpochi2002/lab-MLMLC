# TF-IDF OvR vs Hierarchy 12-4-3 Analysis Materials

This packet is for a separate LLM. It collects evidence and comparison tables without making the final interpretation.

## Files

- `summary.json`: compact machine-readable overview.
- `records_comparison.csv`: one row per matched test record.
- `label_comparison.csv`: per-API TP/FP/FN and prediction frequency for both methods.
- `hierarchy_condition_summary.csv`: 12-4-3 metrics on shorter/longer/candidate-size subsets.
- `case_examples.json`: curated examples for success, failure, recall-vs-precision, and hierarchy layer errors.

## Overall Metrics On 336 Matched Records

| method | avg_pred | precision | recall | F1 | micro_P | micro_R | micro_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf_ovr_logistic_unigram | 1.2411 | 0.6292 | 0.5549 | 0.5538 | 0.7434 | 0.4324 | 0.5467 |
| hierarchy_layer3_12_4_3_shorter+longer | 8.1845 | 0.5127 | 0.7646 | 0.5383 | 0.1767 | 0.6778 | 0.2804 |

## Pairwise Record Comparison

- TF-IDF higher F1 records: 109
- Hierarchy higher F1 records: 105
- Equal F1 records: 122
- Mean TF-IDF minus hierarchy F1: 0.0156
- TF-IDF unique true-positive mentions: 42
- Hierarchy unique true-positive mentions: 218

## Hierarchy 12-4-3 Condition Metrics

| condition | n | avg_pred | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- |
| fewer_candidates | 195 | 5.3026 | 0.5955 | 0.7977 | 0.6272 |
| longer_desc | 165 | 9.8121 | 0.4416 | 0.7146 | 0.4803 |
| longer_desc_fewer_candidates | 88 | 5.3182 | 0.5155 | 0.6929 | 0.5393 |
| longer_desc_more_candidates | 77 | 14.9481 | 0.3571 | 0.7394 | 0.4128 |
| more_candidates | 141 | 12.1702 | 0.3981 | 0.7187 | 0.4153 |
| shorter_desc | 171 | 6.6140 | 0.5813 | 0.8128 | 0.5943 |
| shorter_desc_fewer_candidates | 107 | 5.2897 | 0.6614 | 0.8840 | 0.6994 |
| shorter_desc_more_candidates | 64 | 8.8281 | 0.4473 | 0.6938 | 0.4184 |

## Suggested Questions For The Analysis LLM

1. Does hierarchy 12-4-3 mainly improve recall by expanding through predicted categories, and how much precision does that cost?
2. Which examples show the hierarchy selecting the right upper categories but over-predicting leaf APIs?
3. Which examples show TF-IDF OvR being more selective and therefore better despite missing some true APIs?
4. Which API labels are rescued by hierarchy but impossible or hard for TF-IDF OvR because they are unseen or low-frequency in training?
5. Are 12-4-3 errors dominated by layer/category mistakes or by noisy leaf API expansion after correct category prediction?

## Case Example Index

### tfidf_ovr_better_than_hierarchy_examples
- idx=21 name=appbuilder-forms-docusign tfidf_f1=1.000 hier_f1=0.000 true=['docusign-enterprise']
- idx=24 name=tirupatidarshan tfidf_f1=1.000 hier_f1=0.000 true=['i2space']
- idx=33 name=sos-garage tfidf_f1=1.000 hier_f1=0.000 true=['allogarage']

### hierarchy_better_than_tfidf_ovr_examples
- idx=28 name=health-month-game tfidf_f1=0.000 hier_f1=1.000 true=['fitbit']
- idx=37 name=website-screenshot-generator tfidf_f1=0.000 hier_f1=1.000 true=['page2images']
- idx=44 name=your-taxi-meter tfidf_f1=0.000 hier_f1=1.000 true=['google-maps']

### hierarchy_recall_gain_precision_loss_examples
- idx=199 name=omom-online-members-map tfidf_f1=0.250 hier_f1=0.264 true=['google-maps', 'ning', 'facebook', 'hi5', 'orkut', 'bebo', 'myspace']
- idx=92 name=2lingual-bing-search tfidf_f1=0.333 hier_f1=0.267 true=['microsoft-bing', 'microsoft-translator', 'bing', 'bing-translator']
- idx=95 name=dgmapper-disc-golf-tournament-map-and-calendar tfidf_f1=0.400 hier_f1=0.333 true=['google-maps', 'noaa-national-weather-service-nws-ndfd-soap', 'google-maps-elevation', 'google-geocoding']

### hierarchy_layer_error_examples
- idx=2 name=voice-apps tfidf_f1=0.000 hier_f1=0.083 true=['amazon-alexa-skill-voice-service', 'alexa-skill-management', 'amazon-alexa-proactive-events', 'amazon-alexa-smart-home-skills', 'google-assistant', 'dialogflow-standard-webhooks', 'nlp-tools']
- idx=3 name=poluxfr tfidf_f1=0.000 hier_f1=0.000 true=['dog-ceo-dog']
- idx=5 name=virtual-classroom-api-integration tfidf_f1=0.000 hier_f1=0.000 true=['wiziq-virtual-classroom', 'moodle']

### hierarchy_layer_correct_but_leaf_noisy_examples
- idx=7 name=trip-republic tfidf_f1=0.000 hier_f1=0.000 true=['viator', 'bookingcom-content', 'skyscanner-live-pricing']
- idx=15 name=jobjolly tfidf_f1=0.000 hier_f1=0.000 true=['careerbuilder']
- idx=151 name=city-warfare tfidf_f1=0.667 hier_f1=0.000 true=['foursquare']

### tfidf_zero_prediction_hierarchy_hits_examples
- idx=28 name=health-month-game tfidf_f1=0.000 hier_f1=1.000 true=['fitbit']
- idx=37 name=website-screenshot-generator tfidf_f1=0.000 hier_f1=1.000 true=['page2images']
- idx=44 name=your-taxi-meter tfidf_f1=0.000 hier_f1=1.000 true=['google-maps']

### both_good_examples
- idx=30 name=starterpad tfidf_f1=1.000 hier_f1=1.000 true=['facebook', 'twitter']
- idx=45 name=university-michigan-campus-map tfidf_f1=1.000 hier_f1=1.000 true=['google-maps']
- idx=47 name=university-south-florida-campus-map tfidf_f1=1.000 hier_f1=1.000 true=['google-maps']

### both_bad_examples
- idx=11 name=open-data-companion-odc tfidf_f1=0.000 hier_f1=0.000 true=['ckan', 'ckan-ireland', 'brazil-ckan', 'ckan-czech-republic', 'ckan-italy', 'ckan-norway']
- idx=18 name=usersthink tfidf_f1=0.000 hier_f1=0.000 true=['stripe', 'gmail', 'gimmie', 'unbounce', 'launchrock']
- idx=226 name=learnit-lists tfidf_f1=0.000 hier_f1=0.000 true=['google-gadgets', 'facebook', 'opensocial', 'orkut', 'myspace']

### unseen_tfidf_label_helped_by_hierarchy_examples
- idx=4 name=dynamic-insights tfidf_f1=0.000 hier_f1=0.310 true=['xero', 'quickbooks-payments', 'myob-accountright-live', 'myob-essentials-accounting', 'tradegecko', 'neto', 'kounta', 'servicem8', 'workflowmax', 'google-analytics-core-reporting']
- idx=52 name=fastcashier-wallet tfidf_f1=0.000 hier_f1=0.214 true=['payment-vision-pay', 'payments-gateway', 'paymentwall']
- idx=57 name=pizza-coupons tfidf_f1=0.000 hier_f1=0.194 true=['adility', 'retailmenotcom-community-ideas', 'dealsurf']
