# TF-IDF OvR Analysis Materials

This packet is for a separate LLM. It is intentionally evidence-oriented: use it to form hypotheses, but do not treat the bullet points as final conclusions.

## Files

- `summary.json`: compact machine-readable overview.
- `records_comparison.csv`: one row per test record, with predictions/metrics from each run.
- `label_diagnostics.csv`: per-API train/test frequency, TP/FP/FN, score averages, and top coefficient terms.
- `feature_terms_by_label.csv`: top positive/negative TF-IDF coefficient terms for every primary-run label.
- `pairwise_primary_vs_other_runs.csv`: record-level and true-positive overlap against the primary run.
- `case_examples.json`: curated full-text examples for success, failure, near miss, and method-difference analysis.

## Primary Run

- Primary run: `tfidf_ovr_logistic_unigram`
- Records: 336
- Macro/sample precision: 0.6292
- Macro/sample recall: 0.5549
- Macro/sample F1: 0.5538
- Micro precision: 0.7434
- Micro recall: 0.4324
- Micro F1: 0.5467
- Average predicted APIs: 1.2411
- Zero-prediction records: 72
- Records with unseen true labels: 54
- Unseen true-label mentions: 96

## Method Metrics

| run | avg_pred | precision | recall | F1 | micro_P | micro_R | micro_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross_encoder_top1 | 1.0000 | 0.7708 | 0.5072 | 0.5759 | 0.7708 | 0.3612 | 0.4919 |
| tfidf_ovr_logistic_unigram | 1.2411 | 0.6292 | 0.5549 | 0.5538 | 0.7434 | 0.4324 | 0.5467 |
| tfidf_ovr_logistic_bigram | 1.0595 | 0.6405 | 0.5237 | 0.5422 | 0.8146 | 0.4045 | 0.5405 |
| tfidf_ovr_svm_unigram | 0.9940 | 0.6263 | 0.5092 | 0.5305 | 0.8204 | 0.3821 | 0.5214 |
| sbert_ovr_logistic | 3.0506 | 0.3789 | 0.6120 | 0.4252 | 0.3366 | 0.4812 | 0.3961 |
| cross_encoder_top5 | 5.0000 | 0.2732 | 0.7622 | 0.3752 | 0.2732 | 0.6402 | 0.3830 |
| tfidf_cosine_top1 | 1.0000 | 0.3571 | 0.2311 | 0.2662 | 0.3571 | 0.1674 | 0.2279 |
| cross_encoder_threshold05 | 11.4286 | 0.1570 | 0.8428 | 0.2506 | 0.1393 | 0.7462 | 0.2348 |
| tfidf_cosine_top5 | 5.0000 | 0.1506 | 0.4418 | 0.2109 | 0.1506 | 0.3529 | 0.2111 |
| sbert_mpnet_top1 | 1.0000 | 0.2411 | 0.1582 | 0.1812 | 0.2411 | 0.1130 | 0.1538 |
| sbert_mpnet_top5 | 5.0000 | 0.1214 | 0.3725 | 0.1726 | 0.1214 | 0.2845 | 0.1702 |
| tfidf_cosine_top10 | 10.0000 | 0.0964 | 0.5387 | 0.1557 | 0.0964 | 0.4519 | 0.1589 |

## Pairwise Against Primary

| comparison | primary higher F1 records | primary lower F1 records | equal F1 records | mean F1 delta | primary unique TP | comparison unique TP |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf_ovr_logistic_bigram | 23 | 30 | 283 | 0.0116 | 25 | 5 |
| tfidf_ovr_svm_unigram | 37 | 24 | 275 | 0.0233 | 38 | 2 |
| sbert_ovr_logistic | 176 | 59 | 101 | 0.1286 | 38 | 73 |
| cross_encoder_top1 | 83 | 72 | 181 | -0.0221 | 106 | 55 |
| cross_encoder_top5 | 207 | 98 | 31 | 0.1786 | 17 | 166 |
| cross_encoder_threshold05 | 230 | 91 | 15 | 0.3033 | 4 | 229 |
| tfidf_cosine_top1 | 181 | 41 | 114 | 0.2876 | 230 | 40 |
| tfidf_cosine_top5 | 234 | 43 | 59 | 0.3429 | 145 | 88 |
| tfidf_cosine_top10 | 237 | 55 | 44 | 0.3981 | 110 | 124 |
| sbert_mpnet_top1 | 198 | 34 | 104 | 0.3726 | 269 | 40 |
| sbert_mpnet_top5 | 236 | 44 | 56 | 0.3812 | 193 | 87 |

## Suggested Questions For The Analysis LLM

1. Does the primary run's strength come mainly from high precision, reasonable recall, or a favorable average recommendation count?
2. Which labels are easy for unigram TF-IDF OvR, and do their top coefficient terms look like direct lexical matches to mashup descriptions?
3. Where does unigram logistic beat TF-IDF cosine ranking, and is the difference mostly from supervised label priors or from filtering noisy cosine matches?
4. Where does Cross Encoder top1 beat the primary run, especially on records where the primary run predicts nothing?
5. How much of the remaining error is unavoidable for OvR because true labels are unseen in train?
6. Does adding bigrams help precision at the cost of recall, or mostly add sparsity?

## Case Example Index

### primary_exact_or_full_recall_examples
- idx=99 name=onnli.com-job-search-and-realtime-job-listing F1=1.000 true=['indeed', 'google-maps', 'twitter'] pred=['indeed', 'google-maps', 'twitter']
- idx=30 name=starterpad F1=1.000 true=['facebook', 'twitter'] pred=['twitter', 'facebook']
- idx=43 name=map-deprivation-wales F1=1.000 true=['google-maps', 'google-fusion-tables'] pred=['google-fusion-tables', 'google-maps']

### primary_beats_tfidf_cosine_top5_examples
- idx=21 name=appbuilder-forms-docusign F1=1.000 true=['docusign-enterprise'] pred=['docusign-enterprise']
- idx=24 name=tirupatidarshan F1=1.000 true=['i2space'] pred=['i2space']
- idx=76 name=twitatoo F1=1.000 true=['twitter'] pred=['twitter']

### primary_beats_sbert_ovr_examples
- idx=21 name=appbuilder-forms-docusign F1=1.000 true=['docusign-enterprise'] pred=['docusign-enterprise']
- idx=33 name=sos-garage F1=1.000 true=['allogarage'] pred=['allogarage']
- idx=144 name=listen-and-write.com F1=1.000 true=['youtube'] pred=['youtube']

### primary_worse_than_cross_encoder_top1_examples
- idx=3 name=poluxfr F1=0.000 true=['dog-ceo-dog'] pred=[]
- idx=13 name=imwallet F1=0.000 true=['dropbox'] pred=[]
- idx=23 name=targeted-engagement F1=0.000 true=['nationbuilder'] pred=[]

### unigram_logistic_better_than_bigram_logistic_examples
- idx=8 name=weathercheck F1=1.000 true=['openweathermap'] pred=['openweathermap']
- idx=21 name=appbuilder-forms-docusign F1=1.000 true=['docusign-enterprise'] pred=['docusign-enterprise']
- idx=111 name=zappos-peer-pressure F1=1.000 true=['zappos'] pred=['zappos']

### zero_prediction_examples
- idx=112 name=little-corner F1=0.000 true=['flickr', 'delicious-0', 'lastfm', 'twitter', 'tumblr', 'soundcloud', 'github', 'google-analytics-managment', 'foursquare', 'instagram-graph', 'untappd', 'gravatar'] pred=[]
- idx=4 name=dynamic-insights F1=0.000 true=['xero', 'quickbooks-payments', 'myob-accountright-live', 'myob-essentials-accounting', 'tradegecko', 'neto', 'kounta', 'servicem8', 'workflowmax', 'google-analytics-core-reporting'] pred=[]
- idx=2 name=voice-apps F1=0.000 true=['amazon-alexa-skill-voice-service', 'alexa-skill-management', 'amazon-alexa-proactive-events', 'amazon-alexa-smart-home-skills', 'google-assistant', 'dialogflow-standard-webhooks', 'nlp-tools'] pred=[]

### unseen_label_failure_examples
- idx=4 name=dynamic-insights F1=0.000 true=['xero', 'quickbooks-payments', 'myob-accountright-live', 'myob-essentials-accounting', 'tradegecko', 'neto', 'kounta', 'servicem8', 'workflowmax', 'google-analytics-core-reporting'] pred=[]
- idx=2 name=voice-apps F1=0.000 true=['amazon-alexa-skill-voice-service', 'alexa-skill-management', 'amazon-alexa-proactive-events', 'amazon-alexa-smart-home-skills', 'google-assistant', 'dialogflow-standard-webhooks', 'nlp-tools'] pred=[]
- idx=11 name=open-data-companion-odc F1=0.000 true=['ckan', 'ckan-ireland', 'brazil-ckan', 'ckan-czech-republic', 'ckan-italy', 'ckan-norway'] pred=[]

### near_miss_false_negative_examples
- idx=159 name=picwash-photo-retouch-service F1=0.000 true=['flickr', 'smugmug', 'facebook', 'google-picasa', 'photobucket'] pred=[]
- idx=20 name=bookfinder F1=0.000 true=['google-books', 'open-library-books', 'amazon-product-advertising', 'ebay-finding'] pred=['goodreads']
- idx=134 name=musikki F1=0.400 true=['flickr', 'amazon-product-advertising', 'google-maps', 'youtube', 'lastfm', 'twitter', 'musikki-music'] pred=['youtube', 'musicbrainz', 'flickr']

### high_confidence_false_positive_examples
- idx=255 name=touropia F1=0.750 true=['flickr', 'google-maps', 'youtube'] pred=['google-ad-manager', 'google-adsense', 'google-maps', 'flickr', 'youtube']
- idx=31 name=premiumshared.com F1=0.500 true=['4shared', 'lastfm'] pred=['exfm', 'lastfm']
- idx=136 name=camelcamelcamel F1=0.500 true=['twitter', 'amazon-marketplace-web-service'] pred=['amazon-product-advertising', 'twitter']
