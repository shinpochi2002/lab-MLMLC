import json

# クラスター情報を category_cluster.json から読み込む
with open('/home/nakagawa/mashup_data/mashup_data_new.json', 'r') as f:
    mashupdata = json.load(f)

# api_new_data.json のデータを読み込む
with open('/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_test_data.json', 'r') as f:
    incnomashdata = json.load(f)

mnamelist = []

for mname,minfo in mashupdata.items():
    mnamelist.append(mname)

print(mnamelist)

newdict = {}

incnomashdata = [api for api in incnomashdata if api["name"] in mnamelist]

new_id = 1
for key in incnomashdata:
    key["id"] = new_id  # ID番号は各エントリの最後にあることを想定
    new_id += 1

# 新しいファイル名で保存
with open("collaboration_test_data.json", "w", encoding="utf-8") as new_mashup_file:
    json.dump(incnomashdata, new_mashup_file, indent=4, ensure_ascii=False)

filtered_correctly = all(api["name"] in mnamelist for api in incnomashdata)
print(f"All filtered APIs exist in mnamelist: {filtered_correctly}")
