import json

with open("/h/adityac/projects/VAR/data/def_list_all_subset.json") as f:
    mcq_data = json.load(f)

with open("out/gpt4o-subset-def-outputs.jsonl", "r") as f:
    output = f.readlines()
    output = [json.loads(o) for o in output]

for item in mcq_data:

    out_1 = f"request-{item['set_id']}-{item['id']}-{item['def_id']}"

    for o in output:
        if o["custom_id"] == out_1:
            item["predicted"] = o["response"]['body']["choices"][0]["message"]['content']

with open("../../../results/def_list_all_subset_gpt4o.json", "w") as f:
    json.dump(mcq_data, f, indent=4)