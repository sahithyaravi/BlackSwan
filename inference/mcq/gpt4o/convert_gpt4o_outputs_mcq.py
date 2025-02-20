import json

with open("/h/adityac/projects/VAR/data/mcq_list_all_gpt.json") as f:
    mcq_data = json.load(f)

with open("out/gpt4o-all-mcq-output.jsonl", "r") as f:
    output = f.readlines()
    output = [json.loads(o) for o in output]

for item in mcq_data:

    out_1 = f"request-{item['set_id']}-{item['id']}-{item['mcq_id']}"

    for o in output:
        if o["custom_id"] == out_1:
            item["predicted"] = o["response"]['body']["choices"][0]["message"]['content']

with open("../../../results/mcq_list_all_gpt_gpt4o.json", "w") as f:
    json.dump(mcq_data, f, indent=4)