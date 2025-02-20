import json


def parse_output(output_base):
    
    output = output_base.replace('\n', '').replace("```", '').replace("python", '').replace(">>>", '').strip()
    final_output = []
    try:
        output = output.split("[")[1].split("]")[0]
        output = output.split("',") 
        for op in output:
            if '",' in op:
                op = op.split('",')
                final_output+=op
            else:
                final_output.append(op)
        final_output = [o.replace('"', '').replace("'", "").strip() for o in final_output]
    except:
        print("ERROR 1: ", output_base)

    if len(final_output) != 3:
        print("ERROR: ", output_base, len(final_output), final_output)

    return final_output

with open("../../../data/VAR_Data_caption.json") as f:
    data = json.load(f)

with open("out/gpt4o-all-gen-outputs-multi.jsonl", "r") as f:
    output = f.readlines()
    output = [json.loads(o) for o in output]

for item in data:

    out_1 = f"request-{item['set_id']}-{item['id']}-t1"
    out_2 = f"request-{item['set_id']}-{item['id']}-t2"
    out_3 = f"request-{item['set_id']}-{item['id']}-t3"

    for o in output:
        if o["custom_id"] == out_1:
            item["task1_responses"] = parse_output(o["response"]['body']["choices"][0]["message"]['content'])
        elif o["custom_id"] == out_2:
            item["task2_responses"] = parse_output(o["response"]['body']["choices"][0]["message"]['content'])
        elif o["custom_id"] == out_3:
            item["task3_responses"] = [o["response"]['body']["choices"][0]["message"]['content']]

with open("../../../results/VAR_Data_gpt4o.json", "w") as f:
    json.dump(data, f, indent=4)