import json, os

all_files = set()
for json_file in os.listdir("data/sugarcrepe/"):
    with open(f"data/sugarcrepe/{json_file}") as f:
        data = json.load(f)
    for key in data:
        all_files.add(data[key]["filename"])

print(f"Unique SugarCrepe images: {len(all_files)}")