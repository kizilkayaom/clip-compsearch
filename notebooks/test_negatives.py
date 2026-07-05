import json
from src.data.negatives import nlp, swap_objects, swap_attributes

print(swap_objects(nlp, "The cat chased the dog."))

with open("data/sugarcrepe/swap_obj.json", 'r') as f:
    data = json.load(f)

for key in ["0", "1", "2", "3"]:
    cap = data[key]["caption"]
    print(f"Original:      {cap}")
    print(f"Swapped:       {swap_objects(nlp, cap)}")
    print(f"SugarCrepe:    {data[key]['negative_caption']}")
    print()

print(swap_attributes(nlp, "A red mug sits on a blue book."))

with open("data/sugarcrepe/swap_att.json", 'r') as f:
    att_data = json.load(f)

for key in ["0", "1", "2"]:
    cap = att_data[key]["caption"]
    print(f"Original:      {cap}")
    print(f"Swapped:       {swap_attributes(nlp, cap)}")
    print(f"SugarCrepe:    {att_data[key]['negative_caption']}")
    print()