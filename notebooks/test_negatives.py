import json
from src.data.negatives import nlp, swap_objects, swap_attributes
from src.data.negatives import nli_filter


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

print("Pair 1:", nli_filter(
    "A woman prepares a pizza while a man watches.",
    "A man prepares a pizza while a woman watches."
))

print("Pair 2:", nli_filter(
    "A cat sits on the mat.",
    "A cat sits on the mat."
))

print("Pair 3:", nli_filter(
    "A cat sits on its hind legs, and swats at the plant.",
    "A plant sits on its hind legs, and swats at the cat."
))

print("Pair 4:", nli_filter(
    "Two giraffe and a zebra are standing in a field.",
    "Two giraffe and a zebra are standing in a field."
))