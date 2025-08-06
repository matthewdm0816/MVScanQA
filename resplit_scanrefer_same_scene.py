import json
import random
from collections import defaultdict

# Load the original ScanRefer train JSON file
svc_path = "../SVC/scanrefer"
input_train_json_file = f"{svc_path}/ScanRefer_filtered_train.json"
input_val_json_file = f"{svc_path}/ScanRefer_filtered_val.json"

output_train_json_file = f"{svc_path}/ScanRefer_resplit_small_train.json"
output_val_json_file = f"{svc_path}/ScanRefer_resplit_small_val.json"
output_test_json_file = f"{svc_path}/ScanRefer_resplit_small_test.json"

# Control parameters
num_scenes = 20  # Number of scenes to use
train_ratio = 0.9  # Ratio of data to use for training
val_ratio = 0.2

# Load data
with open(input_train_json_file, 'r') as f:
    data = json.load(f)

with open(input_val_json_file, 'r') as f:
    val_data = json.load(f)

# Group data by scene_id
scene_data = defaultdict(list)
for item in data:
    scene_data[item['scene_id']].append(item)

# Select a subset of scenes
random.seed(42)
selected_scenes = random.sample(list(scene_data.keys()), num_scenes)

# Prepare new dataset
new_data = []
for scene in selected_scenes:
    new_data.extend(scene_data[scene])

# Shuffle the new dataset
random.shuffle(new_data)

# Split the data
split_index = int(train_ratio * len(new_data))
new_train_data = new_data[:split_index]
new_val_data = new_data[split_index:]

# Save the new train and validation datasets to JSON files
with open(output_train_json_file, 'w') as train_file:
    json.dump(new_train_data, train_file, indent=2)

with open(output_val_json_file, 'w') as val_file:
    json.dump(new_val_data, val_file, indent=2)

print(f"New small train data saved to {output_train_json_file}")
print(f"New small validation data saved to {output_val_json_file}")
print(f"Number of scenes used: {num_scenes}")
print(f"Total data points: {len(new_data)}")
print(f"Train data points: {len(new_train_data)}")
print(f"Validation data points: {len(new_val_data)}")
