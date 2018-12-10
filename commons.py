
import json



def dump_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_json(file_path):
    with open(file_path, 'r') as f:
        input_meta = json.load(f)
    return input_meta