import yaml

data = {
    'path': 'SDO_Sunspots_Extracted',
    'train': 'train',
    'val': 'valid',
    'test': 'test',
    'nc': 1,
    'names': ['Sunspots']
}

with open('data.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print("data.yaml written successfully!")

with open('data.yaml', 'r') as f:
    print(f.read())