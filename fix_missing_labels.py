import pandas as pd
import os

for split in ['train', 'valid', 'test']:
    folder = f'SDO_Sunspots_Extracted/{split}'
    df = pd.read_csv(f'{folder}/_annotations.csv')
    
    csv_filenames = set(df['filename'].unique())
    disk_images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    
    created = 0
    for img in disk_images:
        if img not in csv_filenames:
            # this image has no annotations — create empty label file
            label_name = os.path.splitext(img)[0] + '.txt'
            label_path = os.path.join(folder, label_name)
            open(label_path, 'w').close()  # empty file
            print(f"  Created empty label: {label_name}")
            created += 1
    
    print(f"{split}: created {created} empty label files\n")

print("Done!")