import pandas as pd
import os

def convert_split(split):
    csv_path = f'SDO_Sunspots_Extracted/{split}/_annotations.csv'
    df = pd.read_csv(csv_path)
    
    classes=df['class'].unique().tolist()
    print(f"Classes found: {classes}")
    
    for filename, group in df.groupby('filename'):
        # label filename: same name but .txt
        label_name = os.path.splitext(filename)[0] + '.txt'
        label_path = f'SDO_Sunspots_Extracted/{split}/{label_name}'
        
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                class_id = classes.index(row['class'])
                
                cx = (row['xmin'] + row['xmax']) / 2 / row['width']
                cy = (row['ymin'] + row['ymax']) / 2 / row['height']
                w  = (row['xmax'] - row['xmin']) / row['width']
                h  = (row['ymax'] - row['ymin']) / row['height']
                
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
    print(f"{split}: converted {df['filename'].nunique()} images, {len(df)} annotations")

# Convert all 3 splits
convert_split('train')
convert_split('valid')
convert_split('test')

print("Done! YOLO labels created.")