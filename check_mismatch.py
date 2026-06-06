import pandas as pd
import os

for split in ['train', 'valid', 'test']:
    folder = f'SDO_Sunspots_Extracted/{split}'
    df = pd.read_csv(f'{folder}/_annotations.csv')
    
    # filenames mentioned in CSV
    csv_filenames = set(df['filename'].unique())
    
    # actual jpg files on disk
    disk_filenames = set(f for f in os.listdir(folder) if f.endswith('.jpg'))
    
    # which CSV filenames have no match on disk
    missing = csv_filenames - disk_filenames
    
    print(f"\n{split}:")
    print(f"  CSV has {len(csv_filenames)} unique filenames")
    print(f"  Disk has {len(disk_filenames)} jpg files")
    print(f"  Missing matches: {len(missing)}")
    if missing:
        for m in list(missing)[:5]:  # show first 5
            print(f"    CSV name: {m}")