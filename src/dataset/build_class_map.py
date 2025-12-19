import json
import os
import glob
import pandas as pd
import sys

# Ensure src modules are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import paths
from utils.logger import logger

def build_class_map():
    logger.info("Building class map from annotations...")
    
    annotation_files = glob.glob(
        os.path.join(paths.TRAIN_ANNOTATIONS_DIR, "**", "*.json"),
        recursive=True
    )    
    if not annotation_files:
        logger.error(f"No json files found in {paths.TRAIN_ANNOTATIONS_DIR}")
        return

    unique_classes = {}
    
    # Iterate through all json files to find all unique classes
    for json_file in annotation_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            categories = data.get('categories', [])
            for cat in categories:
                cat_id = cat['id']
                cat_name = cat['name']
                
                if cat_id not in unique_classes:
                    unique_classes[cat_id] = cat_name
                elif unique_classes[cat_id] != cat_name:
                    logger.warning(f"ID Conflict: {cat_id} maps to {unique_classes[cat_id]} and {cat_name}")
                    
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")

    # Sort by original ID
    sorted_classes = sorted(unique_classes.items(), key=lambda x: int(x[0]))
    
    # Create DataFrame
    # yolo_id will be 0-indexed based on sorted order
    class_map_data = []
    for idx, (orig_id, name) in enumerate(sorted_classes):
        class_map_data.append({
            'orig_id': orig_id,
            'yolo_id': idx,
            'class_name': name
        })
        
    df = pd.DataFrame(class_map_data)
    
    # Save to CSV
    paths.ensure_dirs()
    df.to_csv(paths.CLASS_MAP_PATH, index=False)
    
    logger.info(f"Class map saved to {paths.CLASS_MAP_PATH}")
    logger.info(f"Total unique classes: {len(df)}")
    
    return df

if __name__ == "__main__":
    build_class_map()
