import os
import glob
import random
import sys
import json
from collections import Counter
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import paths
from utils.logger import logger

def split_dataset(val_ratio=0.1, seed=42):
    logger.info(f"Splitting dataset with val_ratio={val_ratio}, seed={seed}")
    
    random.seed(seed)
    
    # List all training images
    # We rely on stems to match annotations later
    image_files = glob.glob(os.path.join(paths.TRAIN_IMAGES_DIR, "*.png"))
    
    if not image_files:
        logger.error(f"No images found in {paths.TRAIN_IMAGES_DIR}")
        return

    # Get stems
    stems = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    logger.info(f"Total images found: {len(stems)}")
    
    # 1. Load classes for each image to check stratification
    # This is expensive but necessary for proper logging as requested
    logger.info("Analyzing class distribution for stratification checks...")
    image_classes = {}
    
    all_classes_counter = Counter()
    
    missing_annotations = 0
    
    for stem in stems:
        json_path = os.path.join(paths.TRAIN_ANNOTATIONS_DIR, f"{stem}.json")
        if not os.path.exists(json_path):
            missing_annotations += 1
            image_classes[stem] = []
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            cats = [c['id'] for c in data.get('categories', [])]
            image_classes[stem] = cats
            all_classes_counter.update(cats)
            
        except Exception as e:
            logger.error(f"Error reading {json_path}: {e}")
            image_classes[stem] = []

    if missing_annotations > 0:
        logger.warning(f"Found {missing_annotations} images without annotations.")

    # 2. Random Split
    shuffled_stems = stems.copy()
    random.shuffle(shuffled_stems)
    
    num_val = int(len(stems) * val_ratio)
    val_stems = shuffled_stems[:num_val]
    train_stems = shuffled_stems[num_val:]
    
    logger.info(f"Train: {len(train_stems)}, Val: {len(val_stems)}")
    
    # 3. Analyze Split Quality
    train_classes = Counter()
    val_classes = Counter()
    
    for s in train_stems:
        train_classes.update(image_classes.get(s, []))
        
    for s in val_stems:
        val_classes.update(image_classes.get(s, []))
        
    # Check for missing classes in Val
    all_class_ids = set(all_classes_counter.keys())
    val_class_ids = set(val_classes.keys())
    missing_in_val = all_class_ids - val_class_ids
    
    logger.info(f"Classes in Train: {len(train_classes)}")
    logger.info(f"Classes in Val: {len(val_classes)}")
    
    if missing_in_val:
        logger.warning(f"Alert: {len(missing_in_val)} classes are missing in Validation Set!")
        # If critical, one might want to re-shuffle, but for now we just log as requested.
        logger.debug(f"Missing classes: {sorted(list(missing_in_val))}")
        
    # 4. Save to files
    paths.ensure_dirs()
    
    with open(paths.SPLIT_TRAIN, 'w') as f:
        f.write('\n'.join(train_stems))
        
    with open(paths.SPLIT_VAL, 'w') as f:
        f.write('\n'.join(val_stems))
        
    logger.info(f"Saved splits to {paths.SPLIT_TRAIN} and {paths.SPLIT_VAL}")

if __name__ == "__main__":
    split_dataset()
