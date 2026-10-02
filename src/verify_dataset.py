import sys
from pathlib import Path
import pandas as pd

# Add the project root to python path if run from inside src/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import load_metadata
from src.preprocessing import split_data

def verify_dataset():
    print("Loading metadata and resolving image paths...\n")
    df = load_metadata()
    
    total_images = len(df)
    duplicates = df['image_id'].duplicated().sum()
    
    print(f"--- Global Dataset Info ---")
    print(f"Total images found and verified: {total_images}")
    print(f"Missing images: {10015 - total_images}") # 10015 is standard HAM10000 total
    print(f"Duplicate image IDs: {duplicates}")
    print("\nSplitting dataset (70% train, 15% val, 15% test)...\n")
    
    train_df, val_df, test_df = split_data(df)
    
    splits = [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]
    
    train_lesions = set(train_df['lesion_id'])
    val_lesions = set(val_df['lesion_id'])
    test_lesions = set(test_df['lesion_id'])
    
    # Check for leakage
    val_leak = train_lesions.intersection(val_lesions)
    test_leak = train_lesions.intersection(test_lesions)
    val_test_leak = val_lesions.intersection(test_lesions)
    
    leakage_found = len(val_leak) > 0 or len(test_leak) > 0 or len(val_test_leak) > 0
    
    for name, split_df in splits:
        print(f"--- {name} Split ---")
        print(f"Image Count: {len(split_df)} ({len(split_df)/total_images*100:.1f}%)")
        print(f"Unique Lesions: {split_df['lesion_id'].nunique()}")
        print("Class Distribution:")
        print(split_df['dx'].value_counts())
        print()
        
    print("--- Leakage Check ---")
    print(f"Lesion IDs in both Train and Val: {len(val_leak)}")
    print(f"Lesion IDs in both Train and Test: {len(test_leak)}")
    print(f"Lesion IDs in both Val and Test: {len(val_test_leak)}")
    
    if not leakage_found:
        print("\n✅ SUCCESS: No data leakage detected. Lesions are strictly segregated.")
    else:
        print("\n❌ ERROR: Data leakage detected!")

if __name__ == "__main__":
    verify_dataset()
