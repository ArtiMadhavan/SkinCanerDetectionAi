# Dataset Directory

This project uses the **HAM10000 dataset**.

## Instructions
1. Download the HAM10000 metadata CSV and images from Kaggle or Harvard Dataverse.
2. Place the `HAM10000_metadata.csv` inside `data/raw/`.
3. Extract all images (`.jpg`) into the `data/raw/HAM10000_images/` directory.

### Final Structure
```
data/
└── raw/
    ├── HAM10000_images/
    │   ├── ISIC_0024306.jpg
    │   ├── ISIC_0024307.jpg
    │   └── ...
    └── HAM10000_metadata.csv
```
