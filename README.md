# AI-Based Skin Lesion and Skin Cancer Detection Using Deep Learning and Explainable AI

This is a Healthcare Analytics project designed to analyze dermoscopic skin-lesion images and classify them into multiple lesion categories using deep learning (EfficientNet).

**Disclaimer:** This system is an educational/research prototype and NOT a medical diagnostic tool. Do NOT use it for medical diagnosis.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset:**
   Download the HAM10000 dataset and place the images in `data/raw/HAM10000_images/` and the metadata in `data/raw/HAM10000_metadata.csv`.
   See `data/README.md` for more details.

3. **Train the Model (Optional):**
   ```bash
   python src/train.py
   ```
   Or run the notebooks in the `notebooks/` directory.

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
