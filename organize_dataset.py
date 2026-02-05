import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# PATHS
# -----------------------------
IMAGE_DIR = "HAM10000_images"
CSV_PATH = "HAM10000_metadata.csv"

OUTPUT_BASE = "dataset"
TRAIN_DIR = os.path.join(OUTPUT_BASE, "train")
TEST_DIR = os.path.join(OUTPUT_BASE, "test")
VAL_DIR = os.path.join(OUTPUT_BASE, "validate")

# Create folders
for split in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
    for cls in ["benign", "malignant"]:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

# -----------------------------
# LOAD METADATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Map 7 classes → 2 classes
benign = ['nv', 'bkl', 'df']
malignant = ['mel', 'bcc', 'akiec', 'vasc']

df["binary_class"] = df["dx"].apply(
    lambda x: "benign" if x in benign else "malignant"
)

# -----------------------------
# TRAIN/TEST/VAL SPLIT
# -----------------------------
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["binary_class"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["binary_class"], random_state=42)

def copy_files(dataframe, target_dir):
    for _, row in dataframe.iterrows():
        img = row["image_id"] + ".jpg"
        cls = row["binary_class"]
        src = os.path.join(IMAGE_DIR, img)
        dst = os.path.join(target_dir, cls, img)

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print("Missing:", src)

# -----------------------------
# COPY IMAGES
# -----------------------------
copy_files(train_df, TRAIN_DIR)
copy_files(test_df, TEST_DIR)
copy_files(val_df, VAL_DIR)

print("\nDataset successfully created!")
