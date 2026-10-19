import os
import pandas as pd
import cv2
from pathlib import Path

# ===== CHANGE THIS TO YOUR OWN DATASET ROOT =====
DATASET_ROOT = Path("DataSet")
LABEL_ROOT = Path("Labels")

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def build_video_index(split_dir):
    """
    Build mapping:
    ClipID -> full video path
    by scanning split/subject/clip_folder/video_file
    """
    video_map = {}

    for subject_dir in split_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        for clip_dir in subject_dir.iterdir():
            if not clip_dir.is_dir():
                continue

            for f in clip_dir.iterdir():
                if f.is_file() and f.suffix.lower() in [".avi", ".mp4"]:
                    video_map[f.name] = str(f)

    return video_map

def check_split(split_name, label_csv):
    print(f"\n===== Checking {split_name} =====")

    df = pd.read_csv(label_csv)
    df = clean_columns(df)

    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    print(df.head())

    split_dir = DATASET_ROOT / split_name
    video_map = build_video_index(split_dir)

    print("Indexed videos found on disk:", len(video_map))

    df["video_path"] = df["ClipID"].map(video_map)

    missing = df[df["video_path"].isna()]
    print("Missing video files for labels:", len(missing))

    if len(missing) > 0:
        print("First 10 missing ClipIDs:")
        print(missing["ClipID"].head(10).tolist())
        missing.to_csv(f"missing_{split_name.lower()}_clips.csv", index=False)
        print(f"Saved missing list to missing_{split_name.lower()}_clips.csv")

    # Test opening first few matched videos
    matched = df[df["video_path"].notna()].head(5)

    print("\nTesting first 5 matched videos with OpenCV:")
    for _, row in matched.iterrows():
        clip_id = row["ClipID"]
        path = row["video_path"]

        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        print(f"{clip_id}")
        print(f"  path: {path}")
        print(f"  opened: {ok}")
        print(f"  frame_count: {frame_count}")
        print(f"  fps: {fps}")

def main():
    check_split("Train", LABEL_ROOT / "TrainLabels.csv")
    check_split("Validation", LABEL_ROOT / "ValidationLabels.csv")
    check_split("Test", LABEL_ROOT / "TestLabels.csv")

if __name__ == "__main__":
    main()