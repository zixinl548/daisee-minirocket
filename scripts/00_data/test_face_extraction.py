import cv2
import pandas as pd
import numpy as np
import os

IMG_SIZE = 224
NUM_FRAMES = 8

# OpenCV Haar cascade face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def sample_uniform_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames


def detect_and_crop_face(frame, face_cascade):
    # frame is RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # choose the largest detected face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # small margin around face
    margin = 0.2
    x1 = max(0, int(x - w * margin))
    y1 = max(0, int(y - h * margin))
    x2 = min(frame.shape[1], int(x + w * (1 + margin)))
    y2 = min(frame.shape[0], int(y + h * (1 + margin)))

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face


def main():
    df = pd.read_csv("train_ready.csv")
    df.columns = df.columns.str.strip()

    sample_path = df.iloc[0]["video_path"]
    print("Testing video:", sample_path)

    frames = sample_uniform_frames(sample_path, num_frames=NUM_FRAMES)
    print("Sampled frames:", len(frames))

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    os.makedirs("face_test_outputs", exist_ok=True)

    detected_count = 0
    for i, frame in enumerate(frames):
        face = detect_and_crop_face(frame, face_cascade)

        if face is not None:
            detected_count += 1
            out_path = f"face_test_outputs/face_{i}.jpg"
            cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"Saved face crop: {out_path}")
        else:
            print(f"No face detected in frame {i}")

    print(f"Detected faces in {detected_count}/{len(frames)} frames")


if __name__ == "__main__":
    main()