import os
import cv2
import mediapipe as mp
import pandas as pd
import math

mp_pose = mp.solutions.pose

# number of landmarks supported by the installed mediapipe version
NUM_LANDMARKS = len(list(mp_pose.PoseLandmark))

video_path = './dataset/raw/'
images_path = './dataset/images/'
csv_path = './dataset/metadata/'

# If True, only write a single unified CSV (pose_data.csv). If False, also write per-video CSVs.
UNIFIED_ONLY = True

os.makedirs(images_path, exist_ok=True)
os.makedirs(csv_path, exist_ok=True)

# allowed video extensions to scan
VIDEO_EXTS = ('.mov', '.mp4', '.avi', '.mkv', '.MOV')

# known action substrings to infer labels from filenames (order matters for specificity)
KNOWN_ACTIONS = ['caminar_atras', 'sentarse', 'pararse', 'caminar', 'girar']

all_data = []

# Create a Pose instance with recommended parameters and use it across videos
with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    # gather video files from raw folder
    try:
        files = [f for f in os.listdir(video_path) if f.lower().endswith(VIDEO_EXTS)]
    except FileNotFoundError:
        print(f"Carpeta de vídeos no encontrada: {video_path}")
        files = []

    if not files:
        print(f"No se encontraron vídeos en {video_path} con extensiones {VIDEO_EXTS}")

    for video_file in files:
        video_file_path = os.path.join(video_path, video_file)
        # infer label by checking known action substrings in filename
        lname = video_file.lower()
        label = 'unknown'
        for a in KNOWN_ACTIONS:
            if a in lname:
                label = a
                break

        print(f"Procesando {video_file} -> etiqueta: {label}")

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"No se pudo abrir {video_file_path}, saltando.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'  FPS del video {video_file}: {fps}')

        frame_idx = 0

        video_folder_name = os.path.splitext(video_file)[0]
        video_images_path = os.path.join(images_path, video_folder_name)
        os.makedirs(video_images_path, exist_ok=True)

        video_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # save frame image (optional, can be large)
            frame_image_path = os.path.join(video_images_path, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(frame_image_path, frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_data = {
                    'video': video_file,
                    'frame_idx': frame_number,
                    'label': label
                }

                # explicitly iterate NUM_LANDMARKS to keep column schema stable
                for i in range(NUM_LANDMARKS):
                    lm = results.pose_landmarks.landmark[i]
                    frame_data[f'x_{i}'] = lm.x
                    frame_data[f'y_{i}'] = lm.y
                    frame_data[f'z_{i}'] = lm.z
                    frame_data[f'visibility_{i}'] = lm.visibility

                video_data.append(frame_data)
                all_data.append(frame_data)

            frame_idx += 1

        cap.release()

        # Optionally save per-video CSV (guarantee consistent columns even if empty)
        if not UNIFIED_ONLY:
            video_csv = os.path.join(csv_path, f"{video_folder_name}.csv")
            dfv = pd.DataFrame(video_data)
            if dfv.empty:
                cols = ['video', 'frame_idx', 'label']
                for i in range(NUM_LANDMARKS):
                    cols.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'visibility_{i}'])
                dfv = pd.DataFrame(columns=cols)
            dfv.to_csv(video_csv, index=False)

# Save aggregated CSV for all videos (unified)
agg_csv = os.path.join(csv_path, 'pose_data.csv')
df_all = pd.DataFrame(all_data)
if df_all.empty:
    cols = ['video', 'frame_idx', 'label']
    for i in range(NUM_LANDMARKS):
        cols.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'visibility_{i}'])
    df_all = pd.DataFrame(columns=cols)
df_all.to_csv(agg_csv, index=False)