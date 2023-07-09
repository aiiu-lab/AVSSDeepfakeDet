import os
import cv2
import subprocess
import face_alignment
import numpy as np
from tqdm import tqdm
from shutil import rmtree
from collections import deque
from dataset.utils import warp_img, cut_patch

def crop_ROI(root):
    STD_SIZE = (256, 256)
    STABLE_POINTS = [33, 36, 39, 42, 45]
    WINDOW_MARGIN = 12
    FACE_SIZE = 256
    MOUTH_SIZE = 96
    mean_face_landmarks = np.load('/home/cssung/AVnet/datasets/FF++/20words_mean_face.npy')
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=False,
        device='cuda:0',
    )

    frames_dir = os.path.join(root, 'frames')

    trans = None
    q_frames, q_landmarks, q_name = deque(), deque(), deque()
    for frame_name in tqdm(sorted(os.listdir(frames_dir))):
        img = cv2.imread(os.path.join(frames_dir, frame_name))
        preds = fa.get_landmarks(img)
        if not preds:
            continue
        
        # Add elements to the queues
        q_frames.append(img)
        q_landmarks.append(preds[0])
        q_name.append(frame_name)

        if len(q_frames) == WINDOW_MARGIN:  # Wait until queues are large enough
            smoothed_landmarks = np.mean(q_landmarks, axis=0)

            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frames.popleft()
            cur_name = q_name.popleft()

            # Get aligned frame as well as affine transformation that produced it
            trans_frame, trans = warp_img(
                smoothed_landmarks[STABLE_POINTS, :], mean_face_landmarks[STABLE_POINTS, :], cur_frame, STD_SIZE
            )
            
            # Apply that affine transform to the landmarks
            trans_landmarks = trans(cur_landmarks)

            # Crop region of interest
            cropped_face = cut_patch(
                trans_frame,
                trans_landmarks,
                FACE_SIZE // 2,
                FACE_SIZE // 2,
            )

            cropped_mouth = cut_patch(
                trans_frame,
                trans_landmarks[48:68],
                MOUTH_SIZE // 2,
                MOUTH_SIZE // 2,
            )

            # Save image
            face_path = os.path.join(root, 'faces', cur_name)
            cv2.imwrite(face_path, cropped_face.astype(np.uint8))
            mouth_path = os.path.join(root, 'mouths', cur_name)
            cv2.imwrite(mouth_path, cropped_mouth.astype(np.uint8))


def preprocess(video_path, save_root):
    # Intialize
    video_name = video_path.split('/')[-1].split('.')[0]
    temp_video_path = os.path.join(save_root, video_name, 'video.avi')
    audio_path = os.path.join(save_root, video_name, 'audio.wav')
    frames_dir = os.path.join(save_root, video_name, 'frames')
    landmarks_dir = os.path.join(save_root, video_name, 'landmarks')
    faces_dir = os.path.join(save_root, video_name, 'faces')
    mouths_dir = os.path.join(save_root, video_name, 'mouths')

    # Delete exists directories
    if os.path.exists(frames_dir):
        rmtree(frames_dir)
    if os.path.exists(landmarks_dir):
        rmtree(landmarks_dir)
    if os.path.exists(faces_dir):
        rmtree(faces_dir)
    if os.path.exists(mouths_dir):
        rmtree(mouths_dir)

    # Create new directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(landmarks_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(mouths_dir, exist_ok=True)

    # Change video fps to 25
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" %
               (video_path, temp_video_path))
    _ = subprocess.call(command, shell=True, stdout=None)

    # Get frames from video
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" %
                (temp_video_path, os.path.join(frames_dir, '%06d.jpg')))
    _ = subprocess.call(command, shell=True, stdout=None)

    # Get audio from video
    command = ("ffmpeg -i %s -f wav -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" %
                (temp_video_path, audio_path))
    _ = subprocess.call(command, shell=True, stdout=None)

    # Delete temp video file
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    # Crop faces
    process_root = os.path.join(save_root, video_name)
    crop_ROI(process_root)

    return