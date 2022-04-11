'''Given video path, extract frames for all videos. Check if frames exist first.'''

import os
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

video_path = './PSI_Intention/Dataset/RawVideos'
frames_path = './PSI_Intention/Dataset/frames'

#create 'data/frames' folder
if not os.path.exists(frames_path):
    os.makedirs(frames_path)
    print("Created 'frames' folder.")
    
for video in tqdm(os.listdir(video_path)):
    name = "video" + video[7:12]
    video_target = os.path.join(video_path, video)
    frames_target = os.path.join(frames_path, name)

    if not os.path.exists(frames_target):
        os.makedirs(frames_target)
        print(f'Created frames folder for video {name}')

    try:
        vidcap = cv2.VideoCapture(video_target)
        if not vidcap.isOpened():
            raise Exception(f'Cannot open file {video_target}')
    except Exception as e:
        raise e

    cur_frame = 0
    while(True):
        success, frame = vidcap.read()
        if success:
            frame_num = str(cur_frame).zfill(3)
            cv2.imwrite(os.path.join(frames_target, f'{frame_num}.jpg'), frame)
        else:
            break
        cur_frame += 1
    vidcap.release()