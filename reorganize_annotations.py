import os
import argparse
from pathlib import Path
import glob
import pandas as pd
import shutil

video_path = './PSI_Intention/Dataset/RawVideos'
frames_path = './PSI_Intention/Dataset/frames'
xml_path = './PSI_Intention/Dataset/XmlFiles'

nlp_annotation_path = './PSI_Intention/Dataset/nlp_annotations'
cv_annotation_path = './PSI_Intention/Dataset/cv_annotations'

#create 'data/cv_annotations' folder
if not os.path.exists(cv_annotation_path):
    os.makedirs(cv_annotation_path)
    print("Created 'cv_annotations' folder.")
    
    
#create 'data/nlp_annotation_path' folder
if not os.path.exists(nlp_annotation_path):
    os.makedirs(nlp_annotation_path)
    print("Created 'nlp_annotation' folder.")
    
# re-organize cv annots
for video_file in os.listdir(frames_path):
    video_num = video_file.split('_')[1]
    if not os.path.exists(os.path.join(cv_annotation_path, video_file)):
        os.mkdir(os.path.join(cv_annotation_path, video_file))
        
    src = os.path.join(xml_path, video_num + '.xml')
    dst = os.path.join(cv_annotation_path, video_file, 'annotations.xml')
    try:
        shutil.copyfile(src, dst)
    except:
        print("Failed copying {} to {}".format(src, dst))
print("WARNING: video_0060 and video_0093 cv_annotations are missing. These two samples are abandoned.")

# re-organize cv annots
df = pd.read_excel('./PSI_Intention/Dataset/IntentAnnotations.xlsx')
for video_file in os.listdir(frames_path):
    video_num = video_file.split('_')[1]
    if not os.path.exists(os.path.join(nlp_annotation_path, video_file)):
        os.mkdir(os.path.join(nlp_annotation_path, video_file))
    
    try:
        sub_df = df[df['video_id'] == int(video_num)]
        dst = os.path.join(nlp_annotation_path, video_file, 'intentSegmentation.csv')
        sub_df.to_csv(dst, index=None, header=True)
    except:
        print("Failed create nlp annotations {}".format(dst))