import os
import argparse
from pathlib import Path
import glob
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import math
import time
import xml.etree.ElementTree as ET
import copy
from tqdm import tqdm

data_root = './PSI_Intention/Dataset'
database_path = './PSI_Intention/Dataset/database'
args = {}
args['annot_path'] = os.path.join(data_root, 'cv_annotations')
args['nlp_path'] = os.path.join(data_root, 'nlp_annotations')
args['frames_path'] = os.path.join(data_root, 'frames')
args['pedID_path'] = os.path.join(data_root, 'additional_info/pedID.xlsx')
args['mapping_path'] = os.path.join(data_root, 'additional_info/video_name_mapping.xlsx')
args['vf_path'] = os.path.join(data_root, 'visual_features')
args['save_path'] = database_path


'''
    This function creates a .csv file mapping the pedeistrian's ID and the video ID
'''
def get_pedID(root_dir, args):
    """creates dataframe with pedID, video name, and video ID"""
    cols = ['ID', 'NLP Annotation', 'video_name']
    pedID_df = pd.read_excel(args['pedID_path'], usecols=cols)
    #removing rows that aren't a main pedestrian
    pedID_df = pedID_df.loc[pedID_df['NLP Annotation'] != 0]
    name_df = pd.read_excel(args['mapping_path'])

    merged_df = pd.merge(pedID_df, name_df, on='video_name')

    return merged_df
pedID_df = get_pedID(root_dir=data_root, args=args)
pid = pedID_df


'''
    This function initialize the database dict based on the pedID
'''
'''
    db = {
        'video_0001': {
            '1_MC': {
                'frames': None (lits of frame #s), pedestrians appear. 
                'mean_intention': None (0, 0.5, 1)
                'major_intention': None
                'disagree_score': None # consider all total votes as 24
                'valid_disagree_score': None # only calculate the valid votes sum
                'bbox': None
                'reason_feats': None
                'description_feats': None
                'original_intention': list of all annotators
                'original_reason': list of all annotators
                'labeled_frames': list of frames with labels, overlap with 'frames'
            }
        }
    }
'''
def create_db(root_dir, args, pedID_df):
    db = {}
    for index, row in pedID_df.iterrows():
        video_name = 'video_' + str(row["video_id"]).zfill(4)
        pedID = row["ID"]
        db[video_name] = {pedID: {'frames': None, 'mean_intention': None, 'major_intention': None,
                                  'disagree_score': None, 'labeled_frames': None,
                                  'bbox' : None, 'reason_feats': None,'original_reason': None,
                                  'valid_disagree_score': None,'original_intention': None}}
                                  

    #TODO: get cv annotations for the excluded videos
    db.pop('video_0060')
    db.pop('video_0093')
    return db
database = create_db(data_root, args, pid)


'''
    Get samples with cv annotations
'''
def load_xml(video, root_dir):
    #Loads XML file and gets bbox coordinates and creates id for each bbox in the XML file
    tree = ET.parse(os.path.join(root_dir, 'cv_annotations', video, 'annotations.xml'))
    root = tree.getroot()
    file_location = os.path.join(root_dir, 'visual_features', video)
    #finds all track nodes
    for obj in tqdm(root.findall('track')):
        #print(obj.get('label'))
        label = obj.get('label')
        #for the found track node, list out bbox attributes
        for box in obj.findall('box'):
            if box.get('outside') == '1':
                continue
            else:
                framenum = box.get('frame')
                framenum = framenum.zfill(3)
                bbox = (float(box.get('xtl')),
                        float(box.get('ytl')),
                        float(box.get('xbr')),
                        float(box.get('ybr'))
                        )
                #Check whether 'ID' field is filled
                file_name = None
                for attribute in box.iter('attribute'):
                    if attribute.get('name') == 'ID':
                        #No ID
                        if attribute.text == 'n/a':
                            id = obj.get('id')
                            file_name = video + '_' + 'f' + framenum + '_' + label + id + '.npz'
                            file_location = os.path.join(root_dir, 'visual_features', video)
                        #Specified ID
                        else:
                            id = (attribute.text)
                            file_name = video + '_' + 'f' + framenum + '_' + label + id + '.npz'
                            file_location = os.path.join(root_dir, 'visual_features', video)

                if not os.path.exists(file_location):
                    os.makedirs(file_location)
                if file_name:
                    if not os.path.exists(os.path.join(file_location, file_name)):
                        features = np.array([]) #load_process_image(args, root_dir, video, framenum, bbox, model)
                        save_path = os.path.join(file_location, file_name)
                        np.savez_compressed(save_path, features)
                else:
                    print("No attributes found frame {}_{}".format(framenum, label))
                    

if not os.path.exists(os.path.join(data_root, 'visual_features')):
    for video in sorted(os.listdir(os.path.join(data_root, 'cv_annotations'))):
        try:
            print(f'Processing {video}.')
            load_xml(video, data_root)
        except:
            print("Faild processing {}".format(video))
else:
    print("Frame lists already exist!")


'''
    This function returns the frames number list of each specific pedID appears.
    Notice: This frames list is not obtained directly from xml annotations, but from the
    VGG features already processed based on each bbox.
    e.g., database['video_0001']['139_MC']['frames'] = [135, 136, ..., 256]
'''
def get_frames(root_dir, args, db, df):

    for index, row in df.iterrows():
        video_name = 'video_' + str(row["video_id"]).zfill(4)
        pedID = row["ID"]
        vf_path = os.path.join(args['vf_path'], video_name)
#         print(vf_path)
        try:
            vf_files = os.listdir(vf_path)
#             print(vf_files)
            vf_files.sort()
            f = [file_name[12:15] for file_name in vf_files if file_name[(-4 - len(pedID)):-4] == pedID]
            db[video_name][pedID]['frames'] = f
        except:
            print(f'Could not find {video_name} in database.')

    return db
database = get_frames(data_root, args, database, pid)


'''
    Return the annotated pedestrians bbox list of each frame.
    Notice here only take the pedestrians bbox, so each frame has 1, all sequence of bbox
    has same length as the frames for each pedestrian.
'''
def get_bbox(root_dir, args, db, df):
    for index, row in df.iterrows():
        video_name = 'video_' + str(row["video_id"]).zfill(4)
        pedID = row["ID"]
        bbox = []
        try:
            tree = ET.parse(os.path.join(args['annot_path'], video_name, 'annotations.xml'))
            root = tree.getroot()
            for frame in db[video_name][pedID]['frames']:
                # for each frame
                for obj in root.findall('track'):
                    if obj.get('label') == 'pedestrian':
                        # get the bbox labeled as 'pedestrian'
                        for box in obj.findall('box'):
                            if box.get('frame') == frame.lstrip('0'):
                                for attribute in box.iter('attribute'):
                                    if attribute.get('name') == 'ID':
                                        # if the bbox pedID same as the feature extracted before
                                        if attribute.text == pedID:
                                            box = [float(box.get('xtl')),
                                                   float(box.get('ytl')),
                                                   float(box.get('xbr')),
                                                   float(box.get('ybr'))]
                                            x1,y1,x2,y2 = box
                                            if (x2 - x1) < 1 or (y2 - y1) < 1:
                                                print(video_name, pedID, box)

                                            bbox.append(box)
            # Each frame will only have one specific pedestrian box, so concatenate as list
            db[video_name][pedID]['bbox'] = bbox

        except:
            print(f'Could not find {video_name} in database.')

    return db

bbox_database = get_bbox(data_root, args, copy.deepcopy(database), pid)


# video_name = 'video_' + str(83).zfill(4)

# cols = ['video_time', 'ped_intention_cat', 'user_id', 'ped_reasoning']
# int_df = pd.read_csv(os.path.join(args['nlp_path'], video_name, 'intentSegmentation.csv'), usecols=cols)


'''
    This function get crossing intention of each pedestrians
'''
def get_intention(root_dir, args, db, df):
    total = 0
    int_count = [0, 0, 0]
    for index, row in df.iterrows(): # For each ped_id & vid_id
#         if row['video_id'] != 2:
#             continue
        video_name = 'video_' + str(row["video_id"]).zfill(4)
        pedID = row["ID"]
        cols = ['video_time', 'ped_intention_cat', 'user_id', 'ped_reasoning']#'reasoning_labeled']
#         int_df = pd.read_csv(os.path.join(args['nlp_path'], video_name, 'intentSegmentation_' + video_name[6:] + '_labeled.csv'), usecols=cols)
        int_df = pd.read_csv(os.path.join(args['nlp_path'], video_name, 'intentSegmentation.csv'), usecols=cols)
        
        # for each frame with annotations
        for row_id, row in int_df.iterrows():
            #conver seconds to frames
            time = row['video_time']
            int_df.at[row_id,'video_time'] = math.trunc(time * 30) # change time to frame #
            #convert text to numerical class
            intention = row['ped_intention_cat']
            if intention == 'not_cross':
                int_df.at[row_id,'ped_intention_cat'] = 0
            elif intention == 'not_sure':
                int_df.at[row_id,'ped_intention_cat'] = 0.5
            elif intention == 'cross':
                int_df.at[row_id,'ped_intention_cat'] = 1
        int_df['video_time'] = int_df['video_time'].astype(int) # already changed to frame #
        #re-arrange dataframe so each column is a different user
        int_df = int_df.drop_duplicates(subset = ['video_time', 'user_id'], keep = 'last')
        ori_int_df = copy.deepcopy(int_df)
        
        isna = int_df['ped_reasoning'].isna()
        print(int_df['ped_intention_cat'].isna().sum(), " nan intention cat | ", isna.sum(), " nan reasoning labels")

#         print(int_df.shape)
        time_intent_map = int_df.pivot(index = 'video_time', columns='user_id', values = 'ped_intention_cat')

        start_frame, end_frame = time_intent_map.index[0], time_intent_map.index[-1]
        print("Start_frame: ", start_frame, " End frame: ", end_frame)
        total += 450 - start_frame + 1 #end_frame - start_frame + 1
        
#         time_intent_map = time_intent_map.reindex(list(range(0,451)),fill_value=np.nan).iloc[start_frame: end_frame+1, :]
        
        # Note: here all last frames are annotated with the last intention label, and they will have all reasons as 0s
        time_intent_map = time_intent_map.reindex(list(range(0,451)),fill_value=np.nan).iloc[start_frame: , :]
        
        time_intent_map.fillna(method = 'ffill', inplace=True)
        
        print(time_intent_map.isna().sum().sum(), " -1 are added.")
        
        time_intent_map.fillna(-1.0, inplace=True) 
        # Scott: '-1' means this kind of labels should be ignored!
        
#         int_df['avg'] = int_df.mean(axis = 1) 
#         print(int_df['avg'].values[100:])
        # Scott: those filled with -1.0 values shouldn't be used.
        frame_length = time_intent_map.shape[0]
        major_intention = [-1] * frame_length
        mean_intention = [-1] * frame_length
        original_intention = []
        disagree_score = [-1] * frame_length
        valid_disagree_score = [-1] * frame_length
        for i in range(frame_length):
            frame_id = start_frame + i
#             if frame_id != 60:
#                 continue
            cur_frame_int = time_intent_map.values[i, :] # may contain -1, which should be ignored
            original_intention.append(cur_frame_int)
            int_lbl, votes = np.unique(cur_frame_int, return_counts=True)
#             print(int_lbl, votes)
            total_valid_votes = 0

         #**************************************************
            # Store the voted rates for 3 intention categories
            temp_int = [0, 0, 0]
            max_vote = 0
            for j in range(len(int_lbl)): # unique intent lbl list
                if int_lbl[j] == -1:
                    continue
                else:
                    if int_lbl[j] == 0.0:
                        cur_int = 0
                    elif int_lbl[j] == 0.5:
                        cur_int = 1
                    elif int_lbl[j] == 1.0:
                        cur_int = 2
                    else:
                        raise Exception("Error int_lbl[j]")
                    int_count[cur_int] += 1
                    
                    cur_vot = votes[j] # number of cur int votes
                    total_valid_votes += votes[j]
#                     print(cur_int, cur_vot, type(cur_int), type(cur_vot))
                    temp_int[cur_int] = cur_vot
                    if cur_vot > max_vote:
                        max_vote = cur_vot
                    else:
                        continue
            disagree_score[i] = 1 - max_vote / 24
            valid_disagree_score[i] = 1 - max_vote / total_valid_votes
            major_intention[i] = [temp_int[k] / total_valid_votes for k in range(3)]
            # major_intention[i] is 3 dimension list
            
            # Get mean intention votes
            temp_sum = 0
            temp_cnt = 0
            for j in range(len(int_lbl)):
                if int_lbl[j] == -1:
                    continue
                else:
                    temp_sum += int_lbl[j] * votes[j]
                    temp_cnt += votes[j]
            assert temp_cnt == total_valid_votes
            assert temp_cnt > 0
            mean_intention[i] = temp_sum / temp_cnt
            # mean intention of one float in [0, 1]
#             print("temp sum: ", temp_sum)
#             print("mean intent: ", mean_intention[i], temp_cnt)    
            
#         print("major intent: ", major_intention)
#         print("disagree score: ", disagree_score)
#         print("mean intent: ", mean_intention)

        try:
            db[video_name][pedID]['major_intention'] = major_intention
            db[video_name][pedID]['mean_intention'] = mean_intention
            db[video_name][pedID]['original_intention'] = original_intention
            db[video_name][pedID]['disagree_score'] = disagree_score
            db[video_name][pedID]['valid_disagree_score'] = valid_disagree_score
            db[video_name][pedID]['labeled_frames'] = time_intent_map.index.tolist()
            print("Ped appear frames: ", db[video_name][pedID]['frames'][0], " -- ", db[video_name][pedID]['frames'][-1])
            print("Labeled frames: ", db[video_name][pedID]['labeled_frames'][0], ' -- ', db[video_name][pedID]['labeled_frames'][-1])
        except:
            print(f'{video_name} not part of dataset.')

            
#         # Reason feats --------------------------
        print("----- reason ------")
        time_rsn_map = ori_int_df.pivot(index = 'video_time', columns='user_id', values = 'ped_reasoning')
        start_frame, end_frame = time_rsn_map.index[0], time_rsn_map.index[-1]
        print("Start_frame: ", start_frame, " End frame: ", end_frame)
        

        # Note: last frames reasons are fill with 0s
        time_rsn_map = time_rsn_map.reindex(list(range(0,451)),fill_value=np.nan).iloc[start_frame: , :]
        
        time_rsn_map.fillna(method = 'bfill', inplace=True)
        print(time_rsn_map.isna().sum().sum(), " -1 are added.")
        
        time_rsn_map.fillna(-1.0, inplace=True) 
        
        original_reason = []
        reason_feats = []
        for vtime, feats in time_rsn_map.iterrows(): # only labeled frames
#             vtime_sum_feats = [0] * 62
            vtime_ori_rsn = []
            for uid in time_rsn_map.columns: # wr columns
                vtime_ori_rsn.append(feats[uid])
                
                if feats[uid] == -1:
                    vtime_ori_rsn.append(-1)
#                     uid_rsn = [0 for _ in range(62)]
#                     assert len(vtime_sum_feats) == len(uid_rsn)
#                     vtime_sum_feats = [a+b for a,b in zip(vtime_sum_feats, uid_rsn)]
                else:
                    vtime_ori_rsn.append(feats[uid])
#                     uid_rsn = [int(i) for i in feats[uid][1:-1].split(",")]
#                     assert len(vtime_sum_feats) == len(uid_rsn)
#                     vtime_sum_feats = [a+b for a,b in zip(vtime_sum_feats, uid_rsn)]
#             reason_feats.append(vtime_sum_feats)
            original_reason.append(vtime_ori_rsn)
        try:
            db[video_name][pedID]['original_reason'] = original_reason
            db[video_name][pedID]['reason_feats'] = reason_feats
            assert len(db[video_name][pedID]['reason_feats']) == len(db[video_name][pedID]['labeled_frames'])
        except:
            print(f'{video_name} not part of dataset.')
    print("Intention count: ", int_count, " | total=", total)
    return db



intent_database = get_intention(data_root, args, copy.deepcopy(bbox_database), pid)
# , intention, reason


print(len(intent_database['video_0001']['139_MC']['original_reason']))
print(intent_database['video_0001']['139_MC']['original_reason'][-1])


print(len(intent_database['video_0027']['150_MC']['bbox']))
intent_database['video_0027']['150_MC']['bbox'][-5:]



'''
    Only keep the intention labels corresponding to each pedestrian, instead of all pedestrianID
    takes all frames intention labels
    Notice: Such operation will avoid frames no Pedestrian appears!
    Notice: Also should slice the reaoning/description features
'''
def slice_intention(db):
    for video, value1 in db.items():
        for pedID, value2 in db[video].items():
#             print(video, pedID)
            db[video][pedID]['frames'] = [int(f) for f in db[video][pedID]['frames']]
            frames = db[video][pedID]['frames'] # original cv annotated frames
            labeled_frames = db[video][pedID]['labeled_frames'] # frames with intention labels
            frame_min, frame_max = int(min(frames)), int(max(frames))
            labeled_min, labeled_max = int(min(labeled_frames)), int(max(labeled_frames))
            
#             print(frame_min, frame_max)
#             print(labeled_min, labeled_max)
            
#             print(frames)
#             print(labeled_frames)
            max_start = max(frame_min, labeled_min)
            min_end = min(frame_max, labeled_max)
            try:
                frame_start_idx, frame_end_idx = frames.index(max_start), frames.index(min_end)
                labeled_start_idx, labeled_end_idx = labeled_frames.index(max_start), labeled_frames.index(min_end)
                
            except:
                print("No element in the list.", video, pedID,  min_end - max_start)
                print("!!! Skip the cut of ", video, "!!!")
                continue
#                 print(frames)
#                 print(labeled_frames)
            # 1. frames, bbox
            db[video][pedID]['frames'] = db[video][pedID]['frames'][frame_start_idx: frame_end_idx+1]
            db[video][pedID]['bbox'] = db[video][pedID]['bbox'][frame_start_idx: frame_end_idx+1]
            
            # original_reason, original_intention
            db[video][pedID]['mean_intention'] = db[video][pedID]['mean_intention'][labeled_start_idx: labeled_end_idx+1]
            db[video][pedID]['major_intention'] = db[video][pedID]['major_intention'][labeled_start_idx: labeled_end_idx+1]
            db[video][pedID]['disagree_score'] = db[video][pedID]['disagree_score'][labeled_start_idx: labeled_end_idx+1]
            db[video][pedID]['valid_disagree_score'] = db[video][pedID]['valid_disagree_score'][labeled_start_idx: labeled_end_idx+1]
            
            db[video][pedID]['labeled_frames'] = db[video][pedID]['labeled_frames'][labeled_start_idx: labeled_end_idx+1]
            db[video][pedID]['reason_feats'] = []#db[video][pedID]['reason_feats'][labeled_start_idx: labeled_end_idx+1]
            db[video][pedID]['original_reason'] = db[video][pedID]['original_reason'][labeled_start_idx: labeled_end_idx+1]
            db[video][pedID]['original_intention'] = db[video][pedID]['original_intention'][labeled_start_idx: labeled_end_idx+1]

            if len(db[video][pedID]['frames']) != len(db[video][pedID]['labeled_frames']):
                print("Different frames v.s. labeled frames: ", video, pedID)
                print(len(db[video][pedID]['frames']), len(db[video][pedID]['bbox']),
                  len(db[video][pedID]['mean_intention']),len(db[video][pedID]['major_intention']), 
                  len(db[video][pedID]['disagree_score']), len(db[video][pedID]['labeled_frames']), 
                  len(db[video][pedID]['reason_feats']), len(db[video][pedID]['original_reason']),
                  len(db[video][pedID]['original_intention']))
    return db


sliced_database = slice_intention(copy.deepcopy(intent_database))



i = 0
j = 0
for v in sliced_database.keys():
    for p in sliced_database[v].keys():
        sample = sliced_database[v][p]
#         for reason in sample['reason_feats']:
#             if len(reason) == 0:
#                 i += 1
    
        j += 1
print("reason feats: ", i, j)

i = 0
j = 0
for v in sliced_database.keys():
    for p in sliced_database[v].keys():
        sample = sliced_database[v][p]
        for intent in sample['major_intention']:
            if intent == -1:

                i += 1
    
            j += 1
print("intent: ", i, j)



def check_missing(db):
    for video, value1 in db.items():
        for pedID, value2 in db[video].items():
            if len(db[video][pedID]['frames']) != len(db[video][pedID]['labeled_frames']):
                print("Different frames v.s. labeled frames: ", video, pedID)
                print(len(db[video][pedID]['frames']), len(db[video][pedID]['bbox']),
                  len(db[video][pedID]['mean_intention']),len(db[video][pedID]['major_intention']), 
                  len(db[video][pedID]['disagree_score']), len(db[video][pedID]['labeled_frames']), 
#                   len(db[video][pedID]['reason_feats']), len(db[video][pedID]['original_reason']),
                  len(db[video][pedID]['original_intention']))
                print("Frame start&end: ", db[video][pedID]['frames'][0], db[video][pedID]['frames'][-1])
                print("labeled_frames start&end: ", db[video][pedID]['labeled_frames'][0], db[video][pedID]['labeled_frames'][-1])
                missing_frames = []
                for l in db[video][pedID]['labeled_frames']:
                    if l not in db[video][pedID]['frames']:
                        missing_frames.append(l)
                print("Missing frames: ", missing_frames)
                if missing_frames[-1] - missing_frames[0] + 1 == len(missing_frames):
                    # only one missing piece, remove intentions labels
                    print("Missing one range: ", missing_frames[0], " - ", missing_frames[-1])
                    missing_pieces = [missing_frames[0],missing_frames[-1]]
                else:
                    # multiple missing pieces, find them
                    missing_pieces = []
                    start = -1
                    for f in range(len(missing_frames)-1):
                        if start == -1:
                            start = missing_frames[f]
                        if missing_frames[f] + 1 == missing_frames[f+1]:
                            if f + 1 == len(missing_frames) - 1:
                                missing_pieces.append([start, missing_frames[f+1]])
                            continue
                        else:
                            # current f is the end of current piece
                            missing_pieces.append([start, missing_frames[f]])
                            start = -1
                    print("Splited missing pieces: ", missing_pieces)
                
                print("--------------------------------------------")
            else:
                if len(db[video][pedID]['frames']) != len(db[video][pedID]['bbox']):
                    print("Different bbox length!", video)
                    print(db[video][pedID]['frames'], db[video][pedID]['bbox'], db[video][pedID]['labeled_frames'])
                else:
                    print("All lengths are the same! ", video)
                no_missing = True
                for f in db[video][pedID]['frames']:
                    if f not in db[video][pedID]['labeled_frames']:
                        print("frames ", f, " not in labeled_frames")
                        no_missing = False
                for l in db[video][pedID]['labeled_frames']:
                    if l not in db[video][pedID]['frames']:
                        print("labeled_frames ", l, " not in frames")
                        no_missing = False
                if no_missing:
                    print("No missing frames! ")


def remove_missing_intention(db):
    for video, value1 in db.items():
        for pedID, value2 in db[video].items():
            if len(db[video][pedID]['frames']) != len(db[video][pedID]['labeled_frames']) or             len(db[video][pedID]['frames']) != len(db[video][pedID]['major_intention']) or             len(db[video][pedID]['major_intention']) != len(db[video][pedID]['labeled_frames']):
                print("Different frames v.s. labeled frames: ", video, pedID)
                print(len(db[video][pedID]['frames']), len(db[video][pedID]['bbox']),
                  len(db[video][pedID]['mean_intention']),len(db[video][pedID]['major_intention']), 
                  len(db[video][pedID]['disagree_score']), len(db[video][pedID]['valid_disagree_score']), 
                      len(db[video][pedID]['labeled_frames']), 
#                   len(db[video][pedID]['reason_feats']), len(db[video][pedID]['original_reason']),
                  len(db[video][pedID]['original_intention']))
                print("Frame start&end: ", db[video][pedID]['frames'][0], db[video][pedID]['frames'][-1])
                print("labeled_frames start&end: ", db[video][pedID]['labeled_frames'][0], db[video][pedID]['labeled_frames'][-1])
                missing_frames = []
                for l in db[video][pedID]['labeled_frames']:
                    if l not in db[video][pedID]['frames']:
                        missing_frames.append(l)
                print("Missing frames: ", missing_frames)
                if missing_frames[-1] - missing_frames[0] + 1 == len(missing_frames):
                    # only one missing piece, remove intentions labels
                    print("Missing one range: ", missing_frames[0], " - ", missing_frames[-1])
                    missing_pieces = [[missing_frames[0],missing_frames[-1]]]
                else:
                    # multiple missing pieces, find them
                    missing_pieces = []
                    start = -1
                    for f in range(len(missing_frames)-1):
                        if start == -1:
                            start = missing_frames[f]
                        if missing_frames[f] + 1 == missing_frames[f+1]:
                            if f + 1 == len(missing_frames) - 1:
                                missing_pieces.append([start, missing_frames[f+1]])
                            continue
                        else:
                            # current f is the end of current piece
                            missing_pieces.append([start, missing_frames[f]])
                            start = -1
                    print("Splited missing pieces: ", missing_pieces)
                
                # remove missing frames (intention labels)
                for piece in missing_pieces:
                    missing_start = db[video][pedID]['labeled_frames'].index(piece[0])
                    missing_end = db[video][pedID]['labeled_frames'].index(piece[1])

                    # original_reason, original_intention
                    del db[video][pedID]['mean_intention'][missing_start: missing_end+1]
                    del db[video][pedID]['major_intention'][missing_start: missing_end+1]
                    del db[video][pedID]['disagree_score'][missing_start: missing_end+1]
                    del db[video][pedID]['valid_disagree_score'][missing_start: missing_end+1]
                    
                    del db[video][pedID]['labeled_frames'][missing_start: missing_end+1]
#                     del db[video][pedID]['reason_feats'][missing_start: missing_end+1]
                    del db[video][pedID]['original_reason'][missing_start: missing_end+1]
                    del db[video][pedID]['original_intention'][missing_start: missing_end+1]
                    
                print("--------------------------------------------")
            else:
                print("Same frames and labels: ", video, pedID)
                if len(db[video][pedID]['frames']) != len(db[video][pedID]['bbox']):
                    print("missing bbox ", len(db[video][pedID]['frames']) - len(db[video][pedID]['bbox']))
                    db[video][pedID]['bbox'].append(db[video][pedID]['bbox'][-1])
                    if len(db[video][pedID]['frames']) - len(db[video][pedID]['bbox']) > 1:
                        print("Missing more than 1 bbox annotation! ")
                for f in db[video][pedID]['frames']:
                    if f not in db[video][pedID]['labeled_frames']:
                        print("frames ", f, " not in labeled_frames")
                
                for l in db[video][pedID]['labeled_frames']:
                    if l not in db[video][pedID]['frames']:
                        print("labeled_frames ", l, " not in frames")
                print("================================================")

    return db         


print(len(sliced_database['video_0083']['1_MC']['major_intention']), len(sliced_database['video_0083']['1_MC']['bbox']))



missing_database = copy.deepcopy(sliced_database)
del missing_database['video_0003']
del missing_database['video_0028']



removed_missing_database = remove_missing_intention(missing_database)



check_missing(removed_missing_database)



uni_db = copy.deepcopy(removed_missing_database)



for v in uni_db.keys():
    for p in uni_db[v].keys():
        sample = uni_db[v][p]
        if not (len(sample['frames']) == len(sample['major_intention']) == len(sample['bbox'])):
#                == len(sample['reason_feats'])):
            print(v, p, len(sample['frames']), len(sample['major_intention']), len(sample['bbox'])
               , len(sample['reason_feats']))




for k in uni_db['video_0023']['6_MC'].keys():
    if uni_db['video_0023']['6_MC'][k]:
        print(k, len(uni_db['video_0023']['6_MC'][k]))




for v in uni_db.keys():
    for p in uni_db[v].keys():
        sample = uni_db[v][p]
        if not (len(sample['frames']) == len(sample['major_intention']) == len(sample['bbox'])):
#                == len(sample['reason_feats'])):
            print(v, p, len(sample['frames']), len(sample['major_intention']), len(sample['bbox'])
               , len(sample['reason_feats']))
            uni_db[v][p]['bbox'].append(uni_db[v][p]['bbox'][-1])




for v in uni_db.keys():
    for p in uni_db[v].keys():
        sample = uni_db[v][p]
        if not (len(sample['frames']) == len(sample['major_intention']) == len(sample['bbox'])):
#                == len(sample['reason_feats'])):
            print(v, p, len(sample['frames']), len(sample['major_intention']), len(sample['bbox'])
               , len(sample['reason_feats']))


database_name = 'database_' + time.strftime("%d%b%Y-%Hh%Mm%Ss") + '.pkl'
if not os.path.exists(os.path.join(args['save_path'])):
    os.makedirs(os.path.join(args['save_path']))
with open(os.path.join(args['save_path'], database_name), 'wb') as fid:
    pickle.dump(uni_db, fid)



overlap_db = copy.deepcopy(uni_db)

int_reason_overlap = True
if int_reason_overlap:
    for v in overlap_db.keys():
        for p in overlap_db[v].keys():
            sample = overlap_db[v][p]
#             print([(k, len(sample[k])) for k in sample.keys()])
            print(v, p, len(sample['frames']), len(sample['major_intention']), len(sample['bbox']), len(sample['original_reason']))
            mis_match_list = []
            for i in range(len(sample['frames'])):
                if sum([1 if r==-1 else 0 for r in sample['original_reason'][i]]) == len(sample['original_reason'][i]):
#                     print(i, 'ori_rsn empty: ', sample['original_reason'][i])
                    mis_match_list.append(i)
            # remove mis-match frames intention labels, because intention labels are always longer than reason, till the end of video
            
            if len(mis_match_list) > 0:
                del overlap_db[v][p]['frames'][mis_match_list[0]: mis_match_list[-1]+1]
                del overlap_db[v][p]['bbox'][mis_match_list[0]: mis_match_list[-1]+1]
                del overlap_db[v][p]['mean_intention'][mis_match_list[0]: mis_match_list[-1]+1]
                del overlap_db[v][p]['major_intention'][mis_match_list[0]: mis_match_list[-1]+1]   
                del overlap_db[v][p]['disagree_score'][mis_match_list[0]: mis_match_list[-1]+1]
                del overlap_db[v][p]['valid_disagree_score'][mis_match_list[0]: mis_match_list[-1]+1]
                del overlap_db[v][p]['labeled_frames'][mis_match_list[0]: mis_match_list[-1]+1]
                # del db[video][pedID]['reason_feats'][missing_start: missing_end+1]
                del overlap_db[v][p]['original_reason'][mis_match_list[0]: mis_match_list[-1]+1]
                del overlap_db[v][p]['original_intention'][mis_match_list[0]: mis_match_list[-1]+1]
                print("Removed mismatch: ", v, p, len(sample['frames']), len(sample['major_intention']), len(sample['bbox']), len(sample['original_reason']))
#             print([(k, len(sample[k])) for k in sample.keys()])



database_name = 'database_' + time.strftime("%d%b%Y-%Hh%Mm%Ss") + '_overlap.pkl'
if not os.path.exists(os.path.join(args['save_path'])):
    os.makedirs(os.path.join(args['save_path']))
with open(os.path.join(args['save_path'], database_name), 'wb') as fid:
    pickle.dump(overlap_db, fid)



