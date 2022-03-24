import glob
import os

import h5py
import numpy as np
import scipy.io


# Input: Path to video. Returns: Number of frames in video
# TODO: Modify function to match above specification
def get_nframe_of_video(path):
    temp_f1 = h5py.File(path, 'r')
    temp_dysub = np.array(temp_f1["dysub"])
    nframe_per_video = temp_dysub.shape[0]          # In this case no. frames = no. of GT (ground truth) measurements
    return nframe_per_video

# TODO: Modify this function to split subjects in another way
def split_subj(data_dir, cv_split, subNum):
    f3 = h5py.File(data_dir + '/M.mat', 'r')
    M = np.transpose(np.array(f3["M"])).astype(np.bool)
    subTrain = subNum[~M[:, cv_split]].tolist()
    subTest = subNum[M[:, cv_split]].tolist()
    return subTrain, subTest


# TODO: Test if sort_video_list works 
def take_last_ele(ele):
    ele = ele.split('.')[0][-2:]
    try:
        return int(ele[-2:])    
    except ValueError:
        return int(ele[-1:])


def sort_video_list(data_dir, taskList, subTrain):
    final = []
    for p in subTrain:
        for t in taskList:
            x = glob.glob(os.path.join(data_dir, 'P' + str(p) + 'T' + str(t) + 'VideoB2*.mat'))
            x = sorted(x)
            x = sorted(x, key=take_last_ele)
            final.append(x)
    return final