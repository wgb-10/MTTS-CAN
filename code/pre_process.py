import glob
import os

import h5py
import numpy as np
import scipy.io


def get_nframe_video(path):
    temp_f1 = h5py.File(path, 'r')
    temp_dysub = np.array(temp_f1["dysub"])
    nframe_per_video = temp_dysub.shape[0]          # In this case no. frames = no. of GT (ground truth) measurements
    return nframe_per_video


def get_nframe_video_val(path):
    temp_f1 = scipy.io.loadmat(path)
    temp_dXsub = np.array(temp_f1["dXsub"])
    nframe_per_video = temp_dXsub.shape[0]
    return nframe_per_video


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


def sort_video_list(data_dir, sub_list):
    final = []
    for sub in sub_list:
        x = glob.glob(os.path.join(data_dir, "subject" + str(sub) + "/vid.avi"))
        x = sorted(x)
        x = sorted(x, key=take_last_ele)
        final.append(x)
    return final
