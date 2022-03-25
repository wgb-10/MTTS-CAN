import glob
import os

import h5py
import numpy as np
import scipy.io

def read_many_hdf5(file_path):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        path   path to file

        Returns:
        ----------
        images       images array, (N, W, H, NC) to be stored (where N: number of images, W: width, H: height, NC: number of channels).
        labels       labels array, (N, 1) to be stored
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(file_path, "r+")

    images = np.array(file["/images"]).astype("float64")
    labels = np.array(file["/labels"]).astype("float64")

    return images, labels

# Input: Path to video. Returns: Number of frames in video
def get_nframe_of_video(path):
    _, labels = read_many_hdf5(path)    # No. of frames = no. of GT (ground truth) measurements
    return labels.shape[0]

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