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

    
# TODO: Shuffle the subjects before splitting them into train and test  
def split_subj(data_dir, cv_split):
    """ Splits the data from data_dir into train and test sets.
    Parameters:
    ---------------
    data_dir:  path to the directory containing the data.
    cv_split:  percentage of the data to be used for testing (written as a float, e.g. 50% = 0.5).

    Returns:
    ----------
    subTrain: list of paths to the training data.
    subTest:  list of paths to the testing data.
"""
    # Get the total number of subjects
    num_sub = len(os.listdir(data_dir))

    # Store the paths of each subject into a list
    sub_paths = [os.path.join(data_dir, sub) for sub in os.listdir(data_dir)]

    # Get the no. of training paths
    num_train = int(num_sub * cv_split)
    
    # Create a list of training paths
    subTrain = sub_paths[:num_train]

    # Create a list of testing paths
    subTest = sub_paths[num_train:]

    return subTrain, subTest


# Not using code below

# def take_last_ele(ele):
#     ele = ele.split('.')[0][-2:]
#     try:
#         return int(ele[-2:])    
#     except ValueError:
#         return int(ele[-1:])


# def sort_video_list(data_dir, taskList, subTrain):
#     final = []
#     for p in subTrain:
#         for t in taskList:
#             x = glob.glob(os.path.join(data_dir, 'P' + str(p) + 'T' + str(t) + 'VideoB2*.mat'))
#             x = sorted(x)
#             x = sorted(x, key=take_last_ele)
#             final.append(x)
#     return final