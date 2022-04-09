import os

from utils import read_hdf5


# Input: Path to video. Returns: Number of frames in video
def get_nframe_of_video(path):
    _, labels = read_hdf5(path)    # No. of frames = no. of GT (ground truth) measurements
    return labels.shape[0]

    
# TODO: Shuffle the subjects before splitting them into train and test

""" Steps needed to take for cross validation:
    - Get all subjects in a list
    - shuffle the subjects
    - divide list into 5 folds
    
    For 5 iterations, leave one fold for testing and use other folds for training. Therefore model is tested on all 5 folds.
    Average the results at the end and choose model with best performance. (TODO: Move this line to train.py)
"""


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