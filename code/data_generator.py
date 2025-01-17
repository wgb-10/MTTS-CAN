'''
Data Generator for Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
Author: Xin Liu
'''

import math

import h5py
import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths_of_videos, nframe_per_video, dim, batch_size=32, frame_depth=10,
                 shuffle=True, temporal=True, respiration=0):
        self.dim = dim
        self.batch_size = batch_size
        self.paths_of_videos = paths_of_videos
        self.nframe_per_video = nframe_per_video
        self.shuffle = shuffle
        self.temporal = temporal
        self.frame_depth = frame_depth                  # This is the window size
        self.respiration = respiration
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.paths_of_videos) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.paths_of_videos[k] for k in indexes]  # List of video paths
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths_of_videos))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_video_temp):
        'Generates data containing batch_size samples'
        
        # This will be the key for HR data
        label_key = 'dysub'

        if self.temporal == 'TS_CAN':
            
            # Init the data and label arrays

            # dim = (36,36)

            # shape of data: (n_frames_per_vid * no. of vids, 36, 36, 6)
            # I think last dimension is 6 because there are 3 frames per channel (RGB) and 3 normalized frames per channel
            # using formula (c(t + 1) - c(t))/(c(t) + c(t + 1)). For the motion branch, the authors calculate normalized frames 
            # using every two adjacent frames by the formula above. We exclude the last frame and use it to normalize the 
            # previous frame.

            data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6), dtype=np.float32)
            label = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)

            # A window is defined by frames per vid / frame depth (window size)
            num_window = int(self.nframe_per_video / self.frame_depth) * len(list_video_temp)

            # TODO: change this block
            for index, temp_path in enumerate(list_video_temp):
                f1 = h5py.File(temp_path, 'r')
                dXsub = np.transpose(np.array(f1["dXsub"])) 
                dysub = np.array(f1[label_key])

                # Storing dXsub and dysub into the arrays defined above
                data[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :, :, :] = dXsub
                label[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub

            motion_data = data[:, :, :, :3]         # First 3 frames are normalized ones
            apperance_data = data[:, :, :, -3:]     # Last 3 frames are RGB ones (will be the other way around for me)
            apperance_data = np.reshape(apperance_data, (num_window, self.frame_depth, self.dim[0], self.dim[1], 3))
            apperance_data = np.average(apperance_data, axis=1)
            apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
            apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
                                                         apperance_data.shape[2], apperance_data.shape[3],
                                                         apperance_data.shape[4]))
            output = (motion_data, apperance_data)

        else:
            raise ValueError('Unsupported Model!')

        return output, label
        # elif self.temporal == 'MTTS_CAN':
        #     data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6), dtype=np.float32)
        #     label_y = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
        #     label_r = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
        #     num_window = int(self.nframe_per_video / self.frame_depth) * len(list_video_temp)
        #     for index, temp_path in enumerate(list_video_temp):
        #         f1 = h5py.File(temp_path, 'r')
        #         dXsub = np.transpose(np.array(f1["dXsub"])) #dRsub for respiration
        #         drsub = np.array(f1['drsub'])
        #         dysub = np.array(f1['dysub'])
        #         data[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :, :, :] = dXsub
        #         label_y[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub
        #         label_r[index * self.nframe_per_video:(index + 1) * self.nframe_per_video, :] = drsub
        #     motion_data = data[:, :, :, :3]
        #     apperance_data = data[:, :, :, -3:]
        #     apperance_data = np.reshape(apperance_data, (num_window, self.frame_depth, self.dim[0], self.dim[1], 3))
        #     apperance_data = np.average(apperance_data, axis=1)
        #     apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
        #     apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
        #                                                  apperance_data.shape[2], apperance_data.shape[3],
        #                                                  apperance_data.shape[4]))
        #     output = (motion_data, apperance_data)
        #     label = (label_y, label_r)
        # if self.temporal == 'CAN_3D':
        #     num_window = self.nframe_per_video - (self.frame_depth + 1)
        #     data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
        #                     dtype=np.float32)
        #     label = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
        #     for index, temp_path in enumerate(list_video_temp):
        #         f1 = h5py.File(temp_path, 'r')
        #         dXsub = np.transpose(np.array(f1["dXsub"]))
        #         dysub = np.array(f1[label_key])
        #         tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
        #                           for f in range(num_window)])
        #         tempY = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
        #                           for f in range(num_window)])
        #         tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
        #         tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
        #         tempY = np.reshape(tempY, (num_window, self.frame_depth)) # (169, 10)
        #         data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
        #         label[index*num_window:(index+1)*num_window, :] = tempY
        #     output = (data[:, :, :, :, :3], data[:, :, :, :, -3:])
        # elif self.temporal == 'MT_CAN_3D':
        #     num_window = self.nframe_per_video - (self.frame_depth + 1)
        #     data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
        #                     dtype=np.float32)
        #     label_y = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
        #     label_r = np.zeros((num_window * len(list_video_temp), self.frame_depth), dtype=np.float32)
        #     for index, temp_path in enumerate(list_video_temp):
        #         f1 = h5py.File(temp_path, 'r')
        #         dXsub = np.transpose(np.array(f1["dXsub"]))
        #         drsub = np.array(f1['drsub'])
        #         dysub = np.array(f1['dysub'])
        #         tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
        #                           for f in range(num_window)])
        #         tempY_y = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
        #                           for f in range(num_window)])
        #         tempY_r = np.array([drsub[f:f + self.frame_depth] # (169, 10, 1)
        #                           for f in range(num_window)])
        #         tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
        #         tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
        #         tempY_y = np.reshape(tempY_y, (num_window, self.frame_depth)) # (169, 10)
        #         tempY_r = np.reshape(tempY_r, (num_window, self.frame_depth))  # (169, 10)
        #         data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
        #         label_y[index*num_window:(index+1)*num_window, :] = tempY_y
        #         label_r[index * num_window:(index + 1) * num_window, :] = tempY_r
        #     output = (data[:, :, :, :, :3], data[:, :, :, :, -3:])
        #     label = (label_y, label_r)
        # elif self.temporal == 'CAN':
        #     data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6), dtype=np.float32)
        #     label = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
        #     for index, temp_path in enumerate(list_video_temp):
        #         f1 = h5py.File(temp_path, 'r')
        #         dXsub = np.transpose(np.array(f1["dXsub"])) #dRsub for respiration
        #         dysub = np.array(f1[label_key])
        #         data[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :, :, :] = dXsub
        #         label[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub
        #     output = (data[:, :, :, :3], data[:, :, :, -3:])
        # elif self.temporal == 'MT_CAN':
        #     data = np.zeros((self.nframe_per_video * len(list_video_temp), self.dim[0], self.dim[1], 6),
        #                     dtype=np.float32)
        #     label_y = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
        #     label_r = np.zeros((self.nframe_per_video * len(list_video_temp), 1), dtype=np.float32)
        #     for index, temp_path in enumerate(list_video_temp):
        #         f1 = h5py.File(temp_path, 'r')
        #         dXsub = np.transpose(np.array(f1["dXsub"]))  # dRsub for respiration
        #         drsub = np.array(f1['drsub'])
        #         dysub = np.array(f1['dysub'])
        #         data[index * self.nframe_per_video:(index + 1) * self.nframe_per_video, :, :, :] = dXsub
        #         label_y[index*self.nframe_per_video:(index+1)*self.nframe_per_video, :] = dysub
        #         label_r[index * self.nframe_per_video:(index + 1) * self.nframe_per_video, :] = drsub
        #     output = (data[:, :, :, :3], data[:, :, :, -3:])
        #     label = (label_y, label_r)
        # elif self.temporal == 'MT_Hybrid_CAN':
        #     num_window = self.nframe_per_video - (self.frame_depth + 1)
        #     data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
        #                     dtype=np.float32)
        #     label_y = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
        #     label_r = np.zeros((num_window * len(list_video_temp), self.frame_depth), dtype=np.float32)
        #     for index, temp_path in enumerate(list_video_temp):
        #         f1 = h5py.File(temp_path, 'r')
        #         dXsub = np.transpose(np.array(f1["dXsub"]))
        #         drsub = np.array(f1['drsub'])
        #         dysub = np.array(f1['dysub'])
        #         tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
        #                           for f in range(num_window)])
        #         tempY_y = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
        #                           for f in range(num_window)])
        #         tempY_r = np.array([drsub[f:f + self.frame_depth] # (169, 10, 1)
        #                           for f in range(num_window)])
        #         tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
        #         tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
        #         tempY_y = np.reshape(tempY_y, (num_window, self.frame_depth)) # (169, 10)
        #         tempY_r = np.reshape(tempY_r, (num_window, self.frame_depth))  # (169, 10)
        #         data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
        #         label_y[index*num_window:(index+1)*num_window, :] = tempY_y
        #         label_r[index * num_window:(index + 1) * num_window, :] = tempY_r
        #     motion_data = data[:, :, :, :, :3]
        #     apperance_data = np.average(data[:, :, :, :, -3:], axis=-2)
        #     output = (motion_data, apperance_data)
        #     label = (label_y, label_r)
        # elif self.temporal == 'Hybrid_CAN':
        #     num_window = self.nframe_per_video - (self.frame_depth + 1)
        #     data = np.zeros((num_window*len(list_video_temp), self.dim[0], self.dim[1], self.frame_depth, 6),
        #                     dtype=np.float32)
        #     label = np.zeros((num_window*len(list_video_temp), self.frame_depth), dtype=np.float32)
        #     for index, temp_path in enumerate(list_video_temp):
        #         f1 = h5py.File(temp_path, 'r')
        #         dXsub = np.transpose(np.array(f1["dXsub"]))
        #         dysub = np.array(f1[label_key])
        #         tempX = np.array([dXsub[f:f + self.frame_depth, :, :, :] # (169, 10, 36, 36, 6)
        #                           for f in range(num_window)])
        #         tempY = np.array([dysub[f:f + self.frame_depth] # (169, 10, 1)
        #                           for f in range(num_window)])
        #         tempX = np.swapaxes(tempX, 1, 3) # (169, 36, 36, 10, 6)
        #         tempX = np.swapaxes(tempX, 1, 2) # (169, 36, 36, 10, 6)
        #         tempY = np.reshape(tempY, (num_window, self.frame_depth)) # (169, 10)
        #         data[index*num_window:(index+1)*num_window, :, :, :, :] = tempX
        #         label[index*num_window:(index+1)*num_window, :] = tempY
        #     motion_data = data[:, :, :, :, :3]
        #     apperance_data = np.average(data[:, :, :, :, -3:], axis=-2)
        #     output = (motion_data, apperance_data)
