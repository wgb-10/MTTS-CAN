# This file contains utility functions for the project.

import numpy as np
import h5py
from PIL import Image
import csv
from glob import iglob
import os
from pathlib import Path
import matplotlib.pyplot as plt


# Reference: https://realpython.com/storing-images-in-python/#reading-many-images (Accessed 18/03/2022)
def display_frame(frame, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(frame)

# %%
def read_many_disk(num_images, imagesPath, gtPath):
    images, labels = [], []


     # For each frame
    for imagePath in imagesPath:
        
        # Stop when the total images read = num_images (to keep the no. of images same across all subjects)
        if len(images) == num_images:
            break

        # Store each frame 
        # print(f'[INFO] Working on Image: {image}')

        # Read and resize the image
        # Reference: https://pillow.readthedocs.io/en/stable/reference/Image.html (Accessed 21/03/2022)

        with Image.open(imagePath) as image:
            image_resized = image.resize((36, 36))
            images.append(np.array(image_resized))
        
        # print(f'[INFO] images list contains: {len(images)} elements  of type {type(images[0])}')

    with open(gtPath, "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=","
        )

        for idx, row in enumerate(reader):
            
            # Skip the title row
            if idx > 0:
                
                # Skip the ppg recording for the last frame as it doesn't have a successor for normalization. 
                # This frame will only be used to normalize the 2nd last frame.
                if len(labels) < num_images - 1:        
                    ppg = float(row[2])                 # row[2] is the column containing ppg signal (label)
                    # print(f'[INFO] ppg: {ppg}')
                    labels.append(ppg)  

    # print(f'[INFO] labels list contains: {len(labels)} elements  of type {type(labels[0])}')

    # List containing the images with normailzed frames added in the 3rd dimension
    expanded_images = []

    # Perform frame normalization using every two adjacent frames as (c(t + 1) - c(t))/(c(t) + c(t + 1))
    # where c is the channel of the frame.
    for idx, image in enumerate(images):
        if idx < len(images) - 1:
            for i in range(3):

                # print(f'[INFO] Shape of Frame {idx}: {(images[idx][:, :, i]).shape}')

                # Displaying the frame at channel i
                # display_frame(images[idx][:, :, i], f'Frame {idx} Channel {i}')   

                # Normalized frame calculated by the formula above
                normalizedFrame = (images[idx + 1][:, :, i] - images[idx][:, :, i]
                ) / (images[idx][:, :, i] + images[idx + 1][:, :, i])

                # Displaying the normalized frame at channel i
                # display_frame(normalizedFrame, f'Normalized Frame {idx} Channel {i}')

                # print(f'[INFO] Shape of Normalized Frame {idx}: {normalizedFrame.shape}')

                # Adding an extra dimension to the normalized frame to make it possible to append to original image
                normalizedFrame = np.expand_dims(normalizedFrame, axis=2)

                image = np.append(image, normalizedFrame, axis=2)
            
            #     print(f'shape of normalizedFrame: {normalizedFrame.shape}')
            #     print(f'shape of image: {image.shape}')

            # print(f'shape of image after going through each channel: {image.shape}')
            
            # Storing the expanded images 
            expanded_images.append(image)


    return np.array(expanded_images), np.array(labels)

    # # Loop over all IDs and read each image in one by one
    # for image_id in range(num_images):
    #     images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))


# %%
def store_many_hdf5(target_dir, subID, images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        target_dir:  path to the directory where the HDF5 file will be stored.
        subID:       subject ID.
        images       images array, (N, W, H, NC) to be stored (where N: number of images, W: width, H: height, NC: number of channels).
        labels       labels array, (N, 1) to be stored

        Returns:
        ----------
        pathToTarget    path to the HDF5 file.
    """

    pathToTarget = os.path.join(target_dir, f"{subID}.h5")

    # Create a new HDF5 file
    file = h5py.File(pathToTarget, "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", data=images
    )
    meta_set = file.create_dataset(
        "labels", data=labels
    )
    file.close()
    
    return pathToTarget

# %%
def read_hdf5(file_path):
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


# %%
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
    # Get the total no. of subjects
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


# %%
# Function that returns the minimum number of frames of all videos in the dataset
def get_min_num_of_frames(globExp):

    # Variables that will store the min no. of frames and the corresponding directory of the subject 
    min_num_files = float('inf')
    folder_with_min_num_files = ''

    # Get iterator over different subjects
    imageDirs = iglob(globExp)

    for path_ in imageDirs:

        num_images = len(os.listdir(path_))
        
        if num_images < min_num_files:
            min_num_files = num_images
            folder_with_min_num_files = path_

    # print(f'folder with min images: { folder_with_min_num_files} \nfile count:{min_num_files} \n')

    return min_num_files