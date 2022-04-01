# This file contains utility functions for the project.

import numpy as np
import h5py


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