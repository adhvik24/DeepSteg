# This code logic is from github we have seperated it from the github https://github.com/daniellerch/hstego and we have modified it to work with our project and appropriately commented our understandings. We extracted the Juniward part from this so to encode images with J-UNIWARD we have to use this code logic.

import numpy as np
import os
import sys
import glob
import copy
from ctypes import *

# Load JPEG Toolbox library
base = os.path.dirname(__file__)
jpg_pattern = 'hstego_jpeg_toolbox_extension.so'

# If we are in a PyInstaller bundle, the library is in sys._MEIPASS
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base = sys._MEIPASS
    
jpg_candidates = glob.glob(os.path.join(base, jpg_pattern))

# if JPEG Toolbox library is not found, try to find it in the build directory
if not jpg_candidates and sys.platform == "linux": # devel mode
    jpg_candidates = glob.glob('build/lib.linux/'+jpg_pattern)
if not jpg_candidates and sys.platform == "win32": # devel mode
    jpg_candidates = glob.glob('build/lib.win*/'+jpg_pattern)
if not jpg_candidates:
    print("JPEG Toolbox library not found:", base)
    sys.exit(0)

# Load the library
jpeg = CDLL(jpg_candidates[0])


def jpeg_load(path, use_blocks=False):

    if not os.path.isfile(path):
        raise FileNotFoundError

    # we load the JPEG image using the jpeg_toolbox library
    jpeg.write_file.argtypes = c_char_p,
    jpeg.read_file.restype = py_object
    r = jpeg.read_file(path.encode())
    # the quant tables are uint16, so we convert them to numpy arrays
    # we get them from the jpeg library as lists

    r["quant_tables"] = np.array(r["quant_tables"])

    # the ac and dc huffman tables are lists of dictionaries
    # they are used for encoding, so we convert them to numpy arrays

    for i in range(len(r["ac_huff_tables"])):
        r["ac_huff_tables"][i]["counts"] = np.array(r["ac_huff_tables"][i]["counts"])
        r["ac_huff_tables"][i]["symbols"] = np.array(r["ac_huff_tables"][i]["symbols"])

    for i in range(len(r["dc_huff_tables"])):
        r["dc_huff_tables"][i]["counts"] = np.array(r["dc_huff_tables"][i]["counts"])
        r["dc_huff_tables"][i]["symbols"] = np.array(r["dc_huff_tables"][i]["symbols"])

    # the coef_arrays are lists of lists of lists of lists
    if not use_blocks:
        # we convert them to numpy arrays
        chn = len(r["coef_arrays"])
        for c in range(chn):
            r["coef_arrays"][c] = np.array(r["coef_arrays"][c])
            h = r["coef_arrays"][c].shape[0]*8
            w = r["coef_arrays"][c].shape[1]*8
            # we move the axes to get the correct shape
            r["coef_arrays"][c] = np.moveaxis(r["coef_arrays"][c], [0,1,2,3], [0,2,1,3])
            # we reshape the array to get the correct shape
            r["coef_arrays"][c] = r["coef_arrays"][c].reshape((h, w))

    return r


# saving a JPEG image
def jpeg_save(data, path, use_blocks=False):

    # here we use the jpeg_toolbox library to save the JPEG image
    jpeg.write_file.argtypes = py_object,c_char_p
    # the function deepcopies the data, so we don't have to worry about it

    r = copy.deepcopy(data)
    r["quant_tables"] = r["quant_tables"].astype('uint16').tolist()

    for i in range(len(r["ac_huff_tables"])):
        r["ac_huff_tables"][i]["counts"] = r["ac_huff_tables"][i]["counts"].tolist()
        r["ac_huff_tables"][i]["symbols"] = r["ac_huff_tables"][i]["symbols"].tolist()

    for i in range(len(r["dc_huff_tables"])):
        r["dc_huff_tables"][i]["counts"] = r["dc_huff_tables"][i]["counts"].tolist()
        r["dc_huff_tables"][i]["symbols"] = r["dc_huff_tables"][i]["symbols"].tolist()

    # here is same as we did in the assignment

    if not use_blocks:
        chn = len(r["coef_arrays"])
        for c in range(chn):
            h = r["coef_arrays"][c].shape[0]
            w = r["coef_arrays"][c].shape[1]
            r["coef_arrays"][c] = r["coef_arrays"][c].reshape((h//8, 8, w//8, 8))
            r["coef_arrays"][c] = np.moveaxis(r["coef_arrays"][c], [0,1,2,3], [0,2,1,3])
            r["coef_arrays"][c] = r["coef_arrays"][c].tolist()

    jpeg.write_file(r, path.encode())