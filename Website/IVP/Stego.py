# This code logic is from github we have seperated it from the github https://github.com/daniellerch/hstego and we have modified it to work with our project. We extracted the Juniward part from this so to encode images with J-UNIWARD we have to use this code logic.

import numpy as np
import struct
import sys
import glob
import os
from ctypes import *

INF = 2**31-1

base = os.path.dirname(__file__)

# Load STC library
stc_pattern = 'hstego_stc_extension.so'

# PyInstaller creates a temp folder and stores path in _MEIPASS
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base = sys._MEIPASS


# Try to find the library
stc_candidates = glob.glob(os.path.join(base, stc_pattern))

# if STC library is not found, try to find it in the build directory


if not stc_candidates and sys.platform == "linux": # devel mode
    stc_candidates = glob.glob('build/lib.linux/'+stc_pattern)
if not stc_candidates and sys.platform == "win32": # devel mode
    stc_candidates = glob.glob('build/lib.win*/'+stc_pattern)
if not stc_candidates:
    print("STC library not found:", base)
    sys.exit(0)


# Load the library
stc = CDLL(stc_candidates[0])


# Stego class
class Stego:
    def __init__(self):
        pass

    def bytes_to_bits(self, data):
        array=[]
        for b in data:
            for i in range(8):
                array.append((b >> i) & 1)
        return array

    # Hide a message in a cover image
    def hide_stc(self, cover_array, costs_array, message_bits, mx=255, mn=0):
        cover = (c_int*(len(cover_array)))()
        for i in range(len(cover_array)):
            cover[i] = int(cover_array[i])

        # Prepare costs
        costs = (c_float*(len(costs_array)*3))()
        for i in range(len(costs_array)):
            if cover[i]<=mn:
                costs[3*i+0] = INF
                costs[3*i+1] = 0
                costs[3*i+2] = costs_array[i]
            elif cover[i]>=mx:
                costs[3*i+0] = costs_array[i]
                costs[3*i+1] = 0 
                costs[3*i+2] = INF
            else:
                costs[3*i+0] = costs_array[i]
                costs[3*i+1] = 0
                costs[3*i+2] = costs_array[i]


        m = len(message_bits)
        message = (c_ubyte*m)()
        for i in range(m):
            message[i] = message_bits[i]

        # Hide message
        stego = (c_int*(len(cover_array)))()
        _ = stc.stc_hide(len(cover_array), cover, costs, m, message, stego)

        # stego data to numpy
        stego_array = cover_array.copy()
        for i in range(len(cover_array)):
            stego_array[i] = stego[i]
     
        return stego_array


    def hide(self, message, cover_matrix, cost_matrix, mx=255, mn=0):
        # Convert message to bits
        message_bits = self.bytes_to_bits(message)


        height, width = cover_matrix.shape
        cover_array = cover_matrix.reshape((height*width,)) 
        costs_array = cost_matrix.reshape((height*width,)) 

        # Hide data_len (32 bits) into 64 pixels (0.5 payload)
        data_len = struct.pack("!I", len(message_bits))
        data_len_bits = self.bytes_to_bits(data_len)

        stego_array_1 = self.hide_stc(cover_array[:64], costs_array[:64], data_len_bits, mx, mn)
        stego_array_2 = self.hide_stc(cover_array[64:], costs_array[64:], message_bits, mx, mn)
        stego_array = np.hstack((stego_array_1, stego_array_2))

        stego_matrix = stego_array.reshape((height, width))
        
        return stego_matrix
