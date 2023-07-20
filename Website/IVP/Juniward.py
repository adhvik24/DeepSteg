# This code logic is from github we have seperated it from the github https://github.com/daniellerch/hstego and we have modified it to work with our project and appropriately commented our understandings. We extracted the Juniward part from this so to encode images with J-UNIWARD we have to use this code logic.

import numpy as np
import scipy.signal
import scipy.fftpack
import imageio
import sys
from Stego import Stego
from Jpeg import *

INF = 2**31-1
MAX_PAYLOAD = 0.05

class J_UNIWARD:

    # we apply the dct to the cover image
    def dct2(self, a):
        return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho' ), axis=1, norm='ortho')
        
    # we apply the inverse dct to the cover image
    def idct2(self, a):
        return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

    # this function embeds the message into the cover image
    def cost_fn(self, coef_arrays, quant_tables, spatial):

        # hpdf is the high pass filter
        hpdf = np.array([-0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,  
                         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,  
                         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,  
                         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])        

        # sign is the sign of the high pass filter
        sign = np.array([-1 if i%2 else 1 for i in range(len(hpdf))])
        # lpdf is the low pass filter
        lpdf = hpdf[::-1] * sign


        # F is the wavelet filter
        F = []
        F.append(np.outer(lpdf.T, hpdf))
        F.append(np.outer(hpdf.T, lpdf))
        F.append(np.outer(hpdf.T, hpdf))
        

        # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
        spatial_impact = {}
        for i in range(8):
            for j in range(8):
                test_coeffs = np.zeros((8, 8))
                test_coeffs[i, j] = 1
                # we apply the inverse dct to the test coefficients
                spatial_impact[i, j] = self.idct2(test_coeffs) * quant_tables[i, j]

        # Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
        wavelet_impact = {}
        for f_index in range(len(F)):
            for i in range(8):
                for j in range(8):
                    #we apply the wavelet transform to the spatial impact
                    # and we correlate it with the wavelet filter
                    wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.) # XXX


        # Create reference cover wavelet coefficients (LH, HL, HH)
        pad_size = 16 # XXX
        spatial_padded = np.pad(spatial, (pad_size, pad_size), 'symmetric')


        # we correlate the padded spatial with the wavelet filter
        RC = []
        for i in range(len(F)):
            f = scipy.signal.correlate2d(spatial_padded, F[i], mode='same', boundary='fill')
            RC.append(f)

        # compute the number of non zero AC coefficients
        coeffs = coef_arrays
        k, l = coeffs.shape
        nzAC = np.zeros((k, l))
        # rho is for the cost function
        rho = np.zeros((k, l))
        tempXi = [0.]*3
        sgm = 2**(-6)

        # Computation of costs
        for row in range(k):
            for col in range(l):
                # we compute the number of non zero AC coefficients
                mod_row = row % 8
                mod_col = col % 8
                sub_rows = list(range(row-mod_row-6+pad_size-1, row-mod_row+16+pad_size))
                sub_cols = list(range(col-mod_col-6+pad_size-1, col-mod_col+16+pad_size))

                # here in this loop we compute the cost function
                # we compute the cost function for each channel
                # we compute the cost function for each subband
                for f_index in range(3):
                    RC_sub = RC[f_index][sub_rows][:,sub_cols]
                    wav_cover_stego_diff = wavelet_impact[f_index, mod_row, mod_col]
                    tempXi[f_index] = abs(wav_cover_stego_diff) / (abs(RC_sub)+sgm)

                rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
                rho[row, col] = np.sum(rho_temp)


        rho[np.isnan(rho)] = INF
        rho[rho>INF] = INF

        return rho

    def embed(self, input_img_path, msg_file_path, output_img_path):
        with open(msg_file_path, 'rb') as f:
            data = f.read()

        I = imageio.imread(input_img_path)
        # jpeg_load is a function that loads the jpeg image
        jpg = jpeg_load(input_img_path)

        # we check if the image is in grayscale or in color
        n_channels = 3
        if len(I.shape) == 2:
            n_channels = 1
            I = I[..., np.newaxis]


        message = open(msg_file_path, "rb").read()

        # Real capacity, without headers
        capacity = 0
        for channel in range(len(jpg["coef_arrays"])):
            nz_coeff = np.count_nonzero(jpg["coef_arrays"][channel])
            capacity += int((nz_coeff*MAX_PAYLOAD)/8)

        if len(message) > capacity:
            print("ERROR, message too long:", len(message), ">", capacity)
            sys.exit(0)


        # we create a stego object
        stego = Stego()
    
        if n_channels == 1:
            msg_bits = [ message ] 
        else:
            l = len(data)//3
            msg_bits = [ message[:l], message[l:2*l], message[2*l:] ]
        
        # 
        for c in range(n_channels):
            quant = jpg["quant_tables"][0]
            if c > 2:
                quant = jpg["quant_tables"][1]

            # we compute the cost function and hide the message
            cost = self.cost_fn(jpg["coef_arrays"][c], quant, I[:,:,c])
            jpg["coef_arrays"][c] = stego.hide(msg_bits[c], jpg["coef_arrays"][c], 
                                               cost, mx=1016, mn=-1016)

        # we save the stego image
        jpeg_save(jpg, output_img_path)