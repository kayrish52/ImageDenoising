import numpy as np
import pywt


def calcWavelet(img, method='haar'):
    # calculate the wavelets
    (wA, (wH, wV, wD)) = pywt.dwt2(img[:, :, 0], method)

    # append the wavelets into a stack
    imgW = np.stack((wA, wH, wV, wD), 0)

    return imgW


def calcIWavelet(w, method='haar'):
    return pywt.idwt2((w[0, 0, :, :], (w[1, 0, :, :], w[2, 0, :, :], w[3, 0, :, :])), method)

