import cv2
import numpy as np
from skimage.measure.simple_metrics import compare_psnr


def noise_generator(img, noiseL, noiseMethod):
    if noiseMethod == 0:
        noise = 0.0001*np.random.normal(0, 1, img.shape)
    elif noiseMethod == 1:
        noise = agn(img, noiseL)
    elif noiseMethod == 2:
        noise = mng(img, 21, 11)
    elif noiseMethod == 3:
        noise = gsg(img)
    elif noiseMethod == 4:
        noise = agn(img, noiseL) + mng(img, 21, 11)
    elif noiseMethod == 5:
        noise = agn(img, noiseL) + gsg(img)
    elif noiseMethod == 6:
        noise = agn(img, noiseL) + gsg(img) + mng(img, 21, 11)
    else:
        noise = 0.0001*np.random.normal(0, 1, img.shape)

    return noise


def agn(img, noiseL):
    # additive gaussian noise generation
    additive_gaussian_noise = np.random.normal(0, noiseL/100, img.shape)

    return additive_gaussian_noise


def mng(img, degree, angle):
    # create motion kernel
    m = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    # blur the image
    blurred_numpy = cv2.filter2D(img, -1, motion_blur_kernel, borderType=cv2.BORDER_REPLICATE)
    motion_noise = blurred_numpy - img

    return motion_noise


def gsg(img):
    # create motion kernel
    m = cv2.getGaussianKernel(5, 2)

    # blur the image
    blurred_numpy = cv2.filter2D(img, -1, m, borderType=cv2.BORDER_REPLICATE)
    gaussian_smoothing_noise = blurred_numpy - img

    return gaussian_smoothing_noise


def mngPSF(degree=21, angle=11):
    # create motion kernel
    cPoint = int((degree-1)/2)
    m = cv2.getRotationMatrix2D((cPoint, cPoint), angle, 1)
    motion_blur_kernel = np.zeros((degree, degree))
    motion_blur_kernel[cPoint, :] = 1
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    psf = motion_blur_kernel

    return psf


def gsgPSF():
    # create motion kernel
    m = cv2.getGaussianKernel(5, 2)

    return m


def batch_psnr(img, imclean, data_range):
    Img = img.astype(np.float32)
    Iclean = imclean.astype(np.float32)
    PSNR = np.zeros((img.shape[0], 1))
    for j in np.arange(0, img.shape[0]):
        PSNR[j] = compare_psnr(Iclean[j, :, :, :], Img[j, :, :, :], data_range=data_range)
    return PSNR
