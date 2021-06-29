from calcWavelet import calcWavelet
from calcWavelet import calcIWavelet
import os.path
import torch
import torch.nn as nn
from torch.autograd import Variable
import easydict
from models import DnCNN
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from noise_generator import noise_generator
from noise_generator import batch_psnr as nPSNR
from noise_generator import mngPSF
from noise_generator import gsgPSF
from skimage.color import rgb2gray
from skimage.restoration import denoise_bilateral as bilateral
from skimage.restoration import wiener


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotion/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotion/Lena4BM.png"
noiseMethod = 2
testMethod = 'Wiener'
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})


def main(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod):

    # Build Model and Move to GPU
    torch.cuda.empty_cache()
    model = nn.DataParallel(DnCNN(1, num_of_layers=opt.num_of_layers)).cuda()

    # Load the Model
    model.load_state_dict(torch.load(networkPath))
    model.eval()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # read the image
    img = np.array(io.imread(imgFile))/255
    img = rgb2gray(img)
    noise = noise_generator(img, opt.noiseL, noiseMethod)
    imgNoise = img + noise
    imgNoise = np.clip(np.array(imgNoise, dtype='float64'), 0.00001, 1)

    # test the image
    if testMethod == 'CNN':
        # convert images to torch images and move to GPU
        imgNoise = Variable(torch.from_numpy(np.expand_dims(np.expand_dims(imgNoise, 0), 0)).float().cuda())

        # pass the prefiltered image through the model
        residualCNN = model(imgNoise).cpu().detach().numpy()

        # calculate the denoised image and the PSNR
        outCNN = imgNoise.cpu().numpy() - residualCNN
        imgGT = np.expand_dims(np.expand_dims(img, 0), 0)
        outPSNR = nPSNR(outCNN, imgGT, 1.)
        imgNPSNR = nPSNR(imgNoise.cpu().numpy(), imgGT, 1.)

        # generate full image plots
        fig = plt.figure()
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(img, cmap='gray')
        plt.subplot(132)
        plt.title("Noisy Image\nPSNR = %4.2f" % imgNPSNR)
        plt.imshow(imgNoise[0, 0, :, :].cpu().numpy(), cmap='gray')
        plt.subplot(133)
        plt.imshow(outCNN[0, 0, :, :], cmap='gray')
        plt.title("Prefiltered CNN Denoised\nPSNR = %4.2f" % outPSNR)
        plt.tight_layout()
        plt.savefig(saveFile + "_CNN.png")
        # plt.show()

        print("CNN Processing Complete")
    elif testMethod == 'Bilateral':
        # prefilter the image
        imgPrefiltered = np.array(bilateral(imgNoise, multichannel=False))
        imgPrefiltered = np.expand_dims(np.expand_dims(imgPrefiltered, 0), 0)

        # convert images to torch images and move to GPU
        imgPrefiltered = Variable(torch.from_numpy(imgPrefiltered).float().cuda())

        # pass the prefiltered image through the model
        residualPrefiltered = model(imgPrefiltered).cpu().detach().numpy()

        # calculate the denoised image and the PSNR
        outPrefiltered = imgPrefiltered.cpu().numpy() - residualPrefiltered
        imgGT = np.expand_dims(np.expand_dims(img, 0), 0)
        imgNoise = np.expand_dims(np.expand_dims(imgNoise, 0), 0)
        outPSNR = nPSNR(outPrefiltered, imgGT, 1.)
        imgNPSNR = nPSNR(imgNoise, imgGT, 1.)

        # generate full image plots
        fig = plt.figure()
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(img, cmap='gray')
        plt.subplot(132)
        plt.title("Noisy Image\nPSNR = %4.2f" % imgNPSNR)
        plt.imshow(imgNoise[0, 0, :, :], cmap='gray')
        plt.subplot(133)
        plt.imshow(outPrefiltered[0, 0, :, :], cmap='gray')
        plt.title("Prefiltered CNN Denoised\nPSNR = %4.2f" % outPSNR)
        plt.tight_layout()
        plt.savefig(saveFile + "_Bilateral.png")
        # plt.show()

        print("Prefilter Processing Complete")
    elif testMethod == 'Wavelets':
        # create wavelets
        imgGTW = calcWavelet(np.expand_dims(img, 2))
        imgGTW = np.expand_dims(imgGTW, 1)
        imgW = calcWavelet(np.expand_dims(imgNoise, 2))
        imgW = np.expand_dims(imgW, 1)

        # convert wavelets to torch images and move to GPU
        imgW = Variable(torch.from_numpy(imgW).float().cuda())

        # predict the residuals per wavelet
        imgWRes = model(imgW).cpu().detach().numpy()
        outW = imgW.cpu().numpy() - imgWRes
        imgWPSNR = nPSNR(imgW.cpu().numpy(), imgGTW, 1.)
        outWPSNR = nPSNR(outW, imgGTW, 1.)

        # recombine the images
        outRecon = calcIWavelet(outW)
        outReconPSNR = nPSNR(np.expand_dims(np.expand_dims(outRecon, 0), 0),
                             np.expand_dims(np.expand_dims(img, 0), 0), 1.)
        imgNPSNR = nPSNR(np.expand_dims(np.expand_dims(imgNoise, 0), 0),
                         np.expand_dims(np.expand_dims(img, 0), 0), 1.)

        # generate full image plots
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.subplot(132)
        plt.imshow(imgNoise, cmap='gray')
        plt.title("Noisy Image\nPSNR = %4.2f" % imgNPSNR)
        plt.subplot(133)
        plt.imshow(outRecon, cmap='gray')
        plt.title("Wavelet CNN Reconstructed\nPSNR = %4.2f" % outReconPSNR)
        plt.tight_layout()
        plt.savefig(saveFile + "_Wavelets_1.png")
        # plt.show()

        # generate ground truth wavelet plots
        fig = plt.figure()
        plt.subplot(221)
        plt.title("Original Image\nLL Wavelet")
        plt.imshow(imgGTW[0, 0, :, :], cmap='gray')
        plt.subplot(222)
        plt.title("Original Image\nLH Wavelet")
        plt.imshow(imgGTW[1, 0, :, :], cmap='gray')
        plt.subplot(223)
        plt.title("Original Image\nHL Wavelet")
        plt.imshow(imgGTW[2, 0, :, :], cmap='gray')
        plt.subplot(224)
        plt.title("Original Image\nHH Wavelet")
        plt.imshow(imgGTW[3, 0, :, :], cmap='gray')
        plt.tight_layout()
        plt.savefig(saveFile + "_Wavelets_2.png")
        # plt.show()

        # generate noisy wavelet plots
        fig = plt.figure()
        plt.subplot(221)
        plt.title("Noisy Image\nLL Wavelet\nPSNR = %4.2f" % imgWPSNR[0])
        plt.imshow(imgW[0, 0, :, :].cpu().numpy(), cmap='gray')
        plt.subplot(222)
        plt.title("Noisy Image\nLH Wavelet\nPSNR = %4.2f" % imgWPSNR[1])
        plt.imshow(imgW[1, 0, :, :].cpu().numpy(), cmap='gray')
        plt.subplot(223)
        plt.title("Noisy Image\nHL Wavelet\nPSNR = %4.2f" % imgWPSNR[2])
        plt.imshow(imgW[2, 0, :, :].cpu().numpy(), cmap='gray')
        plt.subplot(224)
        plt.title("Noisy Image\nHH Wavelet\nPSNR = %4.2f" % imgWPSNR[3])
        plt.imshow(imgW[3, 0, :, :].cpu().numpy(), cmap='gray')
        plt.tight_layout()
        plt.savefig(saveFile + "_Wavelets_3.png")
        # plt.show()

        # generate wavelet CNN plots
        fig = plt.figure()
        plt.subplot(221)
        plt.title("Wavelet CNN Denoised\nLL Wavelet\nPSNR = %4.2f" % outWPSNR[0])
        plt.imshow(outW[0, 0, :, :], cmap='gray')
        plt.subplot(222)
        plt.title("Wavelet CNN Denoised\nLH Wavelet\nPSNR = %4.2f" % outWPSNR[1])
        plt.imshow(outW[1, 0, :, :], cmap='gray')
        plt.subplot(223)
        plt.title("Wavelet CNN Denoised\nHL Wavelet\nPSNR = %4.2f" % outWPSNR[2])
        plt.imshow(outW[2, 0, :, :], cmap='gray')
        plt.subplot(224)
        plt.title("Wavelet CNN Denoised\nHH Wavelet\nPSNR = %4.2f" % outWPSNR[3])
        plt.imshow(outW[3, 0, :, :], cmap='gray')
        plt.tight_layout()
        plt.savefig(saveFile + "_Wavelets_4.png")
        # plt.show()

        print("Wavelet Processing Complete")
    else:
        # Get PSF
        if noiseMethod == 2:
            psf = mngPSF(21, 11)
        else:
            psf = gsgPSF()

        # Apply Wiener Filter
        balance = 30
        imgWNR = wiener(imgNoise, psf, balance)

        # calculate the noise
        imgNPSNR = nPSNR(np.expand_dims(np.expand_dims(imgNoise, 0), 0),
                         np.expand_dims(np.expand_dims(img, 0), 0), 1.)
        imgWNRPSNR = nPSNR(np.expand_dims(np.expand_dims(imgWNR, 0), 0),
                           np.expand_dims(np.expand_dims(img, 0), 0), 1.)

        # plot Wiener denoised image
        fig = plt.figure()
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(img, cmap='gray')
        plt.subplot(132)
        plt.title("Noisy Image\nPSNR = %4.2f" % imgNPSNR)
        plt.imshow(imgNoise, cmap='gray')
        plt.subplot(133)
        plt.title("Wiener Denoised Image\nPSNR = %4.2f" % imgWNRPSNR)
        plt.imshow(imgWNR, cmap='gray')
        plt.tight_layout()
        plt.savefig(saveFile + "_Wiener.png")
        # plt.show()

        print("Wiener Processing Complete")

    return


if __name__ == "__main__":
    main(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
