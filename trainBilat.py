from dataset import Dataset
import easydict
import matplotlib.pyplot as plt
from models import DnCNN
from noise_generator import noise_generator
import os
from skimage.restoration import denoise_bilateral as bilateral
from statistics import mean
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *

dPath = "/media/kayrish52/DataStorage/NWPU"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 60,
        "num_of_layers": 4,
        "epochs": 30,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
savePath = "test"
noiseMethod = 2


def main(opt, savePath, noiseMethod):
# def main():

    # prepare_data(data_path=dPath, trSamples=200, valSamples=40)

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    step = 0
    temp = []
    trainPSNR = []
    valPSNR = []
    valLoss = []

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):

            # training step

            temp1 = []
            trainPSNR1 = []
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = np.expand_dims(data.numpy(), 1)
            img_prefiltered = np.zeros(img_train.shape)

            # generate and add noise to the training images
            for j in np.arange(0, img_train.shape[0]):
                img_train_np = img_train[j, :, :, :]

                noise = noise_generator(img_train_np, opt.noiseL, noiseMethod)
                img_train_np = img_train_np + noise
                img_train_np = np.clip(img_train_np, 0.00001, 1.0)

                # prefilter the images
                img_prefiltered[j, :, :, :] = np.array(bilateral(np.float64(img_train_np[0, :, :]), multichannel=False))

            # convert prefiltered images to tensors
            img_prefiltered = torch.from_numpy(img_prefiltered).float()

            # convert images to GPU variables
            img_train = Variable(torch.from_numpy(img_train).cuda())
            img_prefiltered = Variable(img_prefiltered.cuda())
            noise1 = Variable(torch.from_numpy(noise).cuda())

            # train the model
            out_train = model(img_prefiltered)
            loss = criterion(out_train.float(), noise1.float()) / (img_prefiltered.size()[0]*2)
            temp1.append(loss.item())

            # optimization step
            loss.backward()
            optimizer.step()

            # results
            model.eval()
            out_train = torch.clamp(img_prefiltered-model(img_prefiltered), 0., 1.)

            # calculate the PSNR of the output image
            psnr_train = batch_psnr(out_train, img_train, 1.)
            trainPSNR1.append(psnr_train.item())
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % (epoch+1, i+1, len(loader_train), loss.item(),
                                                                     psnr_train))

            step += 1
        # the end of each epoch

        model.eval()

        # validate
        psnr_val = 0
        valPSNR1 = []
        valLoss1 = []

        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0).numpy()
            noise = noise_generator(img_val[0, :, :], opt.val_noiseL, noiseMethod)
            img_noise_val = img_val[0, :, :] + noise

            # prefilter the images
            inputImg = np.clip(np.array(img_noise_val, dtype='float64'), 0.00001, 1)
            img_val_prefiltered = np.zeros((1, img_noise_val.shape[0], img_noise_val.shape[1]))
            img_val_prefiltered[0, :, :] = np.array(bilateral(inputImg, multichannel=False))

            # convert variables to GPU variables
            img_val = np.expand_dims(img_val, 0)
            img_val = Variable(torch.from_numpy(img_val).cuda())
            img_val_prefiltered = np.expand_dims(img_val_prefiltered, 0)
            img_val_prefiltered = Variable(torch.from_numpy(img_val_prefiltered).float().cuda())
            noise1 = Variable(torch.from_numpy(noise).cuda())

            # pass the prefiltered image through the model
            out_val = model(img_val_prefiltered)
            lossVal = criterion(out_val.float(), noise1.float()) / (img_val_prefiltered.size()[0]*2)
            lossVal.backward()
            valLoss1.append(lossVal.item())

            # calculate the residual and the PSNR
            out_val = torch.clamp(img_val_prefiltered - model(img_val_prefiltered), 0., 1.)
            psnr_val = batch_psnr(out_val, img_val, 1.)
            valPSNR1.append(psnr_val.item())

        print("[epoch %d]   ValLoss: %.4f, PSNR_val: %.4f\n" % (epoch+1, mean(valLoss1), psnr_val))

        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        # out_train = torch.clamp(img_noise_train-model(img_noise_train), 0., 1.)
        # Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        # Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        # Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, savePath, 'net.pth'))
        temp.append(mean(temp1))
        trainPSNR.append(mean(trainPSNR1))
        valPSNR.append(mean(valPSNR1))
        valLoss.append(mean(valLoss1))

    plt.subplot(121)
    plt.plot(np.arange(0, opt.epochs), temp, label="Training Loss")
    plt.plot(np.arange(0, opt.epochs), valLoss, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Value per Epoch')
    plt.legend()
    plt.subplot(122)
    plt.plot(np.arange(0, opt.epochs), trainPSNR, label="Training PSNR")
    plt.plot(np.arange(0, opt.epochs), valPSNR, label="Validation PSNR")
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR per Epoch')
    plt.legend()
    plt.savefig(os.path.join(opt.outf, savePath, 'Training'))
    # plt.show()


if __name__ == "__main__":
    # prepare_data(data_path=dPath, patch_size=40, stride=10, aug_times=1)
    # if opt.preprocess:
    #     if opt.mode == 'S':
    #         prepare_data(data_path='../dataset/NWPU', patch_size=40, stride=10, aug_times=1)
    #     if opt.mode == 'B':
    #         prepare_data(data_path='../dataset/NWPU', patch_size=50, stride=10, aug_times=2)
    main(opt, savePath, noiseMethod)
