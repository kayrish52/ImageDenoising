from calcWavelet import calcWavelet as cw
from dataset import Dataset
import easydict
import matplotlib.pyplot as plt
from models import DnCNN
from noise_generator import noise_generator
import os
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
        "batchSize": 50,
        "num_of_layers": 4,
        "epochs": 10,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
savePath = "test"
noiseMethod = 6


def main(opt, savePath, noiseMethod):
# def main():

    # prepare_data(data_path=dPath, trSamples=100, valSamples=20)

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

            # initialize the training parameters
            temp1 = []
            trainPSNR1 = []
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = np.expand_dims(data.numpy(), 1)

            # generate and add noise to the training images
            for j in np.arange(0, img_train.shape[0]):
                img_train_np = img_train[j, :, :, :]
                img_train_np = np.transpose(np.clip(img_train_np, 0.00001, 1.0), (2, 1, 0))
                noise = noise_generator(img_train_np, opt.noiseL, noiseMethod)

                # generate wavelets for training images
                imgW = cw(img_train_np, method='haar')
                imgW = np.expand_dims(imgW, 1)
                imgW = np.transpose(imgW, (0, 3, 2, 1))

                # generate wavelets for noise
                imgWN = cw(noise, method='haar')
                imgWN = np.expand_dims(imgWN, 1)
                imgWN = np.transpose(imgWN, (0, 3, 2, 1))

                # stack wavelets into training set
                if j == 0:
                    gt_wavelets = imgW
                    noiseW = imgWN
                    img_train_wavelets = imgW + imgWN
                else:
                    gt_wavelets = np.vstack((gt_wavelets, imgW))
                    noiseW = np.vstack((noiseW, imgWN))
                    img_train_wavelets = np.vstack((img_train_wavelets, (imgW + imgWN)))

            gt_wavelets = torch.from_numpy(gt_wavelets).float()
            img_train_wavelets = torch.from_numpy(img_train_wavelets).float()
            noiseW = torch.from_numpy(noiseW).float()

            # convert images to GPU variables
            gt_wavelets = Variable(gt_wavelets.cuda())
            img_train_wavelets = Variable(img_train_wavelets.cuda())
            noiseW = Variable(noiseW.cuda())

            # train model
            img_train_wavelets = torch.transpose(img_train_wavelets, 1, 3)
            noiseW = torch.transpose(noiseW, 1, 3)
            out_train = model(img_train_wavelets)
            loss = criterion(out_train, noiseW) / (img_train_wavelets.size()[0]*2)
            temp1.append(loss.item())

            # optimization step
            loss.backward()
            optimizer.step()

            # results
            model.eval()
            out_train = torch.clamp(img_train_wavelets-model(img_train_wavelets), 0., 1.)

            # calculate the PSNR of the output image
            gt_wavelets = torch.transpose(gt_wavelets, 1, 3)
            psnr_train = batch_psnr(out_train, gt_wavelets, 1.)
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
            img_val = torch.unsqueeze(dataset_val[k], 0)
            img_val_np = img_val[:, :, :].numpy()
            img_val_np = np.transpose(np.clip(img_val_np, 0.00001, 1.0), (2, 1, 0))
            noise = noise_generator(img_val_np, opt.noiseL, noiseMethod)

            # generate wavelets for training images
            imgW_val = cw(img_val_np, method='haar')
            imgW_val = np.expand_dims(imgW_val, 1)
            imgW_val = np.transpose(imgW_val, (0, 3, 2, 1))

            # generate wavelets for noise
            imgWN_val = cw(noise, method='haar')
            imgWN_val = np.expand_dims(imgWN_val, 1)
            imgWN_val = np.transpose(imgWN_val, (0, 3, 2, 1))

            # stack wavelets into training set
            gt_wavelets_val = imgW_val
            noiseW_val = imgWN_val
            img_val_wavelets = imgW_val + imgWN_val

            # convert the numpy arrays into torch tensors
            gt_wavelets_val = torch.from_numpy(gt_wavelets_val).float()
            img_val_wavelets = torch.from_numpy(img_val_wavelets).float()
            noiseW_val = torch.from_numpy(noiseW_val).float()

            # convert images to GPU variables
            gt_wavelets_val = Variable(gt_wavelets_val.cuda())
            img_val_wavelets = Variable(img_val_wavelets.cuda())
            noiseW_val = Variable(noiseW_val.cuda())
            noiseW_val = torch.transpose(noiseW_val, 1, 3)

            # perform validation runs
            img_val_wavelets = torch.transpose(img_val_wavelets, 1, 3)
            out_val = model(img_val_wavelets)

            # calculate loss
            lossVal = criterion(out_val, noiseW_val) / (img_val_wavelets.size()[0]*2)
            lossVal.backward()
            valLoss1.append(lossVal.item())
            out_val = torch.clamp(img_val_wavelets - model(img_val_wavelets), 0., 1.)
            gt_wavelets_val = torch.transpose(gt_wavelets_val, 1, 3)

            # calculate PSNR of validation
            psnr_val = batch_psnr(out_val, gt_wavelets_val, 1.)
            valPSNR1.append(psnr_val.item())

        print("[epoch %d]  ValLoss: %.4f, PSNR_val: %.4f\n" % (epoch+1, mean(valLoss1), psnr_val))

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
    # plt.tight_layout(h_pad=10, w_pad=10, rect=[0, 0, 0.9, 0.9])
    plt.legend()
    # plt.subplots_adjust(top=0.8)
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
