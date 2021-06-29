import trainBilat as trainBilat
import trainWavelet as trainWavelet
import trainCNN as trainCNN
from dataset import prepare_data
import easydict

dPath = "/media/kayrish52/DataStorage/NWPU"
epochs = 40

prepare_data(data_path=dPath, trSamples=100, valSamples=20)

savePath = "Bilateral/4BlkNoNoise"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 0
trainBilat.main(opt, savePath, noiseMethod)

savePath = "Bilateral/4BlkAdditive"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 1
trainBilat.main(opt, savePath, noiseMethod)

savePath = "Bilateral/4BlkMotion"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 2
trainBilat.main(opt, savePath, noiseMethod)

savePath = "Bilateral/4BlkSmoothing"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 3
trainBilat.main(opt, savePath, noiseMethod)

savePath = "Bilateral/4BlkMotionPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 50,
        "num_of_layers": 4,
        "epochs": epochs,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 4
trainBilat.main(opt, savePath, noiseMethod)

savePath = "Bilateral/4BlkSmoothingPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 50,
        "num_of_layers": 4,
        "epochs": epochs,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 5
trainBilat.main(opt, savePath, noiseMethod)

savePath = "Wavelets/4BlkNoNoise"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 0
trainWavelet.main(opt, savePath, noiseMethod)

savePath = "Wavelets/4BlkAdditive"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 1
trainWavelet.main(opt, savePath, noiseMethod)

savePath = "Wavelets/4BlkMotion"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 2
trainWavelet.main(opt, savePath, noiseMethod)

savePath = "Wavelets/4BlkSmoothing"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 3
trainWavelet.main(opt, savePath, noiseMethod)

savePath = "Wavelets/4BlkMotionPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 50,
        "num_of_layers": 4,
        "epochs": epochs,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 4
trainWavelet.main(opt, savePath, noiseMethod)

savePath = "Wavelets/4BlkSmoothingPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 50,
        "num_of_layers": 4,
        "epochs": epochs,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 5
trainWavelet.main(opt, savePath, noiseMethod)

savePath = "CNN/4BlkNoNoise"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 0
trainCNN.main(opt, savePath, noiseMethod)

savePath = "CNN/4BlkAdditive"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 1
trainCNN.main(opt, savePath, noiseMethod)

savePath = "CNN/4BlkMotion"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 2
trainCNN.main(opt, savePath, noiseMethod)

savePath = "CNN/4BlkSmoothing"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 50,
       "num_of_layers": 4,
       "epochs": epochs,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 3
trainCNN.main(opt, savePath, noiseMethod)

savePath = "CNN/4BlkMotionPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 50,
        "num_of_layers": 4,
        "epochs": epochs,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 4
trainCNN.main(opt, savePath, noiseMethod)

savePath = "CNN/4BlkSmoothingPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 50,
        "num_of_layers": 4,
        "epochs": epochs,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 5
trainCNN.main(opt, savePath, noiseMethod)
