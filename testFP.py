import os.path
import easydict
from testImage import main as test

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkNoNoise/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkNoNoise/Lena"
noiseMethod = 0
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkAdditive/Lena"
noiseMethod = 1
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotion/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotion/Lena"
noiseMethod = 2
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothing/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothing/Lena"
noiseMethod = 3
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotionPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotionPlusAdditive/Lena"
noiseMethod = 4
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothingPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothingPlusAdditive/Lena"
noiseMethod = 5
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkNoNoise/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkNoNoise/Lena"
noiseMethod = 0
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkAdditive/Lena"
noiseMethod = 1
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotion/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotion/Lena"
noiseMethod = 2
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothing/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothing/Lena"
noiseMethod = 3
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotionPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotionPlusAdditive/Lena"
noiseMethod = 4
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothingPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothingPlusAdditive/Lena"
noiseMethod = 5
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkNoNoise/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkNoNoise/Lena"
noiseMethod = 0
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkAdditive/Lena"
noiseMethod = 1
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotion/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotion/Lena"
noiseMethod = 2
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothing/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothing/Lena"
noiseMethod = 3
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotionPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotionPlusAdditive/Lena"
noiseMethod = 4
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothingPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothingPlusAdditive/Lena"
noiseMethod = 5
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkNoNoise/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkNoNoise/Cameraman"
noiseMethod = 0
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkAdditive/Cameraman"
noiseMethod = 1
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotion/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotion/Cameraman"
noiseMethod = 2
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothing/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothing/Cameraman"
noiseMethod = 3
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotionPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkMotionPlusAdditive/Cameraman"
noiseMethod = 4
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothingPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wavelets/4BlkSmoothingPlusAdditive/Cameraman"
noiseMethod = 5
testMethod = "Wavelets"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkNoNoise/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkNoNoise/Cameraman"
noiseMethod = 0
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkAdditive/Cameraman"
noiseMethod = 1
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotion/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotion/Cameraman"
noiseMethod = 2
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothing/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothing/Cameraman"
noiseMethod = 3
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotionPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkMotionPlusAdditive/Cameraman"
noiseMethod = 4
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothingPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Bilateral/4BlkSmoothingPlusAdditive/Cameraman"
noiseMethod = 5
testMethod = "Bilateral"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkNoNoise/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkNoNoise/Cameraman"
noiseMethod = 0
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkAdditive/Cameraman"
noiseMethod = 1
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotion/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotion/Cameraman"
noiseMethod = 2
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothing/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothing/Cameraman"
noiseMethod = 3
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotionPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkMotionPlusAdditive/Cameraman"
noiseMethod = 4
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothingPlusAdditive/net.pth"
imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/CNN/4BlkSmoothingPlusAdditive/Cameraman"
noiseMethod = 5
testMethod = "CNN"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})
test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)

# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkNoNoise/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkNoNoise/Lena"
# noiseMethod = 0
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkAdditive/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkAdditive/Lena"
# noiseMethod = 1
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotion/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotion/Lena"
# noiseMethod = 2
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothing/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothing/Lena"
# noiseMethod = 3
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotionPlusAdditive/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotionPlusAdditive/Lena"
# noiseMethod = 4
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothingPlusAdditive/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Lena.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothingPlusAdditive/Lena"
# noiseMethod = 5
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkNoNoise/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkNoNoise/Cameraman"
# noiseMethod = 0
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkAdditive/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkAdditive/Cameraman"
# noiseMethod = 1
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotion/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotion/Cameraman"
# noiseMethod = 2
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothing/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothing/Cameraman"
# noiseMethod = 3
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotionPlusAdditive/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkMotionPlusAdditive/Cameraman"
# noiseMethod = 4
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
#
# networkPath = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothingPlusAdditive/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584FinalProject/testImages/Cameraman.png"
# saveFile = "/home/kayrish52/PycharmProjects/5584FinalProject/logs/Wiener/4BlkSmoothingPlusAdditive/Cameraman"
# noiseMethod = 5
# testMethod = "Wiener"
# opt = easydict.EasyDict({
#         "num_of_layers": 4,
#         "noiseL": 20.0,
# })
# test(networkPath, imgFile, opt, saveFile, noiseMethod, testMethod)
