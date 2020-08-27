from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
import glob
import os
# model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

# model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models
root = 'E:/Google Drive/Acads/Junior year/Semester 5/Project Course/laneSegmentation/testing/input/*'
for filePath in sorted(glob.glob(root)):
    model.predict_segmentation(
        inp=filePath,
        out_fname="testing/output/" + os.path.basename(filePath)
    )