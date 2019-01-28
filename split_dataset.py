import os
from shutil import copy2

data_path = 'data\\extracted_images'
train_path = 'data\\train'
test_path = 'data\\test'

includedLabels = ['1','2','3','4','5','6','7','8','9','0','+','-','times',',','forward_slash']
# 19 classes
os.mkdir(train_path)
os.mkdir(test_path)
labels = os.listdir(data_path)
for label in labels:
    if label in includedLabels:
        path = os.path.join(data_path, label)
        print(os.listdir(path))
        numfiles = len(os.listdir(path))
        print(numfiles)
        threshold = 80/100 * numfiles
        print(threshold)
        iterator = 0
        os.mkdir(os.path.join(train_path, label))
        os.mkdir(os.path.join(test_path, label))

        for file in os.listdir(path):
            iterator += 1
            print(file)
            srcpath = os.path.join(path, file)
            if (iterator < threshold):
                dstpath = os.path.join(train_path, label)
            else:
                dstpath = os.path.join(test_path, label)

            copy2(srcpath, dstpath)