import os
from shutil import copy2

data_path = 'data\\extracted_images'
train_path = 'data\\math_symbols\\train'
test_path = 'data\\math_symbols\\test'

#5 classes
includedLabels = ['+','-','times',',','forward_slash']
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