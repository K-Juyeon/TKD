# import torch
# print(torch.cuda.get_device_name(0))

import os
from tqdm import tqdm
import shutil

dataset_path = "E:\\tkd_data\\splitdata"

folderlist = os.listdir(dataset_path)

# shutil.rmtree(dataset_path)

'''
    home/data/splitdata/class1/train/img1.jpg
    home/data/splitdata/class1/test/img1.jpg
    home/data/splitdata/class1/validation/img1.jpg
    을 
    home/data/splitdata/train/class1/img1.jpg
    home/data/splitdata/test/class1/img1.jpg
    home/data/splitdata/validation/class1/img1.jpg
    형태로 변경
'''

for folder in folderlist :
    if "test" in folder or "train" in folder or "validation" in folder :
        pass
    else :
        for paths, dirs, files in os.walk(os.path.join(dataset_path, folder)):
            filelist = os.listdir(paths)
            filelist_jpg = [file for file in filelist if file.endswith(".jpg")]
            if len(filelist) == 0 :
                os.removedirs(paths)
for paths, dirs, files in os.walk(dataset_path) :
    filelist = os.listdir(paths)
    filelist_jpg = [file for file in filelist if file.endswith(".jpg")]
    twostep_back = paths.split('\\')[-2] # train set
    onestep_back = paths.split('\\')[-3] # 기본 준비
    for file in tqdm(filelist_jpg, desc='Processing...') :
        if 'train' in twostep_back :
            if not os.path.isdir(os.path.join("E:\\tkd_data\\splitdata_1\\train", onestep_back)) :
                os.makedirs(os.path.join("E:\\tkd_data\\splitdata_1\\train", onestep_back))
            # print(os.path.join("E:\\tkd_data\\splitdata\\train", onestep_back, file))
            shutil.move(os.path.join(paths, file), os.path.join("E:\\tkd_data\\splitdata_1\\train", onestep_back, file))
        elif 'test' in twostep_back :
            if not os.path.isdir(os.path.join("E:\\tkd_data\\splitdata_1\\test", onestep_back)) :
                os.makedirs(os.path.join("E:\\tkd_data\\splitdata_1\\test", onestep_back))
            # print(os.path.join("E:\\tkd_data\\splitdata\\test", onestep_back, file))
            shutil.move(os.path.join(paths, file), os.path.join("E:\\tkd_data\\splitdata_1\\test", onestep_back, file))
        elif 'val' in twostep_back:
            if not os.path.isdir(os.path.join("E:\\tkd_data\\splitdata_1\\validation", onestep_back)) :
                os.makedirs(os.path.join("E:\\tkd_data\\splitdata_1\\validation", onestep_back))
            # print(os.path.join("E:\\tkd_data\\splitdata\\validation", onestep_back, file))
            shutil.move(os.path.join(paths, file), os.path.join("E:\\tkd_data\\splitdata_1\\validation", onestep_back, file))