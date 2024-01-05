from os import listdir
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision.datasets.folder import is_image_file
from torchvision.io.image import read_image
import torch.nn.functional as F

import cv2
from utils.datasets import letterbox
import numpy as np

class DatasetFromFolder(data.Dataset):
    def __init__(self, dataset_dir,dataset_type='train'):
        super(DatasetFromFolder, self).__init__()

        # label string list
        label_list = []
        label_list.append("기본준비")
        label_list.append("내려헤쳐막기")
        label_list.append("돌려차고 앞굽이하고 아래막기")
        label_list.append("돌려차고 앞굽이하고 얼굴바깥막고 지르기")
        label_list.append("두발당성차고 앞굽이하고 안막고 두번지르기")
        label_list.append("뒤꼬아서고 두주먹젖혀지르기")
        label_list.append("뒤꼬아서고 등주먹앞치기")
        label_list.append("뒷굽이하고 거들어바깥막기")
        label_list.append("뒷굽이하고 거들어아래막기")
        label_list.append("뒷굽이하고 바깥막기")
        label_list.append("뒷굽이하고 손날거들어바깥막기")
        label_list.append("뒷굽이하고 손날거들어아래막기")
        label_list.append("뒷굽이하고 손날바깥막기")
        label_list.append("뒷굽이하고 안막기")
        label_list.append("뛰어앞차고 앞굽이하고 안막고 두번지르기")
        label_list.append("모아서고 보주먹")
        label_list.append("범서고 바탕손거들어안막고 등주먹앞치기")
        label_list.append("범서고 바탕손안막기")
        label_list.append("범서고 안막기")
        label_list.append("범서고 손날거들어바깥막기")
        label_list.append("앞굽이하고 가위막기")
        label_list.append("앞굽이하고 거들어세워찌르기")
        label_list.append("앞굽이하고 당겨지르기")
        label_list.append("앞굽이하고 두번지르기")
        label_list.append("앞굽이하고 등주먹앞치기")
        label_list.append("앞굽이하고 등주먹앞치기하고 안막기")
        label_list.append("앞굽이하고 바탕손안막고 지르기")
        label_list.append("앞굽이하고 손날얼굴비틀어막기")
        label_list.append("앞굽이하고 아래막고 안막기")
        label_list.append("앞굽이하고 아래막고 지르기")
        label_list.append("앞굽이하고 아래막기")
        label_list.append("앞굽이하고 안막고 두번지르기")
        label_list.append("앞굽이하고 안막기")
        label_list.append("앞굽이하고 얼굴막기")
        label_list.append("앞굽이하고 얼굴바깥막고 지르기")
        label_list.append("앞굽이하고 얼굴지르기")
        label_list.append("앞굽이하고 엇걸어아래막기")
        label_list.append("앞굽이하고 외산틀막기")
        label_list.append("앞굽이하고 제비품안치기")
        label_list.append("앞굽이하고 지르기")
        label_list.append("앞굽이하고 팔꿈치거들어돌려치기")
        label_list.append("앞굽이하고 팔꿈치돌려치고 등주먹앞치기하고, 지르기")
        label_list.append("앞굽이하고 팔꿈치표적치기")
        label_list.append("앞굽이하고 헤쳐막기")
        label_list.append("앞서고 등주먹바깥치기")
        label_list.append("앞서고 손날안치기")
        label_list.append("앞서고 아래막고 지르기")
        label_list.append("앞서고 아래막기")
        label_list.append("앞서고 안막고 지르기")
        label_list.append("앞서고 안막기")
        label_list.append("앞서고 얼굴막기")
        label_list.append("앞서고 지르기")
        label_list.append("앞차고 뒷굽이하고 바깥막기")
        label_list.append("앞차고 범서고 바탕손안막기")
        label_list.append("앞차고 앞굽이하고 등주먹앞치기")
        label_list.append("앞차고 앞굽이하고 아래막고 안막기")
        label_list.append("앞차고 앞굽이하고 지르기")
        label_list.append("앞차고 앞서고 아래막고 지르기")
        label_list.append("앞차고 앞서고 지르기")
        label_list.append("옆서고 메주먹내려치기")
        label_list.append("옆차고 뒷굽이하고 손날거들어바깥막기")
        label_list.append("주춤서고 손날옆막기")
        label_list.append("주춤서고 옆지르기")
        label_list.append("주춤서고 팔꿈치표적치기") 
        
        # data set type list
        data_set_types = []
        data_set_types.append("test set")
        data_set_types.append("training set")
        data_set_types.append("validation set")

        # image file list
        self.image_path_list_S =[]
        self.image_path_list_M =[]
        self.image_path_list_E =[]
        temp_image_path_list_s = []
        # label list
        temp_image_labes = []

        # dataset type
        self.dataset_type_idx = -1

        for idx,val in enumerate(data_set_types):
            if val.find(dataset_type) != -1:
                self.dataset_type_idx = idx

        # check dataset type
        assert self.dataset_type_idx >= 0, 'dataset_type Not Found ' + dataset_type

        for root, subdirs, files in os.walk(dataset_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                

                # find data type
                data_type_idx = -1
                for idx,val in enumerate(data_set_types):
                    if root.find(val) != -1:
                        data_type_idx = idx
                
                if data_type_idx != self.dataset_type_idx: 
                    continue

                sequence_type = filename[-7]
                sequence_num  = filename[-5]

                #if sequence_num == '2':
                #    continue

                if sequence_type == "S":
                    temp_image_path_list_s.append(file_path)
                
                #if sequence_type == "M":
                #    self.image_path_list_M.append(file_path)

                #if sequence_type == "E":
                #    self.image_path_list_E.append(file_path)

        for each_path in temp_image_path_list_s:
            each_path_m_1 = each_path[0:-7] + "M01" + each_path[-4:]
            each_path_m_2 = each_path[0:-7] + "M02" + each_path[-4:]
            each_path_e_1 = each_path[0:-7] + "E01" + each_path[-4:]
            exist_m_1 = os.path.exists(each_path_m_1)
            exist_m_2 = os.path.exists(each_path_m_2)
            exist_e_1 = os.path.exists(each_path_e_1)
            if (exist_m_1 or exist_m_2) and exist_e_1:
                self.image_path_list_S.append(each_path)
                if exist_m_1:
                    self.image_path_list_M.append(each_path_m_1)
                else:
                    self.image_path_list_M.append(each_path_m_2)

                self.image_path_list_E.append(each_path_e_1)

        for each_path in self.image_path_list_S:
            label = each_path
            label = label[len(dataset_dir):-1]
           

            # for linux
            if label.find("/",1) != -1:
                label = label[0:label.find("/",1)]
                
            # for windows
            if label.find("\\",1) != -1:
                label = label[0:label.find("\\",1)]
                
            label = label.replace("\\","")
            label = label.replace("/","")
            label_idx = label_list.index(label)
            temp_image_labes.append(label_idx)
            

        # label list to tensor
        self.image_labels = torch.empty(len(temp_image_labes),dtype=torch.long)
        for i in range(len(temp_image_labes)):
            self.image_labels[i] = temp_image_labes[i]

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # Load Image
        img1 = read_image(self.image_path_list_S[index]).float()
        img2 = read_image(self.image_path_list_M[index]).float()
        img3 = read_image(self.image_path_list_E[index]).float()
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)

        
        try:
            input = torch.cat((img1,img2,img3))
            
        except:
            print('except S path : \n%s\nM path : \n%s\nE path : \n%s\n'%(self.image_path_list_S[index],self.image_path_list_M[index],self.image_path_list_E[index]))
        target = self.image_labels[index]
        return input, target

    def __len__(self):
        return len(self.image_path_list_S)


class DatasetFromFolderPre(data.Dataset):
    def __init__(self, dataset_dir,img_sz=640):
        super(DatasetFromFolderPre, self).__init__()

        # label string list
        label_list = []
        label_list.append("기본준비")
        label_list.append("내려헤쳐막기")
        label_list.append("돌려차고 앞굽이하고 아래막기")
        label_list.append("돌려차고 앞굽이하고 얼굴바깥막고 지르기")
        label_list.append("두발당성차고 앞굽이하고 안막고 두번지르기")
        label_list.append("뒤꼬아서고 두주먹젖혀지르기")
        label_list.append("뒤꼬아서고 등주먹앞치기")
        label_list.append("뒷굽이하고 거들어바깥막기")
        label_list.append("뒷굽이하고 거들어아래막기")
        label_list.append("뒷굽이하고 바깥막기")
        label_list.append("뒷굽이하고 손날거들어바깥막기")
        label_list.append("뒷굽이하고 손날거들어아래막기")
        label_list.append("뒷굽이하고 손날바깥막기")
        label_list.append("뒷굽이하고 안막기")
        label_list.append("뛰어앞차고 앞굽이하고 안막고 두번지르기")
        label_list.append("모아서고 보주먹")
        label_list.append("범서고 바탕손거들어안막고 등주먹앞치기")
        label_list.append("범서고 바탕손안막기")
        label_list.append("범서고 안막기")
        label_list.append("범서고 손날거들어바깥막기")
        label_list.append("앞굽이하고 가위막기")
        label_list.append("앞굽이하고 거들어세워찌르기")
        label_list.append("앞굽이하고 당겨지르기")
        label_list.append("앞굽이하고 두번지르기")
        label_list.append("앞굽이하고 등주먹앞치기")
        label_list.append("앞굽이하고 등주먹앞치기하고 안막기")
        label_list.append("앞굽이하고 바탕손안막고 지르기")
        label_list.append("앞굽이하고 손날얼굴비틀어막기")
        label_list.append("앞굽이하고 아래막고 안막기")
        label_list.append("앞굽이하고 아래막고 지르기")
        label_list.append("앞굽이하고 아래막기")
        label_list.append("앞굽이하고 안막고 두번지르기")
        label_list.append("앞굽이하고 안막기")
        label_list.append("앞굽이하고 얼굴막기")
        label_list.append("앞굽이하고 얼굴바깥막고 지르기")
        label_list.append("앞굽이하고 얼굴지르기")
        label_list.append("앞굽이하고 엇걸어아래막기")
        label_list.append("앞굽이하고 외산틀막기")
        label_list.append("앞굽이하고 제비품안치기")
        label_list.append("앞굽이하고 지르기")
        label_list.append("앞굽이하고 팔꿈치거들어돌려치기")
        label_list.append("앞굽이하고 팔꿈치돌려치고 등주먹앞치기하고, 지르기")
        label_list.append("앞굽이하고 팔꿈치표적치기")
        label_list.append("앞굽이하고 헤쳐막기")
        label_list.append("앞서고 등주먹바깥치기")
        label_list.append("앞서고 손날안치기")
        label_list.append("앞서고 아래막고 지르기")
        label_list.append("앞서고 아래막기")
        label_list.append("앞서고 안막고 지르기")
        label_list.append("앞서고 안막기")
        label_list.append("앞서고 얼굴막기")
        label_list.append("앞서고 지르기")
        label_list.append("앞차고 뒷굽이하고 바깥막기")
        label_list.append("앞차고 범서고 바탕손안막기")
        label_list.append("앞차고 앞굽이하고 등주먹앞치기")
        label_list.append("앞차고 앞굽이하고 아래막고 안막기")
        label_list.append("앞차고 앞굽이하고 지르기")
        label_list.append("앞차고 앞서고 아래막고 지르기")
        label_list.append("앞차고 앞서고 지르기")
        label_list.append("옆서고 메주먹내려치기")
        label_list.append("옆차고 뒷굽이하고 손날거들어바깥막기")
        label_list.append("주춤서고 손날옆막기")
        label_list.append("주춤서고 옆지르기")
        label_list.append("주춤서고 팔꿈치표적치기") 
        
        # data set type list
        data_set_types = []
        data_set_types.append("test set")
        data_set_types.append("training set")
        data_set_types.append("validation set")

        self.image_path_list =[]
        temp_image_labes = []

        self.img_sz = img_sz
        self.auto_size = 32
        for root, subdirs, files in os.walk(dataset_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                

                # find data type
                data_type_idx = -1
                for idx,val in enumerate(data_set_types):
                    if root.find(val) != -1:
                        data_type_idx = idx
                
                #seqence_type = filename[-7]
                if is_image_file(file_path) == True:
                    self.image_path_list.append(file_path)

        self.image_path_list.sort()

        for each_path in self.image_path_list:
            label = each_path
            label = label[len(dataset_dir):-1]

            # for linux
            if label.find("/",1) != -1:
                label = label[0:label.find("/",1)]
                
            # for windows
            if label.find("\\",1) != -1:
                label = label[0:label.find("\\",1)]
                
            label = label.replace("\\","")
            label = label.replace("/","")
            label_idx = label_list.index(label)
            temp_image_labes.append(label_idx)

        # label list to tensor
        self.image_labels = torch.empty(len(temp_image_labes),dtype=torch.long)
        for i in range(len(temp_image_labes)):
            self.image_labels[i] = temp_image_labes[i]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        stream = open(self.image_path_list[index], "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)

        img0 =  cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)  # BGR
        assert img0 is not None, 'Image Not Found ' + self.image_path_list[index]

        # Padded resize
        #x1 =int(img0.shape[1] /2 - (img0.shape[0])/2)
        #x2 =int(img0.shape[1] /2 + (img0.shape[0])/2)
        #img0 = img0[0:img0.shape[0], x1:x2]
        #target_dim = (self.img_sz,self.img_sz)
        #img = cv2.resize(img0,target_dim)
        img = letterbox(img0, new_shape=(self.img_sz,self.img_sz), auto_size=self.auto_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        path = self.image_path_list[index]
        #print(target.size())
        #target = self.transform(target)

        return img, path

    def __len__(self):
        return len(self.image_path_list)

