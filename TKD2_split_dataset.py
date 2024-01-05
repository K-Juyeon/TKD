import os
import random
import shutil
import argparse
from configuration import TRAIN_SET_RATIO, TEST_SET_RATIO

# Split Dataset 3
# Start~End frame 들을 한 세트로 움직임

# input : \original_dataset\[class]\[subclass]

# output :  \dataset\[class]\[train]\[subclass]
#           \dataset\[class]\[valid]\[subclass]
#           \dataset\[class]\[test]\[subclass]
global strsource
global stroutput
class SplitDataset():
    def __init__(self, dataset_dir, saved_dataset_dir, train_ratio=TRAIN_SET_RATIO, test_ratio=TEST_SET_RATIO,
                 show_progress=False):
        self.dataset_dir = dataset_dir  # original data set
        self.saved_dataset_dir = saved_dataset_dir  # data set(result folder)
        self.saved_train_dir = saved_dataset_dir + "/training set/"
        self.saved_valid_dir = saved_dataset_dir + "/validation set/"
        self.saved_test_dir = saved_dataset_dir + "/test set/"

        self.train_ratio = train_ratio
        self.test_radio = test_ratio
        self.valid_ratio = 1 - train_ratio - test_ratio

        self.train_file_path = []
        self.valid_file_path = []
        self.test_file_path = []
        # [self.index_label_dict[index][class], self.index_label_dict[index][subclass], list((경로,[S01.jpg,M01.jpg]),(경로2,[S01,M01,E01] ),...)]

        self.index_label_dict = {}  # class 이름을 value로 갖는 dict {0:class, 1:B001, 2:B002, class2, ...}
        # >>이중 dict {0:{"class":, "subclass":}, 1: {"class":, "subclass":}.... }

        self.show_progress = show_progress

        self.json_missing = []

        # if not os.path.exists(self.saved_train_dir):
        #     os.mkdir(self.saved_train_dir)
        # if not os.path.exists(self.saved_test_dir):
        #     os.mkdir(self.saved_test_dir)
        # if not os.path.exists(self.saved_valid_dir):
        #     os.mkdir(self.saved_valid_dir)

    def __get_label_names(self):
        label_names = []
        for class_dir in os.listdir(self.dataset_dir):  # original data set의 모든 파일에 대해
            if os.path.isdir(os.path.join(self.dataset_dir, class_dir)):
                for item in os.listdir(os.path.join(self.dataset_dir, class_dir)):
                    # item : string,폴더명 >> item : dict {"class":, "subclass":} / class 폴더는 예외처리
                    item_path = os.path.join(self.dataset_dir, class_dir,
                                             item)  # os.path.join :  두인자를 하나로 합쳐 1개 경로로 만듬, 경로 + 파일명 >> join(root, class, item)
                    class_subclass = item_path[item_path.find(self.dataset_dir) + len(self.dataset_dir) + 1:]
                    class_name = os.path.split(os.path.split(item_path)[0])[1]
                    #class_name = class_subclass[:class_subclass.find("\\")]
                    if os.path.isdir(item_path):  # 디렉토리면
                        label_names.append({"class": class_name, "subclass": item})  # label name 리스트에 추가 <label name
        return label_names  # class 명을 저장하는 1차원배열, [class1, B001, B002, class2, ...  ] >> list of dictionary [{"class":, "item": },{"class":, "item":}, ....]


    def __get_all_file_path(self):
        all_file_path = []
        index = 0
        for file_type in self.__get_label_names():  # 모든 class (폴더명) 에 대하여 >> file_type : {"class":, "item":} 형태
            self.index_label_dict[index] = file_type  # append dict on dict
            index += 1
            # >> filetype : dict {"class":, "subclass":}
            type_file_path = os.path.join(self.dataset_dir, file_type["class"], file_type["subclass"])
            # root + 폴더명 (B003 의 경우 class명 빠진 상태) >> root + class + subclass
            file_path = dict()
            # type_file_path : root/class/subclass/
            for file in os.listdir(type_file_path):  # 해당 class/subclass 를 열고 내부의 모든 파일에 대해
                single_file_path = os.path.join(type_file_path, file[:file.rfind("-")+1])
                #ex : /경로/P-001-001-B003-M-A2009-N-20210819-01-01-

                frame_name = file[file.rfind("-")+1:]
                if not single_file_path in file_path:
                    #해당프레임의 첫 이미지
                    file_path[single_file_path] = list([frame_name])
                else :
                    file_path[single_file_path].append(frame_name)


                    #file path : dict {"경로명+액션명": list(S01.jpg, S01.json....), }
            all_file_path.append(file_path)
        return all_file_path  # list of list , 행 : class/subclass 열 : filename

    def __copy_files(self, type_path, type_saved_dir):
        # type path = dict
        for item in type_path:  # item 1차원리스트 : [self.index_label_dict[index][class], self.index_label_dict[index][subclass], list((경로,[S01.jpg,M01.jpg]),(경로2,[S01,M01,E01] ),...)]
            src_path_list = item[1]  # list
            dst_path = "%s/" % self.saved_dataset_dir + "%s/" % (item[0]["class"]) + type_saved_dir[
                                                                             len(self.saved_dataset_dir)+1:] + "%s/" % (item[0]["subclass"])  # item[0] + item[1]
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            for src_path in src_path_list:
                #src_path : ("original/..../P-006...-04-02-", [E01.jpg, E01.json, .....])
                for file in src_path[1]:
                    full_src_path = src_path[0]+file
                    if os.path.isfile(full_src_path):
                        shutil.copy(full_src_path, dst_path)
                    if self.show_progress:
                        print("Copying file " + full_src_path + " to " + dst_path)





    def __split_dataset(self):
        all_file_paths = self.__get_all_file_path()  # >>all_file_path = [[class1-B001 내부모든파일], [class1-B002 내부모든파일...],....] 2차원배열

        # shuffle jpg files only
        for index in range(len(all_file_paths)):  # 모든 class/subclass 에 대하여
            # filepathlist = 현재보고있는 class 내부에있는 jpg 파일 리스트
            file_path_list = list(all_file_paths[index].items())  # 해당 클래스 내부의 jpg 파일갯수를 세고
            file_path_list_length = len(file_path_list)
            random.shuffle(file_path_list)  # 클래스 /subclass 내부에서섞는다

            train_num = int(file_path_list_length * self.train_ratio)
            test_num = int(file_path_list_length * self.test_radio)

            self.train_file_path.append([self.index_label_dict[index], file_path_list[: train_num]])
            self.test_file_path.append([self.index_label_dict[index], file_path_list[train_num:train_num + test_num]])
            self.valid_file_path.append([self.index_label_dict[index], file_path_list[train_num + test_num:]])
            # append 되는 요소: list [filetype(class), filepath] 중 80%
            # append [self.index_label_dict[index][class], [subclass], filepathlist[]

    def start_splitting(self):
        self.__split_dataset()
        # 파일 받을 경로를 추가
        # saved
        self.__copy_files(type_path=self.train_file_path, type_saved_dir=self.saved_train_dir)
        self.__copy_files(type_path=self.valid_file_path, type_saved_dir=self.saved_valid_dir)
        self.__copy_files(type_path=self.test_file_path, type_saved_dir=self.saved_test_dir)
        if len(self.json_missing) != 0 :
            print("JSON MISSING!")
            print(*self.json_missing, sep='\n')


def parser_init():

    global strsource
    global stroutput
    
    parser = argparse.ArgumentParser(description="Find image sets contain a image not right size")
    parser.add_argument("--x",type=int, help="Width of Image(Pixels), default = 224",default=224)
    parser.add_argument("--y",type=int, help="Height of Image(Pixels), default = 224",default=224)
    parser.add_argument("--source",type=str, help="source directory",default="H:\\TKD_Data\\1차 Crop\\4차_10")
    parser.add_argument("--output",type=str, help="output directory",default="H:\\TKD_Data\\2차 Split\\4차_10")
    args = parser.parse_args()
    target_size_x = args.x
    target_size_y = args.y
    strsource = args.source
    stroutput = args.output


if __name__ == '__main__':
    parser_init()
    split_dataset = SplitDataset(dataset_dir=strsource,
                                 saved_dataset_dir=stroutput,
                                 show_progress=True)
    split_dataset.start_splitting()
