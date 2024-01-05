# -*- encoding: utf-8 -*-
import json
import cv2
import numpy as np
from argparse import ArgumentParser
from PIL import Image

from os import listdir, makedirs, walk
from os.path import isfile, isdir, join, dirname, realpath, exists, splitext, basename


# MPII에서 각 관절 번호
'''
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

KPOP_PARTS = { "가운데 엉덩이": 0, "왼쪽 엉덩이": 1, "왼쪽 무릎": 2, "왼쪽 발목": 3, "왼쪽 엄지 발가락": 4,
                "왼쪽 새끼 발가락": 5, "오른쪽 엉덩이": 6, "오른쪽 무릎": 7, "오른쪽 발목": 8, "오른쪽 엄지 발가락": 9,
                "오른쪽 새끼 발가락": 10, "허리": 11, "가슴": 12, "목": 13, "왼쪽 어깨": 14,
                "왼쪽 팔꿈치": 15, "왼쪽 손목": 16, "왼쪽 엄지 손가락": 17, "왼쪽 중지 손가락": 18, "오른쪽 어깨": 19,
                "오른쪽 팔꿈치": 20, "오른쪽 손목": 21, "오른쪽 엄지 손가락": 22, "오른쪽 중지 손가락": 23, "코": 24,
                "왼쪽 눈": 25, "오른쪽 눈": 26, "왼쪽 귀": 27, "오른쪽 귀": 28 }
'''
TAEKWONDO_PARTS = { "코": 0, "목": 1, "오른쪽 어깨": 2, "오른쪽 팔꿈치": 3, "오른쪽 손목": 4,
                "왼쪽 어깨": 5, "왼쪽 팔꿈치": 6, "왼쪽 손목": 7, "가운데 엉덩이": 8, "오른쪽 엉덩이": 9,
                "오른쪽 무릎": 10, "오른쪽 발목": 11, "왼쪽 엉덩이": 12, "왼쪽 무릎": 13, "왼쪽 발목": 14,
                "오른쪽 눈": 15, "왼쪽 눈": 16, "오른쪽 귀": 17, "왼쪽 귀": 18, "왼쪽 엄지 발가락": 19,
                "왼쪽 새끼 발가락": 20, "왼쪽 뒷꿈치": 21, "오른쪽 엄지 발가락": 22, "오른쪽 새끼 발가락": 23, "오른쪽 뒷꿈치": 24,
                "오른쪽 엄지 손가락": 25, "오른쪽 중지 손가락": 26, "왼쪽 엄지 손가락": 27, "왼쪽 중지 손가락": 28 }

# 관절들을 선으로 이을 때 쌍이 되는 것들
'''
POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
'''
POSE_PAIRS = [ [0, 1], [0, 15], [0, 16], [15, 17], [16, 18], 
                [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], 
                [4, 25], [4, 26], [5, 6], [6, 7], [7, 27], 
                [7, 28], [8, 9], [8, 12], [9, 10], [10, 11], 
                [11, 22], [11, 23], [11, 24], [12, 13], [13, 14], 
                [14, 19], [14, 20], [14, 21] ] 


def main():
    # Argument 처리
    parser = ArgumentParser(description = 'Taekwondo Skeleton.')
    parser.add_argument('SrcPath', type=str, nargs=1, help="Must Input value")
    parser.add_argument('DstPath', type=str, nargs='?')
    
    args = parser.parse_args()

    # 입력 경로 예외처리(디렉토리 존재 여부 확인)
    if not isdir(args.SrcPath[0]):
        print("ERROR> Invalid path")
        return

    #PATH = dirname(realpath(__file__)) 실제 경로를 받는다.
    pathSrc = realpath(args.SrcPath[0])

    # 아규먼트를 통해 받은 이미지 폴더 내 파일들 리스트 생성
    imgFiles = []
    jsonFiles = []

    # 이미지, jpg, 하위 디렉토리 내 파일들 리스트 생성
    for (root, dirs, files) in walk(pathSrc):              
        if len(files) > 0:
            for file_name in files:
                extention = splitext(file_name)[1].lower()

                if extention in '.jpg':
                    imgFiles.append(join(root, file_name))
                elif extention in '.json':
                    jsonFiles.append(join(root, file_name))

    if len(imgFiles) == 0:
        print("ERROR> Empty path")
        return


    # 목적경로
    if not args.DstPath:
        pathDst = join(pathSrc, "Result")
    else:
        pathDst = realpath(args.DstPath)

    if not exists(pathDst.encode("UTF-8")):
        makedirs(pathDst, exist_ok=True)


    for fileImg in imgFiles:
        filename = splitext(basename(fileImg))

        for fileJsn in jsonFiles:
            if filename[0] != splitext(basename(fileJsn))[0]:
                continue

            # 하위폴더 유지하도록..
            strDir = dirname(fileImg)
            if args.DstPath and pathSrc != strDir:
                pathOut = join(pathDst, strDir[len(pathSrc)+1:])
                if not exists(pathOut.encode("UTF-8")):
                    makedirs(pathOut, exist_ok=True)
            else:
                pathOut = pathDst

            analyzeData(fileImg, fileJsn, pathOut)


def analyzeData(pathImg, pathJsn, pathDst):
    with open(pathJsn, "r", encoding="utf-8-sig") as json_file:
        dictionary = json.load(json_file)

    if 'labelingInfo' not in dictionary:
        return False
    
    index = 0
    for label in dictionary['labelingInfo']:
        if 'pose' not in label:
            continue

        if 'location' not in label['pose']:
            continue

        location = label['pose']['location']
        drawSkeleton(location, pathImg, pathDst, index)
        index += 1

    return True


def drawSkeleton(location, pathImg, pathDst, index):
    # 이미지 읽어오기
    #image = cv2.imdecode(np.fromfile(pathImg, dtype=np.uint8), cv2.IMREAD_COLOR)
    imageInfo = Image.open(pathImg)

    # 불러온 이미지에서 사이즈 저장
    imgSize = imageInfo.size

    # 검은색 배경 이미지
    image = np.full((imgSize[1], imgSize[0], 3), (0, 0, 0), dtype=np.uint8)

    points = {}
    # 점 그리기
    for part in TAEKWONDO_PARTS:
        if part not in location:
            continue

        if "x" not in location[part] or "y" not in location[part]:
            continue

        point = (int(location[part]["x"]), int(location[part]["y"]))
        image = cv2.line(image, point, point, (255, 0, 0), 5)
        points[part] = point

    # 선 그리기 (관절들 연결)
    for pair in POSE_PAIRS:
        partA = partB = None
        for k, v in TAEKWONDO_PARTS.items():
            if v == pair[0]:
                partA = points[k]
            elif v == pair[1]:
                partB = points[k]
    
        if partA and partB:
            image = cv2.line(image, partA, partB, (0, 255, 0), 2)

    # 이미지 출력
    retval, img_arr = cv2.imencode('.jpg', image)
    if retval:
        if index > 0:
            outputFile = join(pathDst, '{0}_{1}.jpg'.format(splitext(basename(pathImg))[0], index))
        else:
            outputFile = join(pathDst, '{0}.jpg'.format(splitext(basename(pathImg))[0]))

        with open(outputFile, mode='w+b') as f:
            img_arr.tofile(f)


if __name__ == "__main__":
    main()
