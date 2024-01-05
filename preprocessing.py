import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import logging
import logging.handlers

from numpy import random

from utils.general import (
    non_max_suppression, scale_coords, strip_optimizer)

from utils.torch_utils import select_device, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from taekwon_dataset import DatasetFromFolderPre 

logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s][%(levelname)sl%(filename)s:%(lineno)s >> %(message)s')

streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('./preprocessing.log')

streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.addHandler(fileHandler)
logger.setLevel(level=logging.DEBUG)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def preprocessing(save_img=True):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # source = '/work/datatone_tkd/데이터톤 문제_sourcedata'
    # out = '/work/datatone_tkd/데이터톤 문제_sourcedata_pre3'

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    if device.type != 'cpu':
        model = Darknet(cfg, imgsz).cuda()
    else:
        model = Darknet(cfg, imgsz)

    try:
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, weights)
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    
    dataset = DatasetFromFolderPre(source,img_sz=imgsz)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for batch_i, sample  in enumerate(dataloader):
        img, path = sample[0], sample[1][0]

        # read original image
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        im0s =  cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)  # BGR

        # crop
        #x1_ =int(im0s.shape[1] /2 - (im0s.shape[0])/2)
        #x2_ =int(im0s.shape[1] /2 + (im0s.shape[0])/2)
        #im0s = im0s[0:im0s.shape[0], x1_:x2_]
        

        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            img_width = im0.shape[1]
            img_height = im0.shape[0]
            
            save_path = p.replace(source,out)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                nDetectPerson = 0;
                # Write results
                max_person_box_size = -1
                for *xyxy, conf, cls in det:
                    if names[int(cls)] == 'person':
                        nDetectPerson = 1;
                        current_box_size = (int(xyxy[2]) - int(xyxy[0]) ) * (int(xyxy[3]) - int(xyxy[1]) )
                        if current_box_size > max_person_box_size:
                            max_person_box_size = current_box_size
                            x1 = int(xyxy[0])
                            y1 = int(xyxy[1])
                            x2 = int(xyxy[2])
                            y2 = int(xyxy[3])
                            crop_width =  x2 - x1
                            crop_height = y2 - y1
                            if crop_width > crop_height:
                                center_y = y1 + (crop_height) /2
                                y1 = int(center_y - crop_width/2)
                                y2 = int(center_y + crop_width/2)                             
                            else:
                                center_x = x1 + (crop_width) /2
                                x1 = int(center_x - crop_height/2)
                                x2 = int(center_x + crop_height/2)

                            print(save_path)
                            logger.info(save_path)
                            print('Found Rect [%d,%d,%d,%d]' % (x1,y1,x2,y2))
                            logger.info('Found Rect [%d,%d,%d,%d]' % (x1,y1,x2,y2))
                            if y1 < 0:
                                y2 = y2 - y1
                                y1 = 0
                            
                            if y2 >= img_height:
                                y1 = y1 - (y2 - (img_height-1) )
                                y2 = img_height-1
                                    
                            if x1 < 0:
                                x2 = x2 - x1
                                x1 = 0
                            
                            if x2 >= img_width:
                                x1 = x1 - (x2 - (img_width-1))
                                x2 = img_width-1

                            
                            print('Trim Rect [%d,%d,%d,%d]' % (x1,y1,x2,y2))
                            logger.info('Trim Rect [%d,%d,%d,%d]' % (x1,y1,x2,y2))
                            im0 = im0s[y1:y2, x1:x2]
                            target_dim = (224, 224)
                            im0 = cv2.resize(im0,target_dim)
                        
                    #if save_img or view_img:  # Add bbox to image
                    #    label = '%s %.2f' % (names[int(cls)], conf)
                    #    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            if nDetectPerson == 0:
                print('Not Found Person : %s' % source)
                logger.error('Not Found Person : %s' % source)
                break;
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            logger.info('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (image with detections)
            res, im_png = cv2.imencode('.png',im0)

            save_dir = os.path.dirname(save_path)
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
                
            with open(save_path, 'wb') as f:
                f.write(im_png.tobytes())
                print('Results saved to %s' % Path(out))
                logger.info('Results saved to %s' % Path(out))


    print('Done. (%.3fs)' % (time.time() - t0))
    logger.info('Results saved to %s' % Path(out))

def crop_image_only_outside(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    print("img shape is ")
    print(img.shape)

    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def preprocessing_crop(save_img=True):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # source = '/work/datatone_tkd/데이터톤 문제_sourcedata'
    # out = '/work/datatone_tkd/데이터톤 문제_sourcedata_pre3'

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    if device.type != 'cpu':
        model = Darknet(cfg, imgsz).cuda()
    else:
        model = Darknet(cfg, imgsz)

    try:
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, weights)
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    
    dataset = DatasetFromFolderPre(source,img_sz=imgsz)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for batch_i, sample  in enumerate(dataloader):
        img, path = sample[0], sample[1][0]

        # read original image
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        #im0s =  cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)  # BGR
        im0s =  cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)  # BGR

        # crop
        #x1_ =int(im0s.shape[1] /2 - (im0s.shape[0])/2)
        #x2_ =int(im0s.shape[1] /2 + (im0s.shape[0])/2)
        #im0s = im0s[0:im0s.shape[0], x1_:x2_]
        im0s = crop_image_only_outside(im0s)
        im0s =  cv2.cvtColor(im0s,cv2.COLOR_GRAY2RGB)
        target_dim = (224, 224)
        im0s = cv2.resize(im0s,target_dim)

           

        # Save results (image with detections)
        res, im_png = cv2.imencode('.png',im0s)
        p = path
        save_path = p.replace(source,out)
        save_dir = os.path.dirname(save_path)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
            
        with open(save_path, 'wb') as f:
            f.write(im_png.tobytes())
            print('Results saved to %s' % Path(out))


    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/yolov4-csp.weights', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='E:\\tkd_data\\sourcedata', help='source directory')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='E:\\tkd_data\\sourcedata', help='output directory')  # output folder
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='models/yolov4-csp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    
    opt = parser.parse_args()
    print(opt)
    logger.info(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                preprocessing()
                strip_optimizer(opt.weights)
        else:
            preprocessing()