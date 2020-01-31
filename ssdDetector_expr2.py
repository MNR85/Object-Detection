import numpy as np
import multiprocessing 
from multiprocessing import Process, Value, Queue
from argparse import ArgumentParser
import cv2
import time
import MNR_Net
from imutils.video import VideoStream, FPS
import os
# os.environ["GLOG_minloglevel"] = "1"
os.environ["GLOG_minloglevel"] = "0"
CLASSES = ('background',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def getBoxedImage2(origimg, net_out):
    # print('net: ', net_out)
    h = origimg.shape[0]
    w = origimg.shape[1]
    # box = net_out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    # cls = net_out['detection_out'][0,0,:,1]
    # conf = net_out['detection_out'][0,0,:,2]

    box = net_out[0,0,:,3:7] * np.array([w, h, w, h])

    cls = net_out[0,0,:,1]
    conf = net_out[0,0,:,2]
    
    box, conf, cls = (box.astype(np.int32), conf, cls)
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(origimg, p1, p2, COLORS[int(cls[i])], 2)#(0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        # print('title: '+title+', rect: ',box[i])
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        
    return origimg

def getBoxedImage(origimg, net_out):
    # print('net: ', net_out)
    h = origimg.shape[0]
    w = origimg.shape[1]
    box = net_out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = net_out['detection_out'][0,0,:,1]
    conf = net_out['detection_out'][0,0,:,2]
    
    box, conf, cls = (box.astype(np.int32), conf, cls)
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(origimg, p1, p2, COLORS[int(cls[i])], 2)#(0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        print('title: '+title+', rect: ',box[i])
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        
    return origimg
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-g", '--gpu', required=False,
                        action='store_true', help="Enable GPU boost")
    parser.add_argument("-s", "--serial", required=False,
                        action='store_true', help="Serial or parallel detection")
    parser.add_argument("-p", "--prototxt", required=False, type=str, default='MobileNetSSD_deploy.prototxt',
                        help="path to Caffe 'deploy' prototxt file", metavar="FILE")
    parser.add_argument("-m", "--model", required=False, type=str,
                        default='MobileNetSSD_deploy.caffemodel', help="path to Caffe pre-trained model", metavar="FILE")
    parser.add_argument("-v", "--video", required=False, type=str,
                        default='../part02.mp4', help="path to video input file", metavar="FILE")
    parser.add_argument("-f", "--frame", required=False,
                        type=int, default=20, help="Frame count")
    parser.add_argument("-n", "--name", required=False, type=str,
                        default='GL552vw', help="Name for log file", metavar="FILE")

    args = vars(parser.parse_args())

    detector = MNR_Net.Detector(args['prototxt'], args['model'])
    detector.setRunMode(args['gpu'])

    cap = cv2.VideoCapture(args['video'])
    frame = cv2.imread('example_01.jpg')
    tmpF = cv2.imread('example_01.jpg')
    detector.initNet()
    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    ns.net=detector.net
    detector.runThread.value=True
    # p = Process(name='GpopThread',target=detector.getImageForGPU)
    p = Process(target=detector.getImageForGPU, args=(ns,))
    p.daemon = True
    # p1 = Process(name='CpopThread',target=detector.getImageForCPU)
    p1 = Process(target=detector.getImageForCPU, args=(ns,))
    p1.daemon = True
    p.start()
    time.sleep(3)
    p1.start()
    frameCount=0
    for i in range(0,100):
        detector.addImageToQ2(frame)
        frameCount=frameCount+1
    detector.runThread.value=False
    p.join()
    p1.join()
    print('Finished process!!')

    for i in range (frameCount):
        getBoxedImage(frame, detector.detectionOutputs.get())

    moreInfo = 'mode: serial '+str(args['serial'])+', gpu '+str(args['gpu'])
    if args['serial']==True:
        method = 'Serial'
    else:
        method = 'Pipeline'
    if args['gpu']==True:
        hw = 'GPU'
    else:
        hw = 'CPU'
    gpuName=args['name']
    detector.saveDataToFilesMultiStage("executionTime_python_" + gpuName+"_"+method+"_"+hw, moreInfo, frameCount)

    
    # print('gpu devide:')
    # for i in range(0,10):
    #     netOut = detector.forwardMultiStageDivide(frame, i)
    # print('gpu cpu:')
    # for i in range(0,10):
    #     detector.forwardMultiStageSWgc(frame, i)
    # print('cpu gpu:')
    # for i in range(0,10):
    #     detector.forwardMultiStageSWcg(frame, i)
        #postFrame = getBoxedImage(tmpF, netOut)
        #cv2.imshow("SSD", postFrame)
        #key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        #if key == ord("q"):
            #break

    #key = cv2.waitKey(0)
