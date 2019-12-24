from os import system  # For clearing console
import numpy as np  # For postProcessing
import cv2  # For transform image
import os
import sys
sys.path.append("/home/mnr/caffe-ssd2/python/")
import caffe

import time
from multiprocessing import Process, Value, Queue


class Detector:
    def __init__(self, protxt, caffeModel):
        print('[Starting] Reading protxt')
        if not os.path.exists(protxt):
            print('Could not find protxt file: '+protxt)
            exit()
        if not os.path.exists(caffeModel):
            print('Could not find caffeModel file: '+caffeModel)
            exit()
        self.protxt = protxt
        self.caffeModel = caffeModel
        self.useGPU = False
        self.runThread = Value('b', True)
        self.netIsInit = Value('b', False)
        # self.initNet()
        self.normilizedImages = Queue(maxsize=0)
        self.detectionOutputs = Queue(maxsize=0)
        self.trasformTimes = Queue(maxsize=0)
        self.netTimes = Queue(maxsize=0)
        self.input_geometry_ = []
        self.input_geometry_ = [300, 300] #self.net.params[0][0].data.shape
        system('clear')
        # print('[INFO] Reading from: '+protxt+' and '+caffeModel)
        # print('[INFO] Using GPU mode: '+str(useGPU))
        # print('[INFO] Using caffe transform: '+str(useCaffe))
        # print('[INFO] Transform size: H'+str(self.nh)+' W'+str(self.nw))
    def setRunMode(self, useGPU1):
        self.useGPU = useGPU1
    def configGPUusage(self):
        if (self.useGPU == True):
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

    def initNet(self):
        self.configGPUusage()
        self.net = caffe.Net(self.protxt, self.caffeModel, caffe.TEST)
        self.netIsInit.value=True

    def transformInput(self, image):
        image = cv2.resize(image, (self.input_geometry_[
                           0], self.input_geometry_[1]))
        image = image - 127.5
        image = image * 0.007843
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        preImage = image
        return preImage

    def forwardNet(self, preImage):
        t1 = time.time()
        self.net.blobs['data'].data[...] = preImage
        t2= time.time()
        res=  self.net.forward()
        t3 = time.time()
        self.netTimes.put(t2-t1)
        self.netTimes.put(t3-t2)
        return res

    def serialDetector(self, img):
        t1 = time.time()
        trans = self.transformInput(img)
        t2 = time.time()
        self.trasformTimes.put(t2-t1)
        return self.forwardNet(trans)

    def addImageToQ(self, img):
        t1 = time.time()
        self.normilizedImages.put(self.transformInput(img))
        t2 = time.time()
        self.trasformTimes.put(t2-t1)

    def getImageFromQ(self):
        sample_resized = self.normilizedImages.get()
        return self.forwardNet(sample_resized)

    def getImageFromQThread(self):
        #self.configGPUusage()
        self.initNet()
        while (self.runThread.value or not self.normilizedImages.empty()):
            # print('queue size: ', self.normilizedImages.qsize(), self.runThread.value ,not self.normilizedImages.empty())
            if(self.normilizedImages.empty()):
                time.sleep(0.2)
                continue
            self.detectionOutputs.put(self.getImageFromQ())
        print('Finished thread')

    def pipelineDetectorButWorkSerial(self, img):
        self.addImageToQ(img)
        res = self.getImageFromQ()
        return res

    def clearLogs(self):
        a = 0

    def saveDataToFiles(self, fileName, moreinfo, frameCount):
        f = open(fileName + ".csv", "a")
        f.write(moreinfo+"\n")
        f.write("GPU use = " + str(self.useGPU) + "\n")
        f.write("transTime, feedNetTime, netTime\n")
        for i in range(frameCount):
            f.write(str(self.trasformTimes.get()*1000000)+", "+str(self.netTimes.get()*1000000)+", "+str(self.netTimes.get()*1000000)+"\n")
        f.write("----------------\n\n")
        f.close()
