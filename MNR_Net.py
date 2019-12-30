import cv2  # For transform image
import numpy as np  # For postProcessing
from os import system  # For clearing console
import os
import sys
sys.path.append("/home/mnr/caffe-ssd2/python/")
import caffe
from multiprocessing import Process, Value, Queue
import time



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
        self.useGPU = True
        self.runThread = Value('b', True)
        self.netIsInit = Value('b', False)
        # self.initNet()
        self.normilizedImages = Queue(maxsize=0)
        self.detectionOutputs = Queue(maxsize=0)
        self.trasformTimes = Queue(maxsize=0)
        self.thread1Times = Queue(maxsize=0)
        self.thread2Times = Queue(maxsize=0)
        self.netTimes = Queue(maxsize=0)
        self.input_geometry_ = []
        self.input_geometry_ = [300, 300]  # self.net.params[0][0].data.shape
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
        self.netIsInit.value = True

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
        t2 = time.time()
        res = self.net.forward()
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
    
    def serialDetector2Stage(self, img):
        trans = self.transformInput(img)
        self.net.blobs['data'].data[...] = trans
        t1 = time.time()
        self.forwardNet(trans)
        t2 = time.time()
        self.net.forward(end='conv2')
        dataMid = self.net.blobs['conv2'].data
        t3 = time.time()
        self.net.blobs['conv2'].data[...] = dataMid
        res = self.net.forward(start='conv2')
        t4 = time.time()
        dur1 = (t2-t1)*1000000
        dur2 = (t4-t2)*1000000

        print('time1: '+str(dur1)+', time2: '+str(dur2)+' dif: '+str(dur2-dur1)+' p1: '+str(t3-t2)+' p2: '+str(t4-t3))
        return res

    def forwardMultiStageSWcg(self, img, itr):
        trans = self.transformInput(img)
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        startT = time.time()
        
        t1 = time.time()
        self.net.blobs['data'].data[...] = trans
        startLayer = ''
        # t1 = time.time()
        self.net.forward(end='conv1')
        t2 = time.time()
        print('1: '+str(t2-t1)),
        
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv1'].data
        self.net.blobs['conv1'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv1', end='conv2')
        t2 = time.time()
        print('2: '+str(t2-t1)),
        
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv2'].data
        self.net.blobs['conv2'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv2', end='conv3')
        t2 = time.time()
        print('3: '+str(t2-t1)),
        
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv3'].data
        self.net.blobs['conv3'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv3', end='conv4')
        t2 = time.time()
        print('4: '+str(t2-t1)),

        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv4'].data
        self.net.blobs['conv4'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv4', end='conv5')
        t2 = time.time()
        print('5: '+str(t2-t1)),

        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv5'].data
        self.net.blobs['conv5'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv5', end='conv6')
        t2 = time.time()
        print('6: '+str(t2-t1)),

        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv6'].data
        self.net.blobs['conv6'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv6', end='conv7')
        t2 = time.time()
        print('7: '+str(t2-t1)),

        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv7'].data
        self.net.blobs['conv7'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv7', end='conv8')
        t2 = time.time()
        print('8: '+str(t2-t1)),

        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv8'].data
        self.net.blobs['conv8'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv8', end='conv9')
        t2 = time.time()
        print('9: '+str(t2-t1)),

        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv9'].data
        self.net.blobs['conv9'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv9', end='conv10')
        t2 = time.time()
        print('10: '+str(t2-t1)),
        
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv10'].data
        self.net.blobs['conv10'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv10', end='conv11')
        t2 = time.time()
        print('11: '+str(t2-t1)),
        
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv11'].data
        self.net.blobs['conv11'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv11', end='conv12')
        t2 = time.time()
        print('12: '+str(t2-t1)),
        
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv12'].data
        #t1 = time.time()
        self.net.blobs['conv12'].data[...] = dataMid
        self.net.forward(start = 'conv12', end='conv13')
        t2 = time.time()
        print('13: '+str(t2-t1)),
        
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv13'].data
        self.net.blobs['conv13'].data[...] = dataMid
        # t1 = time.time()
        res = self.net.forward(start='conv13')
        t2 = time.time()
        print('14: '+str(t2-t1)),
        endT = time.time()
        print('T: '+str(endT-startT)),
        print('')

        return res

    def forwardMultiStageSWgc(self, img, itr):
        trans = self.transformInput(img)
        #caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        startT = time.time()
        
        t1 = time.time()
        self.net.blobs['data'].data[...] = trans
        startLayer = ''
        # t1 = time.time()
        self.net.forward(end='conv1')
        t2 = time.time()
        print('1: '+str(t2-t1)),
        
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv1'].data
        self.net.blobs['conv1'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv1', end='conv2')
        t2 = time.time()
        print('2: '+str(t2-t1)),
        
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv2'].data
        self.net.blobs['conv2'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv2', end='conv3')
        t2 = time.time()
        print('3: '+str(t2-t1)),
        
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv3'].data
        self.net.blobs['conv3'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv3', end='conv4')
        t2 = time.time()
        print('4: '+str(t2-t1)),

        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv4'].data
        self.net.blobs['conv4'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv4', end='conv5')
        t2 = time.time()
        print('5: '+str(t2-t1)),

        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv5'].data
        self.net.blobs['conv5'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv5', end='conv6')
        t2 = time.time()
        print('6: '+str(t2-t1)),

        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv6'].data
        self.net.blobs['conv6'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv6', end='conv7')
        t2 = time.time()
        print('7: '+str(t2-t1)),

        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv7'].data
        self.net.blobs['conv7'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv7', end='conv8')
        t2 = time.time()
        print('8: '+str(t2-t1)),

        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv8'].data
        self.net.blobs['conv8'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv8', end='conv9')
        t2 = time.time()
        print('9: '+str(t2-t1)),

        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        
        t1 = time.time()
        dataMid = self.net.blobs['conv9'].data
        self.net.blobs['conv9'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv9', end='conv10')
        t2 = time.time()
        print('10: '+str(t2-t1)),
        
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv10'].data
        self.net.blobs['conv10'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv10', end='conv11')
        t2 = time.time()
        print('11: '+str(t2-t1)),
        
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv11'].data
        self.net.blobs['conv11'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv11', end='conv12')
        t2 = time.time()
        print('12: '+str(t2-t1)),
        
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv12'].data
        #t1 = time.time()
        self.net.blobs['conv12'].data[...] = dataMid
        self.net.forward(start = 'conv12', end='conv13')
        t2 = time.time()
        print('13: '+str(t2-t1)),
        
        caffe.set_device(0)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        t1 = time.time()
        dataMid = self.net.blobs['conv13'].data
        self.net.blobs['conv13'].data[...] = dataMid
        # t1 = time.time()
        res = self.net.forward(start='conv13')
        t2 = time.time()
        print('14: '+str(t2-t1)),
        endT = time.time()
        print('T: '+str(endT-startT)),
        print('')

        return res

    def forwardMultiStageDivide(self, img, itr):
        caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        trans = self.transformInput(img)
        #caffe.set_device(0)
        
        startT = time.time()
        t1 = time.time()
        self.net.blobs['data'].data[...] = trans
        startLayer = ''
        # t1 = time.time()
        self.net.forward(end='conv1')
        t2 = time.time()
        print('1: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv1'].data
        self.net.blobs['conv1'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv1', end='conv2')
        t2 = time.time()
        print('2: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv2'].data
        self.net.blobs['conv2'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv2', end='conv3')
        t2 = time.time()
        print('3: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv3'].data
        self.net.blobs['conv3'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv3', end='conv4')
        t2 = time.time()
        print('4: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv4'].data
        self.net.blobs['conv4'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv4', end='conv5')
        t2 = time.time()
        print('5: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv5'].data
        self.net.blobs['conv5'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv5', end='conv6')
        t2 = time.time()
        print('6: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv6'].data
        self.net.blobs['conv6'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv6', end='conv7')
        t2 = time.time()
        print('7: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv7'].data
        self.net.blobs['conv7'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv7', end='conv8')
        t2 = time.time()
        print('8: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv8'].data
        self.net.blobs['conv8'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv8', end='conv9')
        t2 = time.time()
        print('9: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv9'].data
        self.net.blobs['conv9'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv9', end='conv10')
        t2 = time.time()
        print('10: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv10'].data
        self.net.blobs['conv10'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv10', end='conv11')
        t2 = time.time()
        print('11: '+str(t2-t1)),
        t1 = time.time()
        caffe.set_device(0)
        dataMid = self.net.blobs['conv11'].data
        self.net.blobs['conv11'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv11', end='conv12')
        t2 = time.time()
        print('12: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv12'].data
        t1 = time.time()
        self.net.blobs['conv12'].data[...] = dataMid
        self.net.forward(start = 'conv12', end='conv13')
        t2 = time.time()
        print('13: '+str(t2-t1)),
        dataMid = self.net.blobs['conv13'].data
        t1 = time.time()
        self.net.blobs['conv13'].data[...] = dataMid
        # t1 = time.time()
        res = self.net.forward(start='conv13')
        t2 = time.time()
        print('14: '+str(t2-t1)),
        endT = time.time()
        print('T: '+str(endT-startT)),
        print('')

        return res
        
    def forwardMultiStage(self, img, itr):
        trans = self.transformInput(img)
        #caffe.set_device(0)
        #caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        startT = time.time()
        t1 = time.time()
        self.net.blobs['data'].data[...] = trans
        startLayer = ''
        # t1 = time.time()
        self.net.forward(end='conv1')
        t2 = time.time()
        print('1: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv1'].data
        self.net.blobs['conv1'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv1', end='conv2')
        t2 = time.time()
        print('2: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv2'].data
        self.net.blobs['conv2'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv2', end='conv3')
        t2 = time.time()
        print('3: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv3'].data
        self.net.blobs['conv3'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv3', end='conv4')
        t2 = time.time()
        print('4: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv4'].data
        self.net.blobs['conv4'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv4', end='conv5')
        t2 = time.time()
        print('5: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv5'].data
        self.net.blobs['conv5'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv5', end='conv6')
        t2 = time.time()
        print('6: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv6'].data
        self.net.blobs['conv6'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv6', end='conv7')
        t2 = time.time()
        print('7: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv7'].data
        self.net.blobs['conv7'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv7', end='conv8')
        t2 = time.time()
        print('8: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv8'].data
        self.net.blobs['conv8'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv8', end='conv9')
        t2 = time.time()
        print('9: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv9'].data
        self.net.blobs['conv9'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv9', end='conv10')
        t2 = time.time()
        print('10: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv10'].data
        self.net.blobs['conv10'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv10', end='conv11')
        t2 = time.time()
        print('11: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv11'].data
        self.net.blobs['conv11'].data[...] = dataMid
        # t1 = time.time()
        self.net.forward(start = 'conv11', end='conv12')
        t2 = time.time()
        print('12: '+str(t2-t1)),
        t1 = time.time()
        dataMid = self.net.blobs['conv12'].data
        t1 = time.time()
        self.net.blobs['conv12'].data[...] = dataMid
        self.net.forward(start = 'conv12', end='conv13')
        t2 = time.time()
        print('13: '+str(t2-t1)),
        dataMid = self.net.blobs['conv13'].data
        t1 = time.time()
        self.net.blobs['conv13'].data[...] = dataMid
        # t1 = time.time()
        res = self.net.forward(start='conv13')
        t2 = time.time()
        print('14: '+str(t2-t1)),
        endT = time.time()
        print('T: '+str(endT-startT)),
        print('')

        return res
        # self.net.forward(end='conv1')
        # dataMid = self.net.blobs['conv1'].data
        # self.net.blobs['conv1'].data[...] = dataMid
        # for i in range(2,12):
        #     startLayer = 'conv'+str(i-1)
        #     splitLayer = 'conv'+str(i)
        #     t1 = time.time()
        #     self.net.forward(start = startLayer, end=splitLayer)
        #     dataMid = self.net.blobs[splitLayer].data
        #     self.net.blobs[splitLayer].data[...] = dataMid
        #     t2 = time.time()
        #     print(splitLayer+': '+str(t2-t1), end=' ')
        # print('')
        # res = self.net.forward(start='conv13')
        # return res#self.net.blobs['detection_out'].data



    def serialDetectorMultiStage(self, img, itr):
        trans = self.transformInput(img)
        if itr==0:
            caffe.set_device(0)
            caffe.set_mode_gpu()
            t1 = time.time()
            self.net.blobs['data'].data[...] = trans
            res =self.net.forward()
            t2 = time.time()
            print('p1: '+str(t2-t1))
        else:

            t0 = time.time()
            caffe.set_mode_cpu()
            self.net.blobs['data'].data[...] = trans
            t1 = time.time()
            str1 = 'conv'+str(itr)
            print('target layer : '+str1)
            self.net.forward(end=str1)
            dataMid = self.net.blobs[str1].data
            t2 = time.time()
            self.net.blobs[str1].data[...] = dataMid
            t11 = time.time()
            caffe.set_device(0)
            caffe.set_mode_gpu()
            t22 = time.time()
            t3 = time.time()
            res = self.net.forward(start=str1)
            t4 = time.time()
            print('p1: '+str(t2-t1)+', p2: '+str(t4-t3)+', sw: '+str(t22-t11)+', total: '+str(t4-t0))
        return res

    def addImageToQ(self, img):
        t1 = time.time()
        self.normilizedImages.put(self.transformInput(img))
        t2 = time.time()
        self.trasformTimes.put(t2-t1)

    def getImageFromQ(self):
        sample_resized = self.normilizedImages.get()
        return self.forwardNet(sample_resized)

    def getImageFromQThread(self):
        # self.configGPUusage()
        counter = 0
        self.initNet()
        while (self.runThread.value or not self.normilizedImages.empty()):
            t1 = time.time()
            #print('queue size: ', self.normilizedImages.qsize(), self.runThread.value, not self.normilizedImages.empty())
            if(self.normilizedImages.empty()):
                time.sleep(0.2)
                continue
            # self.getImageFromQ()
            # self.detectionOutputs.put(counter)
            self.detectionOutputs.put(self.getImageFromQ())
            # if(counter>300):
            # self.getImageFromQ()
            # else:
            # self.detectionOutputs.put(self.getImageFromQ())
            counter = counter+1
            self.thread2Times.put(time.time()-t1)

        #print('self.thread2Times', self.thread2Times.qsize())
        #print('self.detectionOutputs', self.detectionOutputs.qsize())
        # self.detectionOutputs.task_done()
        print('Finished thread')
        return
        print('after return')

    def pipelineDetectorButWorkSerial(self, img):
        self.addImageToQ(img)
        res = self.getImageFromQ()
        return res

    def clearLogs(self):
        a = 0

    def newPreprocess(self, timer):
        self.thread1Times.put(timer)

    def saveDataToFiles(self, fileName, moreinfo, frameCount, isSerial):
        f = open(fileName + ".csv", "a")
        f.write(moreinfo+"\n")
        f.write("GPU use = " + str(self.useGPU) + "\n")
        f.write("transTime, feedNetTime, netTime")
        if isSerial==False:
            f.write(", thread#1, thread#2")
        f.write("\n")

        for i in range(frameCount):
            TransTime = self.trasformTimes.get()*1000000
            FeedTime = self.netTimes.get()*1000000
            NetTime = self.netTimes.get()*1000000
            if isSerial==False:
                thread1 = self.thread1Times.get()*1000000
                thread2 = self.thread2Times.get()*1000000
            f.write(str(TransTime)+", "+str(FeedTime)+", " + str(NetTime))
            if isSerial==False:
                f.write(", "+str(thread1)+", "+str(thread2))
            f.write("\n")
        f.write("----------------\n\n")
        f.close()
