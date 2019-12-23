import os
import sys
sys.path.append("/home/mnr/caffe-ssd2/python/")
import caffe
import cv2          #For transform image
import numpy as np  #For postProcessing
from os import system   #For clearing console

class MNET:
        def __init__(self, protxt, caffeModel, useGPU, useCaffe, transformH, transformW, confidence):
            print('[Starting] Reading protxt')
            if not os.path.exists(protxt):
                print('Could not find protxt file: '+protxt)
                exit()
            if not os.path.exists(caffeModel):
                print('Could not find caffeModel file: '+caffeModel)
                exit()
            useCaffe = False
            self.protxt = protxt
            self.caffeModel = caffeModel
            # useGPU=False
            if(useGPU == True):
                caffe.set_mode_gpu()
                caffe.set_device(0)
            else:
                caffe.set_mode_cpu()
            caffe.set_mode_gpu()
            caffe.set_device(0)
            #os.environ["GLOG_minloglevel"] = "1"
            # with open(os.devnull, "w") as devnull:
            #     old_stdout = sys.stdout
            #     sys.stdout = devnull
            self.net = caffe.Net(protxt,caffeModel,caffe.TEST)
            # with open(os.devnull, "w") as devnull:
            #     sys.stdout = old_stdout
            #os.environ["GLOG_minloglevel"] = "0"
            #system('clear')

            self.useCaffe = useCaffe
            self.nh, self.nw = transformH, transformW   #224, 224
            self.confidence = confidence

            self.CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

            self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
            system('clear')
            print ('[INFO] Reading from: '+protxt+' and '+caffeModel)
            print ('[INFO] Using GPU mode: '+str(useGPU))
            print ('[INFO] Using caffe transform: '+str(useCaffe))
            print ('[INFO] Transform size: H'+str(self.nh)+' W'+str(self.nw))
            if(self.useCaffe==True):
                self.img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
                #self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
                #self.transformer.set_transpose('data', (2, 0, 1))  # row to col
                #self.transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
                #self.transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
                #self.transformer.set_mean('data', self.img_mean)
                #self.transformer.set_input_scale('data', 0.017)

        def initNet(self):
            self.net = caffe.Net(self.protxt,self.caffeModel,caffe.TEST)

        def transformInput(self, image):
            if(self.useCaffe == True):
                h, w, _ = image.shape
                if h < w:
                    off = (w - h) / 2
                    image = image[:, off:off + h]
                else:
                    off = (h - w) / 2
                    image = image[off:off + h, :]
                image = caffe.io.resize_image(image, [self.nh, self.nw])

                transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
                transformer.set_transpose('data', (2, 0, 1))  # row to col
                transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
                transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
                transformer.set_mean('data', self.img_mean)
                transformer.set_input_scale('data', 0.017)

                self.net.blobs['data'].reshape(1, 3, self.nh, self.nw)
                preImage = transformer.preprocess('data', image)
                #preImage = self.transformer.preprocessimage('data', image)
            else:
                image = cv2.resize(image, (self.nh,self.nw))
                fs_write = cv2.FileStorage('sample_resizedfilePython.yml', cv2.FILE_STORAGE_WRITE)
                fs_write.write("sample_resized", image)
                image = image - 127.5
                fs_write.write("sample_subed", image)
                image = image * 0.007843
                fs_write.write("sample_muled", image)
                image = image.astype(np.float32)
                fs_write.write("toFloat32", image)
                image = image.transpose((2, 0, 1))
                fs_write.write("transposed", image)
                fs_write.release()
                preImage = image
            return preImage

        def MNetForward(self, preImage):
            self.net.blobs['data'].data[...] = preImage
            return self.net.forward()

        def printProp(self, net_out):
            prob = net_out['prob']
            prob = np.squeeze(prob)
            idx = np.argsort(-prob)

            label_names = np.loadtxt('synset.txt', str, delimiter='\t')
            for i in range(5):
                label = idx[i]
                if label>self.confidence:
                    print('%.2f - %s' % (prob[label], label_names[label]))
            return

        def getBoxedImage(self, origimg, net_out):
            h = origimg.shape[0]
            w = origimg.shape[1]
            a = net_out['detection_out']
            b = a[0,0]
            b1 = a[0,0,:]
            b2 = a[0,0,:,1]
            box = net_out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

            cls = net_out['detection_out'][0,0,:,1]
            conf = net_out['detection_out'][0,0,:,2]

            box, conf, cls = (box.astype(np.int32), conf, cls)

            for i in range(len(box)):
                if conf[i]>self.confidence:
                    p1 = (box[i][0], box[i][1])
                    p2 = (box[i][2], box[i][3])
                    cv2.rectangle(origimg, p1, p2, self.COLORS[int(cls[i])], 2)#(0,255,0))
                    p3 = (max(p1[0], 15), max(p1[1], 15))
                    title = "%s:%.2f" % (self.CLASSES[int(cls[i])], conf[i])
                    cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
                    print (conf[i], box[i])

            return origimg
