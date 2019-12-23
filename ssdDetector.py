import os
os.environ["GLOG_minloglevel"] = "1"
from imutils.video import VideoStream, FPS
import MNR_Net
import time
import cv2
from argparse import ArgumentParser
from multiprocessing import Process, Value, Queue
import numpy as np
os.environ["GLOG_minloglevel"] = "0"
def transformInput(image):
    image = cv2.resize(image, (300,300))
    image = image - 127.5
    image = image * 0.007843
    image = image.astype(np.float32)
    image = image.transpose((2, 0, 1))
    preImage = image
    return preImage

def getBoxedImage(origimg, net_out):
    h = origimg.shape[0]
    w = origimg.shape[1]
    box = net_out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = net_out['detection_out'][0,0,:,1]
    conf = net_out['detection_out'][0,0,:,2]
    
    box, conf, cls = (box.astype(np.int32), conf, cls)
    for i in range(len(box)):
        if conf[i]>confidence:
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            cv2.rectangle(origimg, p1, p2, COLORS[int(cls[i])], 2)#(0,255,0))
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        
    return origimg

def getFrameP(vs, cam_stop, frameQ, frameQSave):
    print ("[INFO] starting getFrameP...")
    while not cam_stop.value:
        ret, frame = vs.read()
        if(ret == True):
            frameQ.put(frame)
            frameQSave.put(frame)

def preFrameP(cam_stop, frameQ, preFrameQ):
    print ("[INFO] starting preFrameP...")
    while not cam_stop.value or not frameQ.empty():
        if(not frameQ.empty()):
            preFrame = transformInput(frameQ.get())
            preFrameQ.put(preFrame)

def netOutP(cam_stop, args, preFrameQ, netOutQ):
    print ("[INFO] starting netOutP...")
    mNet = MNR_Net.MNET(args['prototxt'], args['model'], args['gpu'], args['caffe'], args['height'], args['width'], args['confidence'])
    while not cam_stop.value or not preFrameQ.empty():
        if(not preFrameQ.empty()):
            netOut = mNet.MNetForward(preFrameQ.get())
            netOutQ.put(netOut)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-g", '--gpu', required=False, action='store_true', help="Enable GPU boost")
    parser.add_argument("-ca", "--caffe", required=False, action='store_false', help="Enable Caffe transformInput")
    parser.add_argument("-p", "--prototxt", required=False,type=str, default='chuanqi305_MobileNetSSD_dwc_deploy.prototxt', help="path to Caffe 'deploy' prototxt file", metavar="FILE")
    parser.add_argument("-m", "--model", required=False,type=str, default = 'chuanqi305_MobileNetSSD_deploy.caffemodel', help="path to Caffe pre-trained model", metavar="FILE")
    parser.add_argument("-co", "--confidence", required=False,type=float, default=0.7, help="minimum probability to filter weak detections")
    parser.add_argument("-he", "--height", required=False,type=int, default=300, help="Height of transformInput")
    parser.add_argument("-w", "--width", required=False,type=int, default=300, help="Width of transformInput")
    parser.add_argument("-d", "--display", required=False, action='store_true', help="Display output")

    args = vars(parser.parse_args())
    print str(args)
    CLASSES = ('background',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    confidence = args['confidence']
    #mNet = MNR_Net.MNET(args['prototxt'], args['model'], args['gpu'], args['caffe'], args['height'], args['width'], args['confidence'])
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    vs = cv2.VideoCapture(0)
    time.sleep(0.1)
    frame = vs.read()
    ret, frame = vs.read()
    print str(frame.shape)
    fps = FPS().start()
    frameCount=0

    qMaxSize = 0
    cam_stop=Value('b',False)
    frameQ = Queue(maxsize=qMaxSize)
    frameQSave = Queue(maxsize=qMaxSize)
    preFrameQ = Queue(maxsize=qMaxSize)
    netOutQ = Queue(maxsize=qMaxSize)
    postFrameQ = Queue(maxsize=qMaxSize)

    p = Process(target=getFrameP, args=(vs, cam_stop, frameQ, frameQSave,))
    p.daemon = True
    p.start()

    p = Process(target=preFrameP, args=(cam_stop, frameQ, preFrameQ,))
    p.daemon = True
    p.start()

    p = Process(target=netOutP, args=(cam_stop, args, preFrameQ, netOutQ,))
    p.daemon = True
    p.start()
    try:
        while not cam_stop.value or not netOutQ.empty():
            if not netOutQ.empty():
                origImage = frameQSave.get()
                netOut = netOutQ.get()
                if(args['display']== True):                    
                    postFrame = getBoxedImage(origImage, netOut)
                    cv2.imshow("SSD", postFrame)
                    key = cv2.waitKey(1) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        cam_stop = Value('b',True)
                            #break
                # update the FPS counter
                fps.update()
                frameCount = frameCount+1
    except KeyboardInterrupt as ex:
        cam_stop = Value('b',True)
        #sys.tracebacklimit = 0
    fps.stop()
    #vs.stop()
    vs.release()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed())+" frame counter = "+str(frameCount))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps())+" FPS real: {:.2f}".format(frameCount/fps.elapsed()))

    # do a bit of cleanup
    cv2.destroyAllWindows()