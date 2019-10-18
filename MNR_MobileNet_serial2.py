import MNR_Net
import time
import cv2
from argparse import ArgumentParser
from os import system

if __name__ == '__main__':
    system('clear')
    parser = ArgumentParser()
    parser.add_argument("-p", "--prototxt", required=False,type=str, default='MobileNetSSD_deploy.prototxt', help="path to Caffe 'deploy' prototxt file", metavar="FILE")
    parser.add_argument("-m", "--model", required=False,type=str, default = 'MobileNetSSD_deploy.caffemodel', help="path to Caffe pre-trained model", metavar="FILE")
    parser.add_argument("-co", "--confidence", required=False,type=float, default=0.7, help="minimum probability to filter weak detections")
    parser.add_argument("-g", "--gpu", required=False,type=bool, default=True, help="Enable GPU boost")
    parser.add_argument("-ca", "--caffe", required=False,type=bool, default=False, help="Enable Caffe transformInput")
    parser.add_argument("-he", "--height", required=False,type=int, default=300, help="Height of transformInput")
    parser.add_argument("-w", "--width", required=False,type=int, default=300, help="Width of transformInput")

    args = vars(parser.parse_args())
    mNet = MNR_Net.MNET(args['prototxt'], args['model'], args['gpu'], args['caffe'], args['height'], args['width'], args['confidence'])
    frame = cv2.imread('example_01.jpg')
    preFrame = mNet.transformInput(frame)
    netOut = mNet.MNetForward(preFrame)
    postFrame = mNet.getBoxedImage(frame, netOut)
    cv2.imshow("SSD", postFrame)
    cv2.waitKey(0)   

    # do a bit of cleanup
    cv2.destroyAllWindows()
