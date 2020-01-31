//build:
//g++ -std=c++11 ssdDetector_time.cpp   -lboost_system -lcaffe -lglog -lgflags -ljsoncpp `pkg-config opencv --cflags --libs` -o ssd
//g++ -std=c++11 ssdDetectorM7_pipeline.cpp MNR_Net.cpp   -lboost_system -lcaffe -lglog -lgflags -ljsoncpp -lpthread -lcudart `pkg-config opencv --cflags --libs` -o ssd
//g++ -std=c++11 ssdDetectorM11.cpp MNR_Net.cpp  -lboost_system -lcaffe -lglog -lgflags -ljsoncpp -lpthread -lcudart `pkg-config opencv --cflags --libs` -o ssd
//g++ -std=c++11 ssdDetectorM12_expr.cpp MNR_Net.cpp  -lboost_system -lcaffe -lglog -lgflags -lpthread -lcudart `pkg-config opencv --cflags --libs` -o ssd
//run:
// ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel imageFiles
// ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel ../part02.mp4

//easy version
//./compileRun.sh a
//./compileRun.sh c
//./compileRun.sh t
//./compileRun.sh

#include <opencv2/highgui/highgui.hpp>

#include "MNR_Net.hpp"
#include <iostream>
#include <fstream>
#include <exception>

#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <chrono>

using namespace cv;
using namespace caffe; // NOLINT(build/namespaces)

DEFINE_string(file_type, "image",
              "The file type in the list_file. Currently support image and video.");

cv::Mat drawDetections(cv::Mat img, vector<float> detection)
{
    if (detection[1] == -1)
        return img;
    string classes[] = {"background",
                        "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"};
    cv::Point start, stop;
    start.x = detection[3] * img.cols;
    start.y = detection[4] * img.rows;
    stop.x = detection[5] * img.cols;
    stop.y = detection[6] * img.rows;
    cv::rectangle(img, start, stop, cv::Scalar(255, 0, 0), 2);
    std::stringstream scoreS;
    std::string lable;

    scoreS << detection[2];
    lable = classes[static_cast<int>(detection[1])] + ":" + scoreS.str() + "%";

    cv::putText(img,                         //target image
                lable,                       //text
                cv::Point(start.x, start.y), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(118, 185, 0), //font color
                2);

    return img;
}

int main(int argc, char **argv)
{
    clock_t t1, t2, t3, t4;
    struct sysinfo memInfo;
    ::google::InitGoogleLogging(argv[0]);

    const string &model_file = argv[1];
    const string &weights_file = argv[2];
    const string videoName = argv[3];
    const int frameCount = std::stoi(argv[4]);
    const bool serialDetector = (argv[5][0] == 's' ? true : false);
    const bool useGPU = (argv[6][0] == 'g' ? true : false);
    std::cout << "Frame count: " << frameCount << ", SerialDetector: " << argv[5] << serialDetector << ", UseGPU: " << argv[6] << useGPU << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    string gpuName = prop.name;

    const string &file_type = FLAGS_file_type;

    // Initialize the network.
    Detector detector(model_file, weights_file);
    detector.setRunMode(useGPU);

    int frame_count;
    cv::Mat img = cv::imread("example_01.jpg");
    cv::Mat img2 = cv::imread("example_02.jpg");
    std::vector<vector<float>> detections = detector.serialDetectorMultiStage(img, img2);
    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i)
    {
        drawDetections(img, detections[i]);
    }
    cv::imshow("Display window", img); // Show our image inside it.
    cv::waitKey(0);

    // img = cv::imread("example_02.jpg");

    // detections = detector.serialDetectorMultiStage(img);
    // /* Print the detection results. */
    // for (int i = 0; i < detections.size(); ++i)
    // {
    //     drawDetections(img, detections[i]);
    // }
    // cv::imshow("Display window", img); // Show our image inside it.
    // cv::waitKey(0);
    // cv::waitKey(0);
    return 0;
}
