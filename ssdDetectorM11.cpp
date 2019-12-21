//build:
//g++ -std=c++11 ssdDetector_time.cpp   -lboost_system -lcaffe -lglog -lgflags -ljsoncpp `pkg-config opencv --cflags --libs` -o ssd
//g++ -std=c++11 ssdDetectorM7_pipeline.cpp MNR_Net.cpp   -lboost_system -lcaffe -lglog -lgflags -ljsoncpp -lpthread -lcudart `pkg-config opencv --cflags --libs` -o ssd
//g++ -std=c++11 ssdDetectorM11.cpp MNR_Net.cpp  -lboost_system -lcaffe -lglog -lgflags -ljsoncpp -lpthread -lcudart `pkg-config opencv --cflags --libs` -o ssd
//run:
// ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel imageFiles
// ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel ../part02.mp4
#include <opencv2/highgui/highgui.hpp>

#include "MNR_Net.hpp"
#include <iostream>
#include <fstream>
#include <exception>

#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>

using namespace cv;
using namespace caffe; // NOLINT(build/namespaces)

DEFINE_string(file_type, "image",
              "The file type in the list_file. Currently support image and video.");

cv::Mat drawDetections(cv::Mat img, SSD_Out detection)
{
    string classes[] = {"background",
                        "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"};
    cv::Point start, stop;
    vector<float> rect = detection.getRect();
    start.x = rect[0] * img.cols;
    start.y = rect[1] * img.rows;
    stop.x = rect[2] * img.cols;
    stop.y = rect[3] * img.rows;
    cv::rectangle(img, start, stop, cv::Scalar(255, 0, 0), 2);
    std::stringstream scoreS;
    scoreS << detection.getScore();
    std::string lable = classes[detection.getLable()] + ":" + scoreS.str() + "%";
    cv::putText(img,                         //target image
                lable,                       //text
                cv::Point(start.x, start.y), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(118, 185, 0), //font color
                2);
    return img;
}
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
    // Print output to stderr (while still logging)
    //FLAGS_alsologtostderr = 1;

    const string &model_file = argv[1];
    const string &weights_file = argv[2];
    string videoName = argv[3];
    int mode = 0; // 0 for webcam, 1 for read from files
    // if (argc == 4)
    // {
    //     videoName = argv[3];
    // }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    string gpuName = prop.name;

    const string &file_type = FLAGS_file_type;

    // Initialize the network.
    Detector detector(model_file, weights_file);
    detector.setRunMode(true);
    std::thread popThread(&Detector::getImageFromQThread, &detector); // spawn new thread that calls getImageFromQThread()
    cv::VideoCapture cap(videoName);

    // Default resolution of the frame is obtained.The default resolution is system dependent.
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    int frame_count;
    cv::Mat img;
    if (!cap.isOpened())
    {
        LOG(FATAL) << "Failed to open video: " << videoName;
    }

    frame_count = 0;
    t1 = clock();
    while (true)
    {
        if (frame_count % 10 == 0)
        {
            sysinfo(&memInfo);
             std::cout<<"N: "<<frame_count<<", Total: "<<memInfo.totalram<<", free: "<<memInfo.freeram<<std::endl;
            if (memInfo.totalram - memInfo.freeram < 200000000){
std::cout<<"Used: "<<memInfo.totalram - memInfo.freeram<<std::endl;
                usleep(1000);
}

        }
        else
        {
            // Capture frame-by-frame
            cap >> img;
            // If the frame is empty, break immediately
            if (img.empty())// || frame_count > 20)
                break;

            detector.addImageToQ(img);
            frame_count++;
        }
    }
    std::cout << "Added " << frame_count << " frame!" << std::endl;
    cap.release();

    detector.runThread = false;
    popThread.join();
    t2 = clock();
    detector.FPS = (double(t2 - t1) / double(CLOCKS_PER_SEC)) / double(frame_count);
    std::cout << "Run time: " << (double(t2 - t1) / double(CLOCKS_PER_SEC)) << std::endl;
    detector.saveDataToFiles("executionTime_" + gpuName);

    std::cout << "Now runing on CPU" << std::endl;
    detector.clearLogs();
    detector.setRunMode(false);
    detector.runThread = true;
    std::thread popThreadCPU(&Detector::getImageFromQThread, &detector); // spawn new thread that calls getImageFromQThread()
    cap.open(videoName);

    if (!cap.isOpened())
    {
        LOG(FATAL) << "Failed to open video: " << videoName;
    }

    frame_count = 0;
    t3 = clock();

    while (true)
    {
        if (frame_count % 10 == 0)
        {
            sysinfo(&memInfo);
            // std::cout<<"Total: "<<memInfo.totalram<<", free: "<<memInfo.freeram<<std::endl;
            if (memInfo.totalram - memInfo.freeram < 200000000)
                usleep(1000);
        }
        else
        {
            // Capture frame-by-frame
            cap >> img;
            // If the frame is empty, break immediately
            if (img.empty())// || frame_count > 20)
                break;
            detector.addImageToQ(img);
            frame_count++;
        }
    }
    std::cout << "Added " << frame_count << " frame!" << std::endl;
    cap.release();

    detector.runThread = false;
    popThreadCPU.join();
    t4 = clock();
    std::cout << "Run time: " << (double(t4 - t3) / double(CLOCKS_PER_SEC)) << std::endl;
    detector.FPS = (double(t4 - t3) / double(CLOCKS_PER_SEC)) / double(frame_count);

    detector.saveDataToFiles("executionTime_" + gpuName);

    std::cout << "Finished all" << std::endl;

    // When everything done, release the video capture and write object
    cap.release();
    // cv::waitKey(0);
    return 0;
}
