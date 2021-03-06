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
    std::thread popThread(&Detector::getImageFromQThread, &detector); // spawn new thread that calls getImageFromQThread()
    if (serialDetector)
    {
        detector.runThread = false;
        popThread.join();
    }
    cv::VideoCapture cap(videoName);

    int frame_count;
    cv::Mat img;
    if (!cap.isOpened())
    {
        LOG(FATAL) << "Failed to open video: " << videoName;
    }

    frame_count = 0;
    t1 = clock();
    time_t t111 = time(0);
    auto t11 = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if (frame_count % 10 == 0)
        {
            sysinfo(&memInfo);
            if (memInfo.freeram < 20000000)
            {
                std::cout << "Sleeping.. free RAM: " << memInfo.freeram << std::endl;
                usleep(1000000);
                continue;
            }
        }
        std::chrono::high_resolution_clock::time_point t11 = std::chrono::high_resolution_clock::now();

        if (!cap.read(img) || frame_count >= frameCount)
            break;
        if (serialDetector)
            detector.serialDetector(img);
        else
            detector.addImageToQ(img);
        frame_count++;
        std::chrono::high_resolution_clock::time_point t22 = std::chrono::high_resolution_clock::now();
        detector.newPreprocess(std::chrono::duration_cast<std::chrono::microseconds>(t22 - t11).count());
    }
    std::cout << "Added " << frame_count << " frame!" << std::endl;
    cap.release();

    t3 = clock();
    auto t33 = std::chrono::high_resolution_clock::now();
    time_t t333 = time(0);

    if (!serialDetector)
    {
        detector.runThread = false;
        popThread.join();
    }

    t2 = clock();
    detector.FPS = double(frame_count) / (double(t2 - t1) / double(CLOCKS_PER_SEC));

    auto t22 = std::chrono::high_resolution_clock::now();
    time_t t222 = time(0);
    int duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t33 - t11).count();
    int duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t22 - t11).count();
    std::cout << "FPS CPU: " << double(frame_count) / (double(t3 - t1) / double(CLOCKS_PER_SEC)) << ", Run time: " << (double(t3 - t1) / double(CLOCKS_PER_SEC)) << std::endl;
    std::cout << "Time: " << duration2 / 1000.0 << "ms, FPS: " << 1000000.0 / duration2 << std::endl;
    std::cout << "FPS TOTAL: " << detector.FPS << ", Run time: " << (double(t2 - t1) / double(CLOCKS_PER_SEC)) << std::endl;
    std::cout << "Time: " << duration1 / 1000.0 << "ms, FPS: " << 1000000.0 / duration1 << std::endl;
    std::cout << "TimeR: " << difftime(t222, t111) * 1000.0 << std::endl;
    string method = serialDetector ? "Serial" : "Pipeline";
    string moreInfo = "Detection method: " + method + "\n";
    try
    {
        moreInfo += "Clock = " + std::to_string((double(t3 - t1))) + ", Chrono = " + std::to_string(duration2) + ", TimeR = " + std::to_string(difftime(t333, t111)) + "\n";
        moreInfo += "Clock = " + std::to_string((double(t2 - t1))) + ", Chrono = " + std::to_string(duration1) + ", TimeR = " + std::to_string(difftime(t222, t111)) + "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "In forwardNet(), collecting result - " << e.what() << '\n';
    }
    string hw = useGPU ? "GPU" : "CPU";
    detector.saveDataToFiles("executionTime_" + gpuName + "_" + method + "_" + hw, moreInfo, frame_count, serialDetector);
    // cv::waitKey(0);
    return 0;
}
