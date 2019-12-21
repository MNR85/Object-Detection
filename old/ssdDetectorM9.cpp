//build:
//g++ -std=c++11 ssdDetector_time.cpp   -lboost_system -lcaffe -lglog -lgflags -ljsoncpp `pkg-config opencv --cflags --libs` -o ssd
//g++ -std=c++11 ssdDetectorM7_pipeline.cpp MNR_Net.cpp   -lboost_system -lcaffe -lglog -lgflags -ljsoncpp -lpthread `pkg-config opencv --cflags --libs` -o ssd
//run:
// ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel imageFiles
#include <opencv2/highgui/highgui.hpp>

#include "MNR_Net.hpp"
#include <iostream>
#include <fstream>
#include <exception>
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
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    //FLAGS_alsologtostderr = 1;

    const string &model_file = argv[1];
    const string &weights_file = argv[2];
    string videoName;
    int mode = 0; // 0 for webcam, 1 for read from files
    if (argc == 4)
    {
        videoName = argv[3];
    }
    const string &file_type = FLAGS_file_type;

    // Initialize the network.
    Detector detector(model_file, weights_file);
    detector.setRunMode(true);
    std::thread popThread(&Detector::getImageFromQThread, &detector); // spawn new thread that calls getImageFromQThread()
    cv::VideoCapture cap(videoName);
    if (!cap.isOpened())
    {
        LOG(FATAL) << "Failed to open video: " << videoName;
    }
    // Default resolution of the frame is obtained.The default resolution is system dependent.
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    cv::Mat img;
    int frame_count = 0;
    while (true)
    {
        // Capture frame-by-frame
        cap >> img;
        // If the frame is empty, break immediately
        if (img.empty())
            break;
        detector.addImageToQ(img);
        frame_count++;
        // cv::imshow("Display window", img); // Show our image inside it.
        // cv::waitKey(1);
    }
    std::cout<<"Added "<<frame_count<<" frame!"<<std::endl;
    cap.release();

    detector.runThread = false;
    // detector.getImageFromQThread();
    popThread.join();
    detector.saveDataToFiles("executionTime_" + videoName);
    
    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));
    cap.open(videoName);
    std::cout << "Now writing to video" << std::endl;
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
    std::cout << "detector.detectionOutputs.size()=" << detector.detectionOutputs.size() << std::endl;

    for (int i = 0; i < detector.detectionOutputs.size(); i++)
    {
        cap >> img;

        std::vector<vector<float>> detections = detector.detectionOutputs.front();
        detector.detectionOutputs.pop();
        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i)
        {
            drawDetections(img, detections[i]);
        }

        video.write(img);

        cv::imshow("Display window", img); // Show our image inside it.
        // cv::waitKey(1);
    }
    std::cout << "Finished all" << std::endl;

    // When everything done, release the video capture and write object
    cap.release();
    video.release();
    cv::waitKey(0);
    return 0;
}