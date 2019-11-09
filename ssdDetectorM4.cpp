// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//

//build:
//g++ -std=c++11 ssdDetector_time.cpp   -lboost_system -lcaffe -lglog -lgflags -ljsoncpp `pkg-config opencv --cflags --libs` -o ssd
//run:
// ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel imageFiles
#include <opencv2/highgui/highgui.hpp>

#include "MNR_Net.hpp"

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
    scoreS << detection[2];
    std::string lable = classes[static_cast<int>(detection[1])] + ":" + scoreS.str() + "%";
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
    const string &file_type = FLAGS_file_type;

    // Initialize the network.
    Detector detector(model_file, weights_file);

    // Process image one by one.
    std::ifstream infile(argv[3]);
    std::string file;
    while (infile >> file)
    {
        if (file_type == "image")
        {
            cv::Mat img = cv::imread(file);
            CHECK(!img.empty()) << "Unable to decode image " << file;
            std::vector<vector<float>> detections = detector.serialDetector(img);
            /* Print the detection results. */
            for (int i = 0; i < detections.size(); ++i)
            {
                drawDetections(img, detections[i]);
            }
            cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
            cv::imshow("Display window", img);                      // Show our image inside it.
            cv::waitKey(0);
        }
    }
    return 0;
}