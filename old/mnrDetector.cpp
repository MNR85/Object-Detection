#include "MNR_Net.hpp"
#include <opencv2/highgui/highgui.hpp>
//g++ -std=c++11 mnrDetector.cpp MNR_Net.cpp  -lboost_system -lcaffe -lglog -lgflags -ljsoncpp `pkg-config opencv --cflags --libs` -o mssd
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

int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
    gflags::SetUsageMessage("Do detection using SSD mode.\n"
                            "Usage:\n"
                            "    ssd_detect [FLAGS] model_file weights_file list_file\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4)
    {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
        return 1;
    }

    const string &model_file = argv[1];
    const string &weights_file = argv[2];
    const string &file_type = FLAGS_file_type;

    // Initialize the network.
    Detector detector(model_file, weights_file);
    detector.initLayers();
    //Process image one by one.
    std::ifstream infile(argv[3]);
    std::string file;
    while (infile >> file)
    {
        if (file_type == "image")
        {
            cv::Mat img = cv::imread(file);
            CHECK(!img.empty()) << "Unable to decode image " << file;
            std::vector<SSD_Out> detections = detector.serialDetector(img);
            for (int i = 0; i < detections.size(); ++i) {        
                if(detections[i].isValidDetection()){
                    drawDetections(img, detections[i]);
                    std::cout<<"Detected:\t\t"<<detections[i].ToString();
                }else
                {
                    std::cout<<"False Detection:\t\t"<<detections[i].ToString();
                }

            }
        }
    }
}
