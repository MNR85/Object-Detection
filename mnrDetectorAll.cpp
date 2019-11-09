#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream> 

using namespace caffe;
class SSD_Out
{
public:
    SSD_Out(vector<float> detections)
    {
        detection = detections;
    }
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    vector<float> detection;
    bool isValidDetection()
    {
        return ~(detection[0] == -1 || detection.size() != 7);
    }
    float getImage_ID()
    {
        return detection[0];
    }
    int getLable()
    {
        return static_cast<int>(detection[1]);
    }
    float getScore()
    {
        return detection[2];
    }
    vector<float> getRect()
    {
        vector<float> rect(detection[3], detection[6]);
        return rect;
    }
    std::string ToString()
    {
        std::stringstream s;
        for (int i = 0; i < detection.size(); i++)
        {
            s<<detection[i]<<", ";
        }
        return s.str();
    }
};

class Detector
{
public:
    Detector(const string &model_file,
             const string &weights_file);
    void initLayers();
    void transformInput(const cv::Mat &img, cv::Mat *output);
    vector<SSD_Out> forwardNet(cv::Mat *input);
    vector<SSD_Out> serialDetector(const cv::Mat &img);

private:
    void WrapInputLayer(std::vector<cv::Mat> *input_channels);

private:
    shared_ptr<Net<float>> net_;
    std::vector<cv::Mat> *input_channels;
    //Blob<float> *input_layer;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

Detector::Detector(const string &model_file,
                   const string &weights_file)
{
    std::cout << "1";
    //Caffe::set_mode(Caffe::CPU);
    Caffe::set_mode(Caffe::GPU);
    std::cout << "2";
    /* Load the network. */
    std::cout << "model: " << model_file << std::endl;
    net_.reset(new Net<float>(model_file, TEST)); //prototxt
    std::cout << "weight: " << weights_file << std::endl;
    try
    {
        net_->CopyTrainedLayersFrom(weights_file); //caffemodel
    }
    catch (const std::exception &e)
    {                          // reference to the base of a polymorphic object
        std::cout << e.what(); // information from length_error printed
        return;
    }
    net_->CopyTrainedLayersFrom(weights_file); //caffemodel
    std::cout << "3";
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    std::cout << "4";

    Blob<float> *input_layer = net_->input_blobs()[0];
    std::cout << "5";

    num_channels_ = input_layer->channels();
    std::cout << "6";

    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    std::cout << "8";

    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    std::cout << "9";

    net_->Reshape();
    WrapInputLayer(input_channels);
    std::cout << "fin";
}

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
