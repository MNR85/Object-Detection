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

#include <chrono>

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
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
      s << detection[i] << ", ";
    }
    return s.str();
  }
};
class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file);

  std::vector< vector<float> > Detect(const cv::Mat& img);

 private:

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file) {
  //Caffe::set_mode(Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}

std::vector<vector<float>> Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }

  return detections;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
auto t1 = std::chrono::high_resolution_clock::now();
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  std::cout<<"Channels: "<<img.channels()<<" , num_channels: "<<num_channels_<<std::endl;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else{
      std::cout<<"is in else"<<std::endl;
    sample = img;

  }
  auto t2 = std::chrono::high_resolution_clock::now();
 //cv::cvtColor(img, sample, cv::COLOR_BGR2Luv);
// sample = img;
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;
auto t3 = std::chrono::high_resolution_clock::now();
if (num_channels_ == 3)
    sample_resized.convertTo(sample_resized, CV_64FC3);
  else
    sample_resized.convertTo(sample_resized, CV_64FC1);
    auto t4 = std::chrono::high_resolution_clock::now();
  cv::Mat sample_subed = sample_resized- 127.5;
  auto t5 = std::chrono::high_resolution_clock::now();
  cv::Mat sample_muled = sample_subed*0.007843;
auto t6 = std::chrono::high_resolution_clock::now();
    cv::FileStorage sample_resizedfile("sample_resizedfile.yml", cv::FileStorage::WRITE);
auto t7 = std::chrono::high_resolution_clock::now();

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_muled.convertTo(sample_float, CV_32FC3);
  else
    sample_muled.convertTo(sample_float, CV_32FC1);
  //sample_resizedfile << "sample_resized" << sample_resized<< "sample_subed" << sample_subed<< "sample_muled" << sample_muled<<"sample_float"<<sample_float;

  cv::Mat sample_normalized=sample_float;
  //cv::subtract(sample_float, mean_, sample_normalized);
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");
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
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }
    std::cout<<"mean: "<<FLAGS_mean_file<<std::endl;
  string classes []= {"background",
           "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"};
  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  std::ifstream infile(argv[3]);
  std::string file;
  while (infile >> file) {
    if (file_type == "image") {
      cv::Mat img = cv::imread(file);
      CHECK(!img.empty()) << "Unable to decode image " << file;
      std::vector<vector<float>> detections = detector.Detect(img);
      for (int i = 0; i < detections.size(); ++i)
      {
        // if (detections[i].isValidDetection())
        // {
        //   drawDetections(img, detections[i]);
        //   std::cout << "Detected:\t\t" << detections[i].ToString();
        // }
        // else
        // {
        //   std::cout << "False Detection:\t\t" << detections[i].ToString();
        // }
      }
      }

      
  }
  return 0;
}