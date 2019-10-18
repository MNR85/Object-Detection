#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>

using namespace caffe;

class Detector
{
public:
    Detector(const string &model_file,
             const string &weights_file,
             const string &mean_file,
             const string &mean_value);

    std::vector<vector<float>> Detect(const cv::Mat &img);

private:
/* Load the mean file in binaryproto format. */
    void SetMean(const string &mean_file, const string &mean_value);

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
    void WrapInputLayer(std::vector<cv::Mat> *input_channels);

    void Preprocess(const cv::Mat &img,
                    std::vector<cv::Mat> *input_channels);

private:
    shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};