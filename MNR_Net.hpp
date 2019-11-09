#ifndef MNRNET_H
#define MNRNET_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include "SSD_Out.hpp"
#include <chrono>
#include "queue"

using namespace caffe;

class Detector
{
public:
    Detector(const string &model_file,
             const string &weights_file);

    vector<vector<float>> serialDetector(const cv::Mat &img); //, std::vector<cv::Mat> *input_channels);
    vector<vector<float>> pipelineDetector(const cv::Mat &img);
    void addImageToQ(const cv::Mat &img);
    void feedNetwork(std::vector<cv::Mat> *input_channels);
private:
    void transformInput(const cv::Mat &img, cv::Mat *output);
    void transformInput(const cv::Mat &img, std::vector<cv::Mat> *input_channels);
    cv::Mat transformInputGet(const cv::Mat &img);
    vector<vector<float>> forwardNet();//cv::Mat *input);
    void WrapInputLayer(std::vector<cv::Mat> *input_channels);

    shared_ptr<Net<float>> net_;
    //std::vector<cv::Mat> input_channels;

    //std::vector<cv::Mat> *input_channels;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::queue<cv::Mat> normilizedImages;
};

#endif