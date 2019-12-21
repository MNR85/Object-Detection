#ifndef MNRNET_H
#define MNRNET_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include "SSD_Out.hpp"

using namespace caffe;

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
    /////std::vector<cv::Mat> *input_channels;
    //Blob<float> *input_layer;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

#endif