#ifndef MNRNET_H
#define MNRNET_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <chrono>
#include "queue"
#include <thread> // std::thread
#include <mutex>  // std::mutex
using namespace caffe;

class Detector
{
public:
    Detector(const string &model_file,
             const string &weights_file);

    vector<vector<float>> serialDetector(const cv::Mat &img);
    vector<vector<float>> pipelineDetectorButWorkSerial(const cv::Mat &img);
    void addImageToQ(const cv::Mat &img);
    void getImageFromQThread();

    bool runThread = true;
    double FPS = 0;
    void clearLogs();
    std::queue<vector<vector<float>>> detectionOutputs;

    void setRunMode(bool useGPU);

    void saveDataToFiles(string fileName, string moreInfo, int frameCount);

private:
    void initNet();
    void transformInput(const cv::Mat &img, cv::Mat *output);
    void transformInput(const cv::Mat &img, std::vector<cv::Mat> *input_channels);
    cv::Mat transformInputGet(const cv::Mat &img);
    vector<vector<float>> forwardNet(); //cv::Mat *input);
    void WrapInputLayer(std::vector<cv::Mat> *input_channels);

    vector<vector<float>> getImageFromQ();

    shared_ptr<Net<float>> net_;

    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::queue<cv::Mat> normilizedImages;
    std::mutex mtx; // mutex for critical section

    string model_file;
    string weights_file;
    std::queue<double> netClocks;
    std::queue<int> netTimes;
    std::queue<double> trasformClocks;
    std::queue<int> trasformTimes;

    bool useGPU = false;
    void configGPUusage();
};

#endif