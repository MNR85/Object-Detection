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
    vector<vector<float>> serialDetectorMultiStage(const cv::Mat &img);
    vector<vector<float>> serialDetectorMultiStage(const cv::Mat &img, const cv::Mat &img2);
    vector<vector<float>> pipelineDetectorButWorkSerial(const cv::Mat &img);
    vector<vector<float>> pipelineDetectorMultiStageButWorkSerial(const cv::Mat &img);
    void addImageToQ(const cv::Mat &img);
    void getImageFromQThread();
    void getImageFromQThreadMultiStage();
    void getImageFromQThreadMultiStage2();

    bool runThread = true;
    double FPS = 0;
    void clearLogs();
    std::queue<vector<vector<float>>> detectionOutputs;

    void setRunMode(bool useGPU);

    void newPreprocess(int time);
    void saveDataToFiles(string fileName, string moreInfo, int frameCount, bool isSerial);

private:
    void initNet();
    void transformInput(const cv::Mat &img, cv::Mat *output);
    void transformInput(const cv::Mat &img, std::vector<cv::Mat> *input_channels);
    cv::Mat transformInputGet(const cv::Mat &img);
    vector<vector<float>> forwardNet(); //cv::Mat *input);
    void WrapInputLayer(std::vector<cv::Mat> *input_channels);
    void GetDataLayer(std::vector<cv::Mat> *input_channels, int layerNum);
    void SetDataLayer(std::vector<cv::Mat> *input_channels, int layerNum);
    vector<vector<float>> getImageFromQ();
    std::vector<cv::Mat> getImageFromQMultiStage();
    vector<vector<float>> getImageFromQMultiStage2();
    shared_ptr<Net<float>> net_;

    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::queue<cv::Mat> normilizedImages;
    std::queue<std::vector<cv::Mat>> stage1;
    std::mutex mtx; // mutex for critical section

    string model_file;
    string weights_file;
    std::queue<double> netClocks;
    std::queue<int> netTimes;
    std::queue<double> trasformClocks;
    std::queue<int> trasformTimes;
    std::queue<int> thread1Times;
    std::queue<int> thread2Times;

    bool useGPU = false;
    void configGPUusage();
};

#endif