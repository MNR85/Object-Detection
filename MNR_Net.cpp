#ifndef MNRNET
#define MNRNET
#include "MNR_Net.hpp"
#ifndef DEBUG1
#define DEBUG1
#endif
#ifndef CaptureClock
#define CaptureClock
#endif
#ifndef CaptureTime
#define CaptureTime
#endif
#include <iostream>
#include <fstream>
Detector::Detector(const string &model_file1,
                   const string &weights_file1)
{
    model_file = model_file1;
    weights_file = weights_file1;
    initNet();
}
void Detector::setRunMode(bool useGPU1)
{
    useGPU = useGPU1;
}
void Detector::configGPUusage()
{
    if (useGPU)
        Caffe::set_mode(Caffe::GPU);
    else
        Caffe::set_mode(Caffe::CPU);
}
void Detector::initNet()
{
    configGPUusage();
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    net_->Reshape();
}
/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}
vector<vector<float>> Detector::forwardNet() //cv::Mat *input)
{
    net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float> *result_blob = net_->output_blobs()[0];
    const float *result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float>> detections;
    try
    {
        /* code */
        for (int k = 0; k < num_det; ++k)
        {
            if (result[0] == -1)
            {
                // Skip invalid detection.
                result += 7;
                continue;
            }
            vector<float> detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "In forwardNet(), collecting result - " << e.what() << '\n';
    }

    return detections;
}
void Detector::transformInput(const cv::Mat &img, cv::Mat *output)
{
    cv::Mat sample_resized;
    if (img.size() != input_geometry_)
        cv::resize(img, sample_resized, input_geometry_);
    else
        sample_resized = img;
    sample_resized.convertTo(sample_resized, CV_64FC3);
    sample_resized = sample_resized - 127.5;
    sample_resized = sample_resized * 0.007843;
    sample_resized.convertTo(sample_resized, CV_32FC3);
    output = &sample_resized;
}
//used in serial detector
void Detector::transformInput(const cv::Mat &img, std::vector<cv::Mat> *input_channels)
{
    cv::Mat sample_resized;
    if (img.size() != input_geometry_)
        cv::resize(img, sample_resized, input_geometry_);
    else
        sample_resized = img;
    sample_resized.convertTo(sample_resized, CV_64FC3);
    sample_resized = sample_resized - 127.5;
    sample_resized = sample_resized * 0.007843;
    sample_resized.convertTo(sample_resized, CV_32FC3);
    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_resized, *input_channels);

    // CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
    //     << "Input channels are not wrapping the input layer of the network.";
}
//uesd in  queue
cv::Mat Detector::transformInputGet(const cv::Mat &img)
{
    cv::Mat sample_resized;
    if (img.size() != input_geometry_)
        cv::resize(img, sample_resized, input_geometry_);
    else
        sample_resized = img;
    sample_resized.convertTo(sample_resized, CV_64FC3);
    sample_resized = sample_resized - 127.5;
    sample_resized = sample_resized * 0.007843;
    sample_resized.convertTo(sample_resized, CV_32FC3);
    return sample_resized;
}
void saveMatToFile(std::vector<cv::Mat> *input_channels, string fileName)
{
    std::ofstream fout(fileName);
    if (!fout)
    {
        std::cout << "File Not Opened" << std::endl;
        return;
    }
    for (int k = 0; k < input_channels->size(); k++)
    {
        fout << "Channel " << k << std::endl;
        for (int i = 0; i < input_channels->at(k).rows; i++)
        {
            for (int j = 0; j < input_channels->at(k).cols; j++)
            {
                fout << input_channels->at(k).at<float>(i, j) << ", ";
            }
            fout << std::endl;
        }
    }

    fout.close();
}
void Detector::GetDataLayer(std::vector<cv::Mat> *input_channels, int layerNum)
{
    shared_ptr<Layer<float>> la = net_->layer_by_name("conv3");
    shared_ptr<Blob<float>> input_layer = net_->blobs()[layerNum];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    std::cout << "111" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << "data:" << input_data[i] << ", ";
    }
    std::cout << std::endl;
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat ch = cv::Mat(height, width, 5);
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                float a = float(input_data[i * height * width + j * width + k]);
                ch.at<float>(1, 1) = a;
            }
        }
        input_channels->push_back(ch);
    }
}
void Detector::SetDataLayer(std::vector<cv::Mat> *input_channels, int layerNum)
{
    shared_ptr<Blob<float>> input_layer = net_->blobs()[layerNum];
    // float a[] = {85, 85, 85};
    // try
    // {
    //     input_layer->set_cpu_data(a);
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "In forwardNet(), collecting result - " << e.what() << '\n';
    // }
    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    // float *input_data4 = net_->blobs()[layerNum]->mutable_cpu_data();
    // for (int i = 0; i < input_layer->channels(); ++i)
    //     for (int j = 0; j < height; j++)
    //         for (int k = 0; k < width; k++)
    //             std::cout << "z: " << input_data4[i] << ", ";
    // std::cout << std::endl;
    float data[input_layer->channels()*input_layer->width()*input_layer->height()];
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat ch = input_channels->at(i);
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                input_data[i * height * width + j * width + k] = ch.at<float>(j, k);
            }
        }
    }
    // net_->blobs()[layerNum]->cpu_data()
    // std::cout<<"333"<<std::endl;
    // net_->blobs()[layerNum]->set_cpu_data(data);
    //     std::cout<<"444"<<std::endl;

    // float *input_data2 = net_->blobs()[layerNum]->mutable_cpu_data();
    // for (int i = 0; i < input_layer->channels(); ++i)
    //     for (int j = 0; j < height; j++)
    //         for (int k = 0; k < width; k++)
    //             std::cout << "x: " << input_data2[i] << ", ";
    // std::cout << std::endl;
    // net_->blobs()[layerNum]->set_cpu_data(data);
    // net_->blobs()[layerNum]->Update();
    // float *input_data3 = net_->blobs()[layerNum]->mutable_cpu_data();
    // for (int i = 0; i < input_layer->channels(); ++i)
    //     for (int j = 0; j < height; j++)
    //         for (int k = 0; k < width; k++)
    //             std::cout << "y: " << input_data3[i] << ", ";
    // std::cout << std::endl;
}
vector<vector<float>> Detector::serialDetectorMultiStage(const cv::Mat &img, const cv::Mat &img2)
{
    std::vector<cv::Mat> input_channels;
    std::vector<cv::Mat> input_channels2;
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat temp(height, width, CV_32FC1);
        input_channels2.push_back(temp);
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    //image 1
    transformInput(img, &input_channels); /* Normalize input image: resize, subtract, multiply */
    net_->ForwardTo(14);
    std::vector<cv::Mat> mid1;
    GetDataLayer(&mid1, 14);
    saveMatToFile(&mid1, "mid1.data");
    //image 2
    saveMatToFile(&input_channels, "input_channels.data");
    transformInput(img2, &input_channels); /* Normalize input image: resize, subtract, multiply */
    saveMatToFile(&input_channels, "input_channels1.data");

    net_->ForwardTo(14);
    // saveMatToFile(&mid1, "mid12.data");
    // std::vector<cv::Mat> mid2;
    // GetDataLayer(&mid2, 14);
    // saveMatToFile(&mid2, "mid2.data");
    // saveMatToFile(&mid1, "mid11.data");
    //image 1
    SetDataLayer(&mid1, 14);
    // std::cout << "Now getting data!" << std::endl;
    // GetDataLayer(&mid1, 14);
    // std::cout << "Got!" << std::endl;
    // saveMatToFile(&mid1, "mid111.data");
    // std::cout << "Saved!" << std::endl;
    net_->ForwardFrom(14);
    //net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float> *result_blob = net_->output_blobs()[0];
    const float *result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float>> detections;
    try
    {
        /* code */
        for (int k = 0; k < num_det; ++k)
        {
            if (result[0] == -1)
            {
                // Skip invalid detection.
                result += 7;
                continue;
            }
            vector<float> detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "In forwardNet(), collecting result - " << e.what() << '\n';
    }

    return detections;
}
vector<vector<float>> Detector::serialDetectorMultiStage(const cv::Mat &img)
{
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    transformInput(img, &input_channels); /* Normalize input image: resize, subtract, multiply */
    net_->ForwardTo(14);
    std::vector<cv::Mat> mid1;
    GetDataLayer(&mid1, 14);
    SetDataLayer(&mid1, 14);
    // shared_ptr<Blob<float>> mid_blob = net_->blobs()[14];
    // //mid_blob->
    // float *resultMid = mid_blob->mutable_cpu_data();
    // const float *resultMid = mid_blob->cpu_data();
    // float *r1 = const_cast<float *>(resultMid);
    // // net_->blobs()[14]->set_cpu_data(r1);
    net_->ForwardFrom(14);
    //net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float> *result_blob = net_->output_blobs()[0];
    const float *result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float>> detections;
    try
    {
        /* code */
        for (int k = 0; k < num_det; ++k)
        {
            if (result[0] == -1)
            {
                // Skip invalid detection.
                result += 7;
                continue;
            }
            vector<float> detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "In forwardNet(), collecting result - " << e.what() << '\n';
    }

    return detections;
}
vector<vector<float>> Detector::pipelineDetectorMultiStageButWorkSerial(const cv::Mat &img)
{
}
vector<vector<float>> Detector::serialDetector(const cv::Mat &img) //, std::vector<cv::Mat>* input_channels)
{
#ifdef CaptureClock
    clock_t t11, t21, t22, t33;
    t11 = clock();
#endif
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t111 = std::chrono::high_resolution_clock::now();
#endif
    //cv::Mat sample_normalized;
    std::vector<cv::Mat> input_channels;

#ifdef DEBUG
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    WrapInputLayer(&input_channels);

#ifdef DEBUG
    auto t2 = std::chrono::high_resolution_clock::now();
#endif
#ifdef CaptureClock
    t21 = clock();
#endif
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t221 = std::chrono::high_resolution_clock::now();
#endif

    transformInput(img, &input_channels); /* Normalize input image: resize, subtract, multiply */
#ifdef CaptureClock
    t22 = clock();
#endif
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t222 = std::chrono::high_resolution_clock::now();
#endif
#ifdef DEBUG
    auto t3 = std::chrono::high_resolution_clock::now();
#endif

    vector<vector<float>> res = forwardNet(); //&sample_normalized);
#ifdef CaptureClock
    t33 = clock();
#endif
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t333 = std::chrono::high_resolution_clock::now();
#endif
#ifdef CaptureClock
    trasformClocks.push(double(t22 - t21));
    netClocks.push(double(t21 - t11));
    netClocks.push(double(t33 - t22));
#endif
#ifdef CaptureTime
    trasformTimes.push(std::chrono::duration_cast<std::chrono::microseconds>(t222 - t221).count());
    netTimes.push(std::chrono::duration_cast<std::chrono::microseconds>(t221 - t111).count());
    netTimes.push(std::chrono::duration_cast<std::chrono::microseconds>(t333 - t222).count());
#endif

#ifdef DEBUG
    auto t4 = std::chrono::high_resolution_clock::now();
    int duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    int duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    int duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    int durationTotal = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count();
    std::cout << "Serial detector: "
              << "Times: " << duration1 / 1000.0 << ", " << duration2 / 1000.0 << ", " << duration3 / 1000.0 << ", Total: " << durationTotal / 1000.0 << "ms, FPS: " << 1000000.0 / durationTotal << std::endl;
#endif
    return res;
}

//for pipeline
void Detector::addImageToQ(const cv::Mat &img)
{
#ifdef CaptureClock
    clock_t t1, t2;
#endif
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t11 = std::chrono::high_resolution_clock::now();
#endif
#ifdef CaptureClock
    t1 = clock();
#endif
    cv::Mat converted = transformInputGet(img);
#ifdef CaptureClock
    t2 = clock();
#endif
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t22 = std::chrono::high_resolution_clock::now();
    trasformTimes.push(std::chrono::duration_cast<std::chrono::microseconds>(t22 - t11).count());
#endif
#ifdef CaptureClock
    trasformClocks.push(double(t2 - t1));
#endif
    mtx.lock();
    normilizedImages.push(converted);
    mtx.unlock();
}
void Detector::getImageFromQThreadMultiStage2()
{
    configGPUusage();
    while (runThread || !stage1.empty())
    {
        if (stage1.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        stage1.push(getImageFromQMultiStage());
    }
}
void Detector::getImageFromQThreadMultiStage()
{
    configGPUusage();
    while (runThread || !normilizedImages.empty())
    {
        if (normilizedImages.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        detectionOutputs.push(getImageFromQMultiStage2());
    }
}
vector<vector<float>> Detector::getImageFromQMultiStage2()
{
    std::vector<cv::Mat> sample_resized;
    mtx.lock();
    sample_resized = stage1.front();
    stage1.pop();
    mtx.unlock();
    SetDataLayer(&sample_resized, 14);

    net_->ForwardFrom(14);
    //net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float> *result_blob = net_->output_blobs()[0];
    const float *result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float>> detections;
    try
    {
        /* code */
        for (int k = 0; k < num_det; ++k)
        {
            if (result[0] == -1)
            {
                // Skip invalid detection.
                result += 7;
                continue;
            }
            vector<float> detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "In forwardNet(), collecting result - " << e.what() << '\n';
    }

    return detections;
}
std::vector<cv::Mat> Detector::getImageFromQMultiStage()
{
    cv::Mat sample_resized;
    mtx.lock();
    sample_resized = normilizedImages.front();
    normilizedImages.pop();
    mtx.unlock();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_resized, input_channels);

    net_->ForwardTo(14);
    std::vector<cv::Mat> mid1;
    GetDataLayer(&mid1, 14);
    return mid1;
}
//for pipeline
vector<vector<float>> Detector::getImageFromQ()
{
#ifdef CaptureClock
    clock_t t1, t2, t3;
#endif
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t11 = std::chrono::high_resolution_clock::now();
#endif
#ifdef CaptureClock
    t1 = clock();
#endif
    cv::Mat sample_resized;
    mtx.lock();
    sample_resized = normilizedImages.front();
    normilizedImages.pop();
    mtx.unlock();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_resized, input_channels);
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t22 = std::chrono::high_resolution_clock::now();
#endif
#ifdef CaptureClock
    t2 = clock();
#endif
    vector<vector<float>> res = forwardNet();
#ifdef CaptureTime
    std::chrono::high_resolution_clock::time_point t33 = std::chrono::high_resolution_clock::now();
#endif
#ifdef CaptureClock
    t3 = clock();
#endif
#ifdef CaptureTime
    netTimes.push(std::chrono::duration_cast<std::chrono::microseconds>(t22 - t11).count());
    netTimes.push(std::chrono::duration_cast<std::chrono::microseconds>(t33 - t22).count());
#endif
#ifdef CaptureClock
    netClocks.push(double(t2 - t1));
    netClocks.push(double(t3 - t2));
#endif
    return res;
}
//for pipeline
void Detector::getImageFromQThread()
{
    configGPUusage();
    while (runThread || !normilizedImages.empty())
    {
        std::chrono::high_resolution_clock::time_point t11 = std::chrono::high_resolution_clock::now();

        if (normilizedImages.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
#ifdef DEBUG
        std::cout << "queue size: " << normilizedImages.size();
        auto t1 = std::chrono::high_resolution_clock::now();
#endif
        detectionOutputs.push(getImageFromQ());
        std::chrono::high_resolution_clock::time_point t22 = std::chrono::high_resolution_clock::now();
        thread2Times.push(std::chrono::duration_cast<std::chrono::microseconds>(t22 - t11).count());
#ifdef DEBUG
        auto t2 = std::chrono::high_resolution_clock::now();
        int duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << ", Time: " << duration1 / 1000.0 << "ms, FPS: " << 1000000.0 / duration1 << std::endl;
#endif
        ;
    }
    std::cout << "Finished thread." << std::endl;
}
//for test pipeline work only!!!
vector<vector<float>> Detector::pipelineDetectorButWorkSerial(const cv::Mat &img) //, std::vector<cv::Mat>* input_channels)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    addImageToQ(img);
    auto t2 = std::chrono::high_resolution_clock::now();

    vector<vector<float>> res = getImageFromQ();
    auto t3 = std::chrono::high_resolution_clock::now();
    int duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    int duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    int durationTotal = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count();

    std::cout << "Pipeline detector: "
              << "Times: Preprocess: " << duration1 / 1000.0 << ", Network: " << duration2 / 1000.0 << ", Total: " << durationTotal / 1000.0 << "ms, FPS: " << 1000000.0 / durationTotal << std::endl;

    return res;
}
void Detector::clearLogs()
{
    std::queue<double> empty;
    trasformClocks.swap(empty);
    std::queue<double> empty2;
    netClocks.swap(empty2);
}

void Detector::newPreprocess(int time)
{
    thread1Times.push(time);
}

void Detector::saveDataToFiles(string fileName, string moreInfo, int frameCount, bool isSerial)
{

    std::ofstream myfile;
    myfile.open(fileName + ".csv", ios::out | ios::app);

    myfile << moreInfo << "\n";

    myfile << "GPU use = " << useGPU << "\n";
    if (useGPU)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        myfile << "Name"
               << ", asyncEngineCount"
               << ", maxThreadsPerMultiProcessor"
               << ", totalGlobalMem"
               << ", totalConstMem"
               << ", unifiedAddressing"
               << ", multiProcessorCount"
               << ", clockRate"
               << ", concurrentKernels"
               << ", deviceOverlap"
               << "\n";
        myfile << prop.name << ", " << prop.asyncEngineCount << ", " << prop.maxThreadsPerMultiProcessor << ", " << prop.totalGlobalMem << ", " << prop.totalConstMem << ", " << prop.unifiedAddressing << ", " << prop.multiProcessorCount << ", " << prop.clockRate << ", " << prop.concurrentKernels << ", " << prop.deviceOverlap << "\n";
    }
    myfile << "clockPerSec," << (CLOCKS_PER_SEC) << "\n";
    myfile << "FPS=" << FPS << "\n";
#ifdef CaptureTime
    myfile << "transTime, feedNetTime, netTime, ";
    if (!isSerial)
        myfile << "thread#1, thread#2, ";

#endif
#ifdef CaptureClock
    myfile << "transClock, feedNetClock, netClock";
#endif
    myfile << "\n";
    for (int i = 0; frameCount > i; i++)
    {
#ifdef CaptureTime
        int transTime = trasformTimes.front();
        trasformTimes.pop();
        int feedNetTime = netTimes.front();
        netTimes.pop();
        int netTime = netTimes.front();
        netTimes.pop();
        int thread1Time = 0;
        int thread2Time = 0;
        if (!isSerial)
        {
            thread1Time = thread1Times.front();
            thread1Times.pop();
            thread2Time = thread2Times.front();
            thread2Times.pop();
        }

#endif
#ifdef CaptureClock
        int transClock = trasformClocks.front();
        trasformClocks.pop();
        int feedNetClock = netClocks.front();
        netClocks.pop();
        int netClock = netClocks.front();
        netClocks.pop();
#endif
#ifdef CaptureTime
        myfile << transTime << ", " << feedNetTime << ", " << netTime << ", ";
        if (!isSerial)
            myfile << thread1Time << ", " << thread2Time << ", ";
#endif
#ifdef CaptureClock
        myfile << transClock << ", " << feedNetClock << ", " << netClock;
#endif
        myfile << "\n";
    }
    myfile << "----------------\n\n";
    myfile.close();
}
#endif
