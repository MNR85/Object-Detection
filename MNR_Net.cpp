#ifndef MNRNET
#define MNRNET
#include "MNR_Net.hpp"
//#include <pthread.h>
#ifndef DEBUG1
#define DEBUG1
#endif
// #include <thread>
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
    //cv::split(*input, *&input_channels); /* This operation will write the separate BGR planes directly to the input layer of the network because it is wrapped by the cv::Mat objects in input_channels. */
    // if (reinterpret_cast<float *>((&input_channels)->at(0).data) == net_->input_blobs()[0]->cpu_data())
    //   std::cout << "Eqyal changgel";
    // else
    //   std::cout << "Input channels are not wrapping the input layer of the network.";
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

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
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
vector<vector<float>> Detector::serialDetector(const cv::Mat &img) //, std::vector<cv::Mat>* input_channels)
{
    clock_t t11, t21, t22, t33;

    t11 = clock();
    //cv::Mat sample_normalized;
    std::vector<cv::Mat> input_channels;

#ifdef DEBUG
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    WrapInputLayer(&input_channels);

#ifdef DEBUG
    auto t2 = std::chrono::high_resolution_clock::now();
#endif
    t21 = clock();
    transformInput(img, &input_channels); /* Normalize input image: resize, subtract, multiply */
    t22 = clock();

#ifdef DEBUG
    auto t3 = std::chrono::high_resolution_clock::now();
#endif

    vector<vector<float>> res = forwardNet(); //&sample_normalized);
    t33 = clock();
    trasformClocks.push(double(t22 - t21));
    netClocks.push(double(t21 - t11));
    netClocks.push(double(t33 - t22));

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
    clock_t t1, t2;
    t1 = clock();
    cv::Mat converted = transformInputGet(img);
    t2 = clock();
    trasformClocks.push(double(t2 - t1));
    mtx.lock();
    normilizedImages.push(converted);
    mtx.unlock();
}
//for pipeline
vector<vector<float>> Detector::getImageFromQ()
{
    clock_t t1, t2, t3;
    t1 = clock();
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
    t2 = clock();
    vector<vector<float>> res = forwardNet();
    t3 = clock();
    netClocks.push(double(t2 - t1));
    netClocks.push(double(t3 - t2));
    return res;
}
//for pipeline
void Detector::getImageFromQThread()
{
    configGPUusage();
    while (runThread || !normilizedImages.empty())
    {
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
#ifdef DEBUG
        auto t2 = std::chrono::high_resolution_clock::now();
        int duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << ", Time: " << duration1 / 1000.0 << "ms, FPS: " << 1000000.0 / duration1 << std::endl;
#endif
        ;
    }
    std::cout << "Finished thread." << std::endl;
}
// std::thread *Detector::runNetThread()
// {
//     std::cout << "start thread." << std::endl;
//     std::thread popThread(&Detector::getImageFromQThread, this); // spawn new thread that calls getImageFromQThread()
//     return &popThread;                                           //.join();
// }
// void Detector::feedNetwork(std::vector<cv::Mat> *input_channels)
// {
//     if (!normilizedImages.empty())
//     {
//         cv::Mat normalImage = normilizedImages.front();
//         normilizedImages.pop();
//         /* This operation will write the separate BGR planes directly to the
//    * input layer of the network because it is wrapped by the cv::Mat
//    * objects in input_channels. */
//         cv::split(normalImage, *input_channels);
//     }
// }

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

void Detector::saveDataToFiles(string fileName)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::ofstream myfile;
    myfile.open(fileName + ".csv");
    myfile << "GPU use = "<<useGPU<<"\n";
    myfile << "Name: "<<prop.name << ", totalGlobalMem: "<<prop.totalGlobalMem<< ", totalConstMem: "<<prop.totalConstMem  <<", multiProcessorCount: "<< prop.multiProcessorCount <<", clockRate: "<<prop.clockRate<<"\n";
    myfile << "transTime,feedNet,netTime\n";
    myfile << "clockPerSec: ," << (CLOCKS_PER_SEC) << "\n";
    for (int i = 0; i < trasformClocks.size(); i++)
    {
        int transTime = trasformClocks.front();
        trasformClocks.pop();
        int feedNet = netClocks.front();
        netClocks.pop();
        int netTime = netClocks.front();
        netClocks.pop();
        myfile << transTime << "," << feedNet << "," << netTime << "\n";
    }
    myfile.close();
}
#endif