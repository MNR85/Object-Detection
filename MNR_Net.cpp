#ifndef MNRNET
#define MNRNET
#include "MNR_Net.hpp"
//#include <pthread.h>

// #include <thread>
Detector::Detector(const string &model_file,
                   const string &weights_file)
{
    //Caffe::set_mode(Caffe::CPU);
    Caffe::set_mode(Caffe::GPU);
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
    ///////std::cout << "XXC";
    net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float> *result_blob = net_->output_blobs()[0];
    const float *result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float>> detections;
    try
    {
        /* code */
        ///////std::cout << "XXB";
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
        ///////std::cout << "XXA";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
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
    //cv::Mat sample_normalized;
    std::vector<cv::Mat> input_channels;

    auto t1 = std::chrono::high_resolution_clock::now();

    WrapInputLayer(&input_channels);

    auto t2 = std::chrono::high_resolution_clock::now();

    transformInput(img, &input_channels); /* Normalize input image: resize, subtract, multiply */

    auto t3 = std::chrono::high_resolution_clock::now();

    vector<vector<float>> res = forwardNet(); //&sample_normalized);

    auto t4 = std::chrono::high_resolution_clock::now();
    int duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    int duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    int duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    int durationTotal = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count();
    std::cout << "Serial detector: "
              << "Times: " << duration1 / 1000.0 << ", " << duration2 / 1000.0 << ", " << duration3 / 1000.0 << ", Total: " << durationTotal / 1000.0 << "ms, FPS: " << 1000000.0 / durationTotal << std::endl;
    return res;
}

void Detector::addImageToQ(const cv::Mat &img)
{
    cv::Mat converted = transformInputGet(img);
    mtx.lock();
    normilizedImages.push(converted);
    mtx.unlock();
}

vector<vector<float>> Detector::getImageFromQ()
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
    return forwardNet();
}

void Detector::getImageFromQThread()
{
    while (runThread || !normilizedImages.empty())
    {
        if (normilizedImages.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        std::cout << "queue size: " << normilizedImages.size() << std::endl;
        detectionOutputs.push(getImageFromQ());
    }
    std::cout << "Finished thread." << std::endl;
}
// std::thread *Detector::runNetThread()
// {
//     std::cout << "start thread." << std::endl;
//     std::thread popThread(&Detector::getImageFromQThread, this); // spawn new thread that calls getImageFromQThread()
//     return &popThread;                                           //.join();
// }
void Detector::feedNetwork(std::vector<cv::Mat> *input_channels)
{
    if (!normilizedImages.empty())
    {
        cv::Mat normalImage = normilizedImages.front();
        normilizedImages.pop();
        /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
        cv::split(normalImage, *input_channels);
    }
}
vector<vector<float>> Detector::pipelineDetector(const cv::Mat &img) //, std::vector<cv::Mat>* input_channels)
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
#endif