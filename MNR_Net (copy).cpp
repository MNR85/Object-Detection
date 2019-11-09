#ifndef MNRNET
#define MNRNET
#include "MNR_Net.hpp"

vector<SSD_Out> Detector::forwardNet(cv::Mat *input)
{
    std::cout << "1";
    std::vector<cv::Mat> *input_channels;
    cv::split(*input, *input_channels); /* This operation will write the separate BGR planes directly to the input layer of the network because it is wrapped by the cv::Mat objects in input_channels. */
    std::cout << "2";
    net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float> *result_blob = net_->output_blobs()[0];
    const float *result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<SSD_Out> detections;
    for (int k = 0; k < num_det; ++k)
    {
        vector<float> detection(result, result + 7);
        SSD_Out detectionS(detection);
        detections.push_back(detectionS);
        result += 7;
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

vector<SSD_Out> Detector::serialDetector(const cv::Mat &img)
{
    cv::Mat sample_normalized;
    std::cout << "4";
    transformInput(img, &sample_normalized); /* Normalize input image: resize, subtract, multiply */
    std::cout << "3";
    return forwardNet(&sample_normalized);
}

void Detector::initLayers()
{
    // std::cout<<"1111";
    // std::cout << "re";
    // input_layer->Reshape(1, num_channels_,
    //                      input_geometry_.height, input_geometry_.width);
    // std::cout << "n re";
    // /* Forward dimension change to all layers. */
    // net_->Reshape();
    // std::cout<<"wr";
    // WrapInputLayer(input_channels);
}

Detector::Detector(const string &model_file,
                   const string &weights_file)
{
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

#endif