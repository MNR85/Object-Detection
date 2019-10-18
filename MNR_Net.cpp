#include "MNR_Net.hpp"
#include <opencv2/imgproc/imgproc.hpp>

Detector::Detector(const string &model_file,
                   const string &weights_file,
                   const string &mean_file,
                   const string &mean_value)
{
    //Caffe::set_mode(Caffe::CPU);
    Caffe::set_mode(Caffe::GPU);
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));   //prototxt
    net_->CopyTrainedLayersFrom(weights_file);      //caffemodel

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value); // fill "mean_"
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string &mean_file, const string &mean_value)
{
    cv::Scalar channel_mean;
    if (!mean_file.empty())
    {
        CHECK(mean_value.empty()) << "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float *data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i)
        {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty())
    {
        CHECK(mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ','))
        {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) << "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i)
        {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
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

void Detector::Preprocess(const cv::Mat &img,
                          std::vector<cv::Mat> *input_channels)
{
    // auto t1 = std::chrono::high_resolution_clock::now();
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    std::cout << "Channels: " << img.channels() << " , num_channels: " << num_channels_ << std::endl;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
    {
        std::cout << "is in else" << std::endl;
        sample = img;
    }
    //   auto t2 = std::chrono::high_resolution_clock::now();
    //cv::cvtColor(img, sample, cv::COLOR_BGR2Luv);
    // sample = img;
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;
    // auto t3 = std::chrono::high_resolution_clock::now();
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_resized, CV_64FC3);
    else
        sample_resized.convertTo(sample_resized, CV_64FC1);
    // auto t4 = std::chrono::high_resolution_clock::now();
    cv::Mat sample_subed = sample_resized - 127.5;
    //   auto t5 = std::chrono::high_resolution_clock::now();
    cv::Mat sample_muled = sample_subed * 0.007843;
    // auto t6 = std::chrono::high_resolution_clock::now();
    cv::FileStorage sample_resizedfile("sample_resizedfile.yml", cv::FileStorage::WRITE);
    // auto t7 = std::chrono::high_resolution_clock::now();

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_muled.convertTo(sample_float, CV_32FC3);
    else
        sample_muled.convertTo(sample_float, CV_32FC1);
    //sample_resizedfile << "sample_resized" << sample_resized<< "sample_subed" << sample_subed<< "sample_muled" << sample_muled<<"sample_float"<<sample_float;

    cv::Mat sample_normalized = sample_float;
    //cv::subtract(sample_float, mean_, sample_normalized);
    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

std::vector<vector<float>> Detector::Detect(const cv::Mat &img)
{
    //auto t1 = std::chrono::high_resolution_clock::now();
    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    // auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    // auto t3 = std::chrono::high_resolution_clock::now();

    Preprocess(img, &input_channels);
    // auto t4 = std::chrono::high_resolution_clock::now();

    net_->Forward();
    // auto t5 = std::chrono::high_resolution_clock::now();

    /* Copy the output layer to a std::vector */
    Blob<float> *result_blob = net_->output_blobs()[0];
    const float *result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float>> detections;
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
    // auto t6 = std::chrono::high_resolution_clock::now();
    // int duration1 = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    // int duration2 = std::chrono::duration_cast<std::chrono::milliseconds>( t3 - t2 ).count();
    // int duration3 = std::chrono::duration_cast<std::chrono::milliseconds>( t4 - t3 ).count();
    // int duration4 = std::chrono::duration_cast<std::chrono::milliseconds>( t5 - t4 ).count();
    // int duration5 = std::chrono::duration_cast<std::chrono::milliseconds>( t6 - t5 ).count();

    std::cout << "Times: " << duration1 << ", " << duration2 << ", " << duration3 << ", " << duration4 << ", " << duration5 << std::endl;
    return detections;
}