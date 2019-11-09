#ifndef SSDOUT_H
#define SSDOUT_H
#include <iostream>
#include <string>
#include <vector>
#include <caffe/caffe.hpp>
using namespace caffe;

class SSD_Out
{
public:
    SSD_Out(vector<float> detections)
    {
        detection = detections;
    }
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    vector<float> detection;
    bool isValidDetection()
    {
        return ~(detection[0] == -1 || detection.size() != 7);
    }
    float getImage_ID()
    {
        return detection[0];
    }
    int getLable()
    {
        return static_cast<int>(detection[1]);
    }
    float getScore()
    {
        return detection[2];
    }
    vector<float> getRect()
    {
        vector<float> rect(detection[3], detection[6]);
        return rect;
    }
    std::string ToString()
    {
        std::stringstream s;
        for (int i = 0; i < detection.size(); i++)
        {
            s<<detection[i]<<", ";
        }
        return s.str();
    }
};

#endif