#include <iostream>
#include <opencv2/opencv.hpp>

#include "preprocessing.h"



int main()
{
    cv::Mat img = cv::imread("/home/prassanna/Development/Datasets/KITTI_SEMANTIC2/Training_00/RGB/3.png");;
    cv::imshow("Original",img);
    std::shared_ptr<cvc::cData> dt(new cvc::cData);
    dt->image = img.clone();

    cvc::preProcessing preprocess;
    preprocess.initProcessing();
    preprocess.process(dt);
    cv::imshow("Histogram-ed", dt->image);

    cv::waitKey(0);

    return 0;
}


