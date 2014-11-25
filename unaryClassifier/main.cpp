#include <iostream>
#include "unaryclassifier.h"
#include "../core/Pipeline/cvccore.h"
#include <opencv2/opencv.hpp>
int main()
{
    std::cout << "Hello World!" << std::endl;

    std::shared_ptr<cvc::cData> data(new cvc::cData);
    std::shared_ptr<cvc::unaryClassifier> uc(new cvc::unaryClassifier);
    uc->readData("descriptors_kitti.xml");
    std::vector<float> bla =uc->getGroundTruthStatistics(uc->gt_labels,12);


//    for(int i=0;i<10;i++)
//    {
//        data->gt_label.resize(100,0);
//        data->descriptors_concat_pooled=cv::Mat::zeros(100,128,CV_32F);
//        data->superpixel_neighbours=cv::Mat::zeros(100,100,CV_32F)+i;
//          uc->process(data);
//          std::cout<<"Size :"<<uc->descriptors_stored.rows<<", "<<uc->descriptors_stored.cols;
//    }
    uc->finalOperations(data);


    return 0;
}

