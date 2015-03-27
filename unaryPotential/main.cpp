#include <iostream>

#include <opencv2/opencv.hpp>
#include "linearClassifier.h"


int main()
{
#ifdef OPENMP
    std::cout<<"bla"<<std::endl;
#endif


    cvc::linearClassifier un;

    cv::Mat desc;
    std::vector<int> labels;
    cv::Mat pred_label;
    cv::Mat finalDesc;
    std::vector<int> finalLabels;

    std::string filename = "kitti_cv.svm";

    //Loading tempData
    un.readData("/home/prassanna/Development/workspace/builds/AdasFramework/PipeLine/Release/kitti_descriptors.xml");
    un.resizeSamples(un.descriptors_all, un.labels_all, 10);

    std::cout<<un.descriptors_all.rows<<", "<<un.descriptors_all.cols<<std::endl;
    un.resizeSamples(un.descriptors_all, un.labels_all, 49);
    std::cout<<un.descriptors_all.rows<<", "<<un.descriptors_all.cols<<std::endl;

    un.setSvmFilename(filename);
    un.setTargetRows(1000);

    un.trainEverything(12); //Step 1,2,3 inside everything


    cv::Mat testDesc = desc.rowRange(10,20);

    un.test_svm_cv(testDesc);// Step 4
    un.displayPredictions(un.predictedLabels);


//    for(int i=0;i<finalLabels.size();i++)
//        std::cout<<un.labels_all[i]<<"\t"<<un.descriptors_all.row(i)<<std::endl;


//    //OPENCV-TEST
//    std::cout<<"Constructing Training Data.."<<std::endl;
//    un.constructTrainData();

//    std::cout<<"Training...."<<std::endl;
//    un.train_cv();

    //LibLinear
    //un.trainlinear(finalDesc, finalLabels, filename);


    return 0;
}

