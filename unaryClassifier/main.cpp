#include <iostream>
#include "unaryclassifier.h"
#include "../core/Pipeline/cvccore.h"
#include <opencv2/opencv.hpp>
int main()
{
    std::shared_ptr<cvc::cData> data(new cvc::cData);
    std::shared_ptr<cvc::unaryClassifier> uc(new cvc::unaryClassifier);
    data->mode=cvc::DatasetTypes::TEST;
    uc->setSVMFileName("kitti.svm");
    uc->setDescriptorOutFilename("descriptors_kitti.xml");
    //uc->setSVMParams(adssad);

    uc->readData("descriptors_kitti.xml");
    std::vector<float> bla =uc->getGroundTruthStatistics(uc->gt_labels,12);


    uc->loadSVMfromfilename();
    uc->process(data);
    uc->finalOperations(data);



    return 0;
}

