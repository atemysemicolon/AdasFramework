#include <iostream>

#include <memory>
#include <map>
#include <opencv2/features2d.hpp>
#include "featuremanager.h"

int main()
{
    cv::Mat img = cv::imread("/home/prassanna/Development/DataTest/Lenna.png");
    cv::Mat img_segmented = cv::imread("/home/prassanna/Development/DataTest/Lenna_sup.png",0);
    std::shared_ptr<cvc::cData> dt(new cvc::cData);
    dt->image = img.clone();
    dt->superpixel_segments = img_segmented.clone();

    std::vector<cvc::FeatureList> feat_names;
    feat_names.push_back(cvc::FeatureList::SIFT);
    feat_names.push_back(cvc::FeatureList::SIFT);
    cvc::featureManager feats;
    feats.initFeatures(feat_names);
    feats.processData(dt);
    std::cout<<feats.descriptors_concatenated.rows<<std::endl;

    cvc::codifyFeatures codes;
    codes.aggregateDescriptors(feats.descriptors_concatenated);
    codes.clusterEmAll();
    cv::Mat bla = codes.normalToBowDescriptor(feats.descriptors_concatenated);


    return 0;
}

