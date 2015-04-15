#include <iostream>

#include <memory>
#include <map>
#include <opencv2/features2d/features2d.hpp>
#include "featuremanager.h"

int main()
{
    cv::Mat img = cv::imread("/home/prassanna/Development/DataTest/Lenna.png");
    cv::Mat img_segmented = cv::imread("/home/prassanna/Development/DataTest/Lenna_sup.png");
    cv::cvtColor(img_segmented, img_segmented, cv::COLOR_BGR2GRAY);
    img_segmented.convertTo(img_segmented, CV_32SC1);


    std::shared_ptr<cvc::cData> dt(new cvc::cData);
    dt->image = img.clone();
    dt->superpixel_segments = img_segmented.clone();


    std::vector<cvc::FeatureList> feat_names;
    std::vector<int> cluster_counts;
    feat_names.push_back(cvc::FeatureList::SIFT);
    cluster_counts.push_back(50);

    feat_names.push_back(cvc::FeatureList::COLOR);
    cluster_counts.push_back(20);

    feat_names.push_back(cvc::FeatureList::LOCATION);
    cluster_counts.push_back(20);

    feat_names.push_back(cvc::FeatureList::LBP);
    cluster_counts.push_back(50);

    feat_names.push_back(cvc::FeatureList::HOG);
    cluster_counts.push_back(50);


    cvc::featureManager feats;
    feats.initFeatures(feat_names, cluster_counts);

    feats.processData(dt);


    //feats.calculateDescriptor(img, img_segmented);



 //    codes.clusterEmAll();
    //cv::Mat bla = codes.normalToBowDescriptor(feats.descriptors_concatenated);


    return 0;
}

