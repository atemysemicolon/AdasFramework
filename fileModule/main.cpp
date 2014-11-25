#include <iostream>
#include <filemodule.hpp>

using namespace std;
int main()
{
    cv::Mat img = cv::imread("/home/prassanna/Development/DataTest/Lenna.png");
    cv::Mat img_segmented = cv::imread("/home/prassanna/Development/DataTest/Lenna_sup.png",0);
    cv::Mat img_seg;
    img_segmented.convertTo(img_seg, CV_32F);
    std::shared_ptr<cvc::cData> dt(new cvc::cData);
    dt->image = img.clone();
    dt->superpixel_segments = img_seg.clone();
    dt->descriptors_concat_pooled = img.clone();
    dt->gt_label = {10, 20, 30};
    cvc::fileModule::writeData("gello.xml", *dt);
    cvc::fileModule::readData("gello.xml", *dt);


    std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
    std::shared_ptr<cvc::ClassData> kitti_data(new cvc::KittiClassData);
    cvc::cDataset ds;
    ds.loadDataset(kitti, cvc::DatasetTypes::TRAIN);
    cvc::fileModule::writeDataset("hellu.xml", ds);
    cvc::fileModule::readDataset("hellu.xml", ds);


    cout << "Hello World!" << endl;
    return 0;
}

