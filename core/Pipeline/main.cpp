#include <iostream>
#include "cvccore.h"
#include "../DatasetReader/datasetReader.h"
#include "../../SuperpixelManager/csuperpixel.h"
#include "../../featureManager/featuremanager.h"
#include "../../fileModule/filemodule.hpp"
#include "../../unaryClassifier/unaryclassifier.h"

#include <opencv2/opencv.hpp>
#include <memory>
#include <boost/signals2.hpp>

namespace cvc
{
class dummyPipe: public cPipeModule
{
    virtual void processData(std::shared_ptr<cData> data)
    {
        cv::Mat img = data->image;
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        data->image = img.clone();


    }


};

}



int main()
{
    std::cout << "TESTING DATASET CORE" << std::endl;

    //Initializing Data, Datasets
    std::shared_ptr<cvc::cData> data(new cvc::cData);
    std::shared_ptr<cvc::cDataset> dataset(new cvc::cDataset);
    std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
    std::shared_ptr<cvc::ClassData> kitti_dt(new cvc::KittiClassData);

    dataset->loadDataset(kitti, cvc::DatasetTypes::TRAIN);


    //Initializing annotations
    std::shared_ptr<cvc::annotatonManager> ann(new cvc::annotatonManager);
    ann->init(kitti_dt, dataset);
    ann->data_type=cvc::DATA_SINGLE;

    //Initializing superpixels
    std::shared_ptr<cvc::cSuperpixelManager> sup(new cvc::cSuperpixelManager);
    sup->init(400);
    sup->data_type = cvc::DATA_SINGLE;

    //Initializing FeatureManager
    std::vector<cvc::FeatureList> feat_names;
    feat_names.push_back(cvc::FeatureList::SIFT);
    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    feat->initFeatures(feat_names);

    //Initializing BOW
    std::shared_ptr<cvc::codifyFeatures> codify(new cvc::codifyFeatures);
    codify->init(100);

    std::shared_ptr<cvc::unaryClassifier> classify(new cvc::unaryClassifier);


    //...Setup all the in between pipes

    //Setting up pipes
    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    pipes.push_back(ann);
    pipes.push_back(sup);
    pipes.push_back(feat);
    pipes.push_back(codify);

    char stuff;
    int i=0;
    do
    {
        std::cout<<"On image : "<<++i<<std::endl;
    dataset->load(*data);
    cv::imshow("Ann",data->annotation_orig);
    cv::imshow("Image", data->image);

    /*
     * HERE is where we do some operations
     *
     */
    for(int i=0;i<pipes.size();i++)
        pipes[i]->process(data);


    //Debug o/p
    //cv::imshow("After", data->image);
    //stuff=cv::waitKey(0);
    //;


    }while(dataset->next());

    for(int i=0;i<pipes.size();i++)
        pipes[i]->finalOperations(data);

    feat->initBagOfWords(data->dictionary);


    dataset->startAgain();

    std::cout<<"Round2"<<std::endl;
    pipes.push_back(classify);


    do
    {
        std::cout<<"On image : "<<++i<<std::endl;
    dataset->load(*data);
    cv::imshow("Ann",data->annotation_orig);
    cv::imshow("Image", data->image);

    /*
     * HERE is where we do some operations
     *
     */
    for(int i=0;i<pipes.size();i++)
        pipes[i]->process(data);


    //Debug o/p
    //cv::imshow("After", data->image);
    //stuff=cv::waitKey(0);
    //;


    }while(dataset->next());

    for(int i=0;i<pipes.size();i++)
        pipes[i]->finalOperations(data);


    return 0;
}


