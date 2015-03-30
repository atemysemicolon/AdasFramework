#include <iostream>
#include "cvccore.h"
#include "../DatasetReader/datasetReader.h"
#include "../../preProcesing/preprocessing.h"
#include "../../SuperpixelManager/csuperpixel.h"
#include "../../featureManager/featuremanager.h"
#include "../../fileModule/filemodule.hpp"
#include "../../unaryPotential/linearClassifier.h"
#include "../../Results/resultsManager.h"
#include "../../InterimResultsSaver/saveProgress.h"


#include <opencv2/opencv.hpp>
#include <memory>
//#include <boost/signals2.hpp>
#include <boost/program_options.hpp>


std::string svm_filename = "Camvid.svm";
std::string descriptors_filename = "camvid_descriptors.xml";
std::string dictionary_filename = "camvid_dictionary.xml";
std::string weights_filename="camvid_weights.xml";
int number_superpixels=1000;
//int dictionary_size = 200;

cv::Mat ann_img;

namespace po = boost::program_options;

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





void gentrainBOW()
{
    std::cout << "TRAINING DATASET BOW" << std::endl;


    //Initializing Data, Datasets
    std::shared_ptr<cvc::cData> data(new cvc::cData);
    std::shared_ptr<cvc::cDataset> dataset(new cvc::cDataset);
    //std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
    std::shared_ptr<cvc::CamVidDataset> camvid(new cvc::CamVidDataset);
    //std::shared_ptr<cvc::ClassData> kitti_dt(new cvc::KittiClassData);
    std::shared_ptr<cvc::ClassData> camvid_dt(new cvc::CamVidClassData);

    //dataset->loadDataset(kitti, cvc::DatasetTypes::TRAIN);
    dataset->loadDataset(camvid, cvc::DatasetTypes::TRAIN);


    //Initializing annotations
    std::shared_ptr<cvc::annotatonManager> ann(new cvc::annotatonManager);
    ann->init(camvid_dt, dataset);
    //ann->init(kitti_dt, dataset);
    ann->data_type=cvc::DATA_SINGLE;

    //Initialize Preprocessing operations
    std::shared_ptr<cvc::preProcessing> pre(new cvc::preProcessing);
    pre->initProcessing();
    pre->data_type=cvc::DATA_SINGLE;

    //Initializing superpixels
    std::shared_ptr<cvc::cSuperpixelManager> sup(new cvc::cSuperpixelManager);
    sup->init(number_superpixels);
    sup->data_type = cvc::DATA_SINGLE;



    //Initializing FeatureManager
    std::vector<cvc::FeatureList> feat_names;
    std::vector<int> cluster_counts;
    feat_names.push_back(cvc::FeatureList::SIFT);
    cluster_counts.push_back(50);
    feat_names.push_back(cvc::FeatureList::LBP);
    cluster_counts.push_back(50);
    feat_names.push_back(cvc::FeatureList::COLOR);
    cluster_counts.push_back(25);
    feat_names.push_back(cvc::FeatureList::LOCATION);
    cluster_counts.push_back(25);

    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    feat->initFeatures(feat_names,cluster_counts);



    //Setting up pipes
    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    pipes.push_back(ann);
    pipes.push_back(sup);
    pipes.push_back(feat);



    int i=0;

    do
    {
        std::cout<<"----------------------------------------->"<<++i<<std::endl;
    dataset->load(*data);


    for(int p=0;p<pipes.size();p++)
        pipes[p]->process(data);


    }while(dataset->next());

    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    std::cout<<"Finished one pass"<<std::endl;
    feat->setBagOfWords(true);
    std::shared_ptr<cvc::saveProgress> progressify(new cvc::saveProgress);
    pipes.push_back(progressify);
    std::cout<<"Bag of Words and Saving"<<std::endl;

    i=0;

    do{
        std::cout<<"On image : "<<++i<<std::endl;
        dataset->load(*data);
        for(int p=0;p<pipes.size();p++)
            pipes[p]->process(data);
    }while(dataset->next());
    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    std::cout<<"Done!"<<std::endl;
}

int main(int ac, char *av[])
{

//    train();
//    test();
    gentrainBOW();
    //gentestBOW();
    return 0;
}


