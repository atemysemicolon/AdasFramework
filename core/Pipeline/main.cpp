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
#include <exception>

std::string svm_filename = "Camvid.svm";
std::string descriptors_filename = "camvid_descriptors.xml";
std::string dictionary_filename = "camvid_dictionary.xml";
std::string weights_filename="camvid_weights.xml";
std::string folder_dump = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/Train/";
int number_superpixels=1000;
//int dictionary_size = 200;


cv::Mat ann_img;



std::shared_ptr<cvc::cData> data(new cvc::cData);
std::shared_ptr<cvc::cDataset> dataset(new cvc::cDataset);
//std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
std::shared_ptr<cvc::CamVidDataset> camvid(new cvc::CamVidDataset);
//std::shared_ptr<cvc::ClassData> kitti_dt(new cvc::KittiClassData);
std::shared_ptr<cvc::ClassData> camvid_dt(new cvc::CamVidClassData);
std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;

std::shared_ptr<cvc::annotatonManager> ann(new cvc::annotatonManager);
std::shared_ptr<cvc::preProcessing> pre(new cvc::preProcessing);
std::shared_ptr<cvc::cSuperpixelManager> sup(new cvc::cSuperpixelManager);
std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
std::shared_ptr<cvc::saveProgress> progressify(new cvc::saveProgress);

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





/*void gentrainBOW()
{
    std::cout << "TRAINING DATASET BOW" << std::endl;


    //Initializing Data, Datasets
    std::shared_ptr<cvc::cData> data(new cvc::cData);
    std::shared_ptr<cvc::cDataset> dataset(new cvc::cDataset);
    //std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
    std::shared_ptr<cvc::CamVidDataset> camvid(new cvc::CamVidDataset);
    //std::shared_ptr<cvc::ClassData> kitti_dt(new cvc::KittiClassData);
    std::shared_ptr<cvc::ClassData> camvid_dt(new cvc::CamVidClassData);
    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;

    //dataset->loadDataset(kitti, cvc::DatasetTypes::TRAIN);
    dataset->loadDataset(camvid, cvc::DatasetTypes::TRAIN);


    //Initializing annotations

    ann->init(camvid_dt, dataset);
    //ann->init(kitti_dt, dataset);
    ann->data_type=cvc::DATA_SINGLE;

    //Initialize Preprocessing operations
    pre->initProcessing();
    pre->data_type=cvc::DATA_SINGLE;

    //Initializing superpixels
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

    feat->initFeatures(feat_names,cluster_counts);



    //Setting up pipes

    pipes.push_back(ann);
    pipes.push_back(sup);
    pipes.push_back(feat);



    int i=0;

    do
    {
        try
        {


        std::cout<<"----------------------------------------->"<<++i<<std::endl;
    dataset->load(*data);
    if(i==53)
    {
        dataset->next();
        continue;
    }

    for(int p=0;p<pipes.size();p++)
        pipes[p]->process(data);
    }
        catch(cv::Exception e)
        {
            std::cout<<"CV Exception occured :"<<e.what()<<std::endl;
        }
        catch(std::exception e)
        {
            std::cout<<"CV Exception occured :"<<e.what()<<std::endl;
        }

    }while(dataset->next());

    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    std::cout<<"Finished one pass"<<std::endl;
    feat->clusterFeatures(true);
    feat->clearFeatures();
    std::shared_ptr<cvc::saveProgress> progressify(new cvc::saveProgress);
    pipes.push_back(progressify);
    std::cout<<"Bag of Words and Saving"<<std::endl;

    i=0;
    dataset->startAgain();
    do{
        try
        {
        std::cout<<"On image : "<<++i<<std::endl;
        dataset->load(*data);
        for(int p=0;p<pipes.size();p++)
            pipes[p]->process(data);
        }
        catch(cv::Exception e)
        {
            std::cout<<"CV Exception occured :"<<e.what()<<std::endl;
        }
        catch(std::exception e)
        {
            std::cout<<"CV Exception occured :"<<e.what()<<std::endl;
        }

    }while(dataset->next());
    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    std::cout<<"Done!"<<std::endl;
}
*/
void initPipes()
{




    //Initializing annotations

    ann->init(camvid_dt, dataset);
    //ann->init(kitti_dt, dataset);
    ann->data_type=cvc::DATA_SINGLE;

    //Initialize Preprocessing operations
    pre->initProcessing();
    pre->data_type=cvc::DATA_SINGLE;

    //Initializing superpixels
    sup->init(number_superpixels);
    sup->data_type = cvc::DATA_SINGLE;



    //Initializing FeatureManager
    std::vector<cvc::FeatureList> feat_names;
    std::vector<int> cluster_counts;
    //feat_names.push_back(cvc::FeatureList::SIFT);
    //cluster_counts.push_back(50);
    feat_names.push_back(cvc::FeatureList::LBP);
    cluster_counts.push_back(50);
    feat_names.push_back(cvc::FeatureList::HOG);
    cluster_counts.push_back(50);
    feat_names.push_back(cvc::FeatureList::COLOR);
    cluster_counts.push_back(25);
    feat_names.push_back(cvc::FeatureList::LOCATION);
    cluster_counts.push_back(25);

    feat->initFeatures(feat_names,cluster_counts);

    progressify->initFolderLocation(folder_dump);

    pipes.push_back(ann);
    pipes.push_back(sup);
    pipes.push_back(feat);

    std::cout<<"Initialized...."<<std::endl;

}

void genClusters()
{
    int i=0;
    do
    {
        data = std::make_shared<cvc::cData>();
        try
        {


        std::cout<<"On Image : "<<++i<<std::endl;
    dataset->load(*data);
    if(i==53)
    {
        dataset->next();
        continue;
    } //Some problem with the 54th image.Will investigate later

    for(int p=0;p<pipes.size();p++)
        pipes[p]->process(data);
    }
        catch(cv::Exception e)
        {
            std::cout<<"CV Exception occured :"<<e.what()<<std::endl;
        }
        catch(std::exception e)
        {
            std::cout<<"CV Exception occured :"<<e.what()<<std::endl;
        }
        data.reset();

    }while(dataset->next());

    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    std::cout<<"Finished one pass"<<std::endl;
    feat->clusterFeatures(true);
    feat->clearFeatures();

}

void genFeatures()
{
    feat->loadDictionaries();
    pipes.push_back(progressify);
    std::cout<<"Bag of Words and Saving"<<std::endl;

    int i=0;
    dataset->startAgain();
    do{
        data = std::make_shared<cvc::cData>();
//        try
//        {

        std::cout<<std::endl<<"On image : "<<++i<<std::endl;
        std::cout<<"\t"<<data->filename_current<<std::endl;
        //std::cout<<"Size of Data(data/ptr) is  : "<<sizeof(*data)<<","<<sizeof(data)<<std::endl;
        //std::cout<<"Size of Dataset(data/ptr) is  : "<<sizeof(*dataset)<<","<<sizeof(dataset)<<std::endl;

        dataset->load(*data);
        for(int p=0;p<pipes.size();p++)
        {
            pipes[p]->process(data);
            //std::cout<<"Size of this pipe(data/ptr) is : "<<sizeof(*pipes[p])<<","<<sizeof(pipes[p])<<std::endl;
        }

//        }catch(cv::Exception e)
//        {
//            std::cout<<"CV Exception occured :"<<e.what()<<std::endl;
//        }catch(std::exception e)
//        {
//            std::cout<<"STD Exception occured :"<<e.what()<<std::endl;
//        }

        data.reset();

        
    }while(dataset->next());
    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    std::cout<<"Done!"<<std::endl;

}

int main(int ac, char *av[])
{


    vector<std::string> options = {"test", "train"};
    std::string s;
    bool flag = false;
    if(ac>1)
    {
        if(std::strcmp(av[1],"test")==0)
        {
            folder_dump = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/Test/";
            dataset->loadDataset(camvid, cvc::DatasetTypes::TEST);
        }
        else if(std::strcmp(av[1],"train")==0)
        {
            folder_dump = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/Train/";
             dataset->loadDataset(camvid, cvc::DatasetTypes::TRAIN);
        }
        else if(std::strcmp(av[1],"cluster")==0)
        {
            //CLuster = BOW + Cluster
            folder_dump = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/Train/";
             dataset->loadDataset(camvid, cvc::DatasetTypes::TRAIN);
             flag=true;
        }

    }

    if(!dataset)
    {
         dataset->loadDataset(camvid, cvc::DatasetTypes::TEST);
         cout<<"Loading Test dataset"<<std::endl;
    }
     initPipes();
     if(flag)
      genClusters();
     genFeatures();
    return 0;
}


