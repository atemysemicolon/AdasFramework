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
int dictionary_size = 200;

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




void do_not_execute()
{
    //Function where all other Commented code goes(to wrap it)
    //Remove from this if need be to use


    //static void onMouse( int event, int x, int y, int, void* )
    //{
    //    if( event != EVENT_LBUTTONDOWN )
    //        return;
    //    Point pt = Point(x,y);
    //    cout << ann_img.at<cv::Vec3b>(pt) << std::endl;
    //}



    //void trainMinimal()
    //{
    //    std::cout << "Minimally TRAINING DATASET CORE" << std::endl;

    //    //Initializing Data, Datasets
    //    std::shared_ptr<cvc::cData> data(new cvc::cData);
    //    std::shared_ptr<cvc::cDataset> dataset(new cvc::cDataset);
    //    std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
    //    std::shared_ptr<cvc::ClassData> kitti_dt(new cvc::KittiClassData);

    //    dataset->loadDataset(kitti, cvc::DatasetTypes::TRAIN);

    //    std::shared_ptr<cvc::annotatonManager> ann(new cvc::annotatonManager);
    //    ann->init(kitti_dt, dataset);
    //    ann->data_type=cvc::DATA_SINGLE;

    //    std::shared_ptr<cvc::linearClassifier> classify(new cvc::linearClassifier);
    //    classify->setDescriptorFilename(descriptors_filename);
    //    classify->readData(descriptors_filename);
    //    classify->setSvmFilename(svm_filename);
    //    classify->setasTrainingMode(true);
    //    classify->setTargetRows(1500);



    //    //Setting up pipes
    //    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    //    pipes.push_back(classify);


    //    for(int i=0;i<pipes.size();i++)
    //        pipes[i]->finalOperations(dataset);
    //}





    //void trainMinusBOW()
    //{
    //    std::cout << "TESTING DATASET CORE" << std::endl;

    //    //Initializing Data, Datasets
    //    std::shared_ptr<cvc::cData> data(new cvc::cData);
    //    std::shared_ptr<cvc::cDataset> dataset(new cvc::cDataset);
    //    std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
    //    //std::shared_ptr<cvc::CamVidDataset> camvid(new cvc::CamVidDataset);
    //    std::shared_ptr<cvc::ClassData> kitti_dt(new cvc::KittiClassData);
    //    //std::shared_ptr<cvc::ClassData> camvid_dt(new cvc::CamVidClassData);

    //    dataset->loadDataset(kitti, cvc::DatasetTypes::TRAIN);
    //    //dataset->loadDataset(camvid, cvc::DatasetTypes::TRAIN);


    //    //Initializing annotations
    //    std::shared_ptr<cvc::annotatonManager> ann(new cvc::annotatonManager);
    //    //ann->init(camvid_dt, dataset);
    //    ann->init(kitti_dt, dataset);
    //    ann->data_type=cvc::DATA_SINGLE;

    //    //Initialize Preprocessing operations
    //    std::shared_ptr<cvc::preProcessing> pre(new cvc::preProcessing);
    //    pre->initProcessing();
    //    pre->data_type=cvc::DATA_SINGLE;

    //    //Initializing superpixels
    //    std::shared_ptr<cvc::cSuperpixelManager> sup(new cvc::cSuperpixelManager);
    //    sup->init(number_superpixels);
    //    sup->data_type = cvc::DATA_SINGLE;

    //    //Initializing FeatureManager
    //    std::vector<cvc::FeatureList> feat_names;
    //    feat_names.push_back(cvc::FeatureList::SIFT);
    //    feat_names.push_back(cvc::FeatureList::COLOR);
    //    feat_names.push_back(cvc::FeatureList::LOCATION);
    //    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    //    feat->initFeatures(feat_names);

    //    //Initializing BOW
    //    std::shared_ptr<cvc::codifyFeatures> codify(new cvc::codifyFeatures);
    //    codify->init(dictionary_size);
    //    codify->setDictionaryFilename(dictionary_filename, weights_filename);




    //    //Classifier
    //    std::shared_ptr<cvc::linearClassifier> classify(new cvc::linearClassifier);
    //    classify->setSvmFilename(svm_filename);
    //    classify->setDescriptorFilename(descriptors_filename);
    //    classify->setTargetRows(1500);
    //    classify->setasTrainingMode(true);
    //    //...Setup all the in between pipes

    //    //Setting up pipes
    //    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    //    pipes.push_back(ann);
    //    pipes.push_back(pre);
    //    pipes.push_back(sup);
    //    pipes.push_back(feat);

    //    pipes.push_back(classify);



    //    char stuff;
    //    int i=0;
    //    do
    //    {
    //        std::cout<<"On image : "<<++i<<std::endl;
    //    dataset->load(*data);

    //    /*
    //     * HERE is where we do some operations
    //     *
    //     */
    //    for(int i=0;i<pipes.size();i++)
    //        pipes[i]->process(data);


    //    cv::imshow("Ann",data->annotation_orig);
    //    cv::imshow("Image", data->image);
    //    //cv::waitKey();


    //    }while(dataset->next());

    //    for(int i=0;i<pipes.size();i++)
    //        pipes[i]->finalOperations(data);





    //}


    //void testMinusBow()
    //{
    //    std::cout << "TESTING DATASET CORE" << std::endl;

    //    //Initializing Data, Datasets
    //    //Initializing Data, Datasets
    //    std::shared_ptr<cvc::cData> data(new cvc::cData);
    //    std::shared_ptr<cvc::cDataset> dataset(new cvc::cDataset);
    //    std::shared_ptr<cvc::KittiDataset> kitti(new cvc::KittiDataset);
    //    //std::shared_ptr<cvc::CamVidDataset> camvid(new cvc::CamVidDataset);
    //    std::shared_ptr<cvc::ClassData> kitti_dt(new cvc::KittiClassData);
    //    //std::shared_ptr<cvc::ClassData> camvid_dt(new cvc::CamVidClassData);

    //    dataset->loadDataset(kitti, cvc::DatasetTypes::TRAIN);
    //    //dataset->loadDataset(camvid, cvc::DatasetTypes::TRAIN);


    //    //Initializing annotations
    //    std::shared_ptr<cvc::annotatonManager> ann(new cvc::annotatonManager);
    //    //ann->init(camvid_dt, dataset);
    //    ann->init(kitti_dt, dataset);
    //    ann->data_type=cvc::DATA_SINGLE;

    //    //Initialize Preprocessing operations
    //    std::shared_ptr<cvc::preProcessing> pre(new cvc::preProcessing);
    //    pre->initProcessing();
    //    pre->data_type=cvc::DATA_SINGLE;

    //    //Initializing superpixels
    //    std::shared_ptr<cvc::cSuperpixelManager> sup(new cvc::cSuperpixelManager);
    //    sup->init(number_superpixels);
    //    sup->data_type = cvc::DATA_SINGLE;

    //    //Initializing FeatureManager
    //    std::vector<cvc::FeatureList> feat_names;
    //    feat_names.push_back(cvc::FeatureList::SIFT);
    //    feat_names.push_back(cvc::FeatureList::COLOR);
    //    feat_names.push_back(cvc::FeatureList::LOCATION);
    //    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    //    feat->initFeatures(feat_names);

    //    //Initializing BOW
    //    std::shared_ptr<cvc::codifyFeatures> codify(new cvc::codifyFeatures);
    //    codify->init(dictionary_size);
    //    codify->readDictionary(data->dictionary, dictionary_filename);

    //    std::shared_ptr<cvc::linearClassifier> classify(new cvc::linearClassifier);
    //    classify->setSvmFilename(svm_filename);
    //    classify->setasTestingMode(true);
    //    classify->load_model_cv();


    //    std::shared_ptr<cvc::resultsManager> results(new cvc::resultsManager);




    //    dataset->startAgain();
    //    dataset->changeActiveMode(cvc::DatasetTypes::TEST);
    //    std::cout<<"Round2"<<std::endl;


    //    //...Setup all the in between pipes

    //    //Setting up pipes
    //    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    //    pipes.push_back(ann);
    //    pipes.push_back(pre);
    //    pipes.push_back(sup);
    //    pipes.push_back(feat);
    //    pipes.push_back(classify);
    //    pipes.push_back(results);

    //    int i=0;

    //    cv::namedWindow("Image");
    //    cv::namedWindow("Ann");
    //    cv::namedWindow("AnnPredicted");

    //    do
    //    {
    //        std::cout<<"On image : "<<++i<<std::endl;
    //    dataset->load(*data);

    //    cv::imshow("Image", data->image);

    //    /*
    //     * HERE is where we do some operations
    //     *
    //     */
    //    for(int i=0;i<pipes.size();i++)
    //        pipes[i]->process(data);

    //    cv::imshow("Ann",data->annotation_orig);
    //    cv::imshow("AnnPredicted", data->annotation_predicted);
    //    cv::imshow("AnnPredicted-Rtrees", data->annotation_predicted2);

    //    //Debug o/p
    //    cv::waitKey(0);
    //    //;


    //    }while(dataset->next());

    //    for(int i=0;i<pipes.size();i++)
    //        pipes[i]->finalOperations(data);

    //}
}


void train()
{
    std::cout << "TRAINING DATASET CORE" << std::endl;

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
    feat_names.push_back(cvc::FeatureList::SIFT);
    feat_names.push_back(cvc::FeatureList::COLOR);
    feat_names.push_back(cvc::FeatureList::LOCATION);
    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    feat->initFeatures(feat_names);
    feat->setRegularizeOption(false);

    //Initializing BOW
    std::shared_ptr<cvc::codifyFeatures> codify(new cvc::codifyFeatures);
    codify->init(dictionary_size);
    codify->setDictionaryFilename(dictionary_filename, weights_filename);
    codify->setRegularizeOption(false);
    codify->setUseContextDescriptorsToo(false);

    //Classifier
    std::shared_ptr<cvc::linearClassifier> classify(new cvc::linearClassifier);
    classify->setSvmFilename(svm_filename);
    classify->setDescriptorFilename(descriptors_filename);
    classify->setTargetRows(5000);
    classify->setasTrainingMode(true);
    classify->setCrossValidate(true);
    //...Setup all the in between pipes



    //Setting up pipes
    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    pipes.push_back(ann);
    //pipes.push_back(pre);
    pipes.push_back(sup);
    pipes.push_back(feat);
    pipes.push_back(codify);


    int i=0;

    do
    {
        std::cout<<"On image : "<<++i<<std::endl;
    dataset->load(*data);

    /*
     * HERE is where we do some operations
     *
     */
    for(int p=0;p<pipes.size();p++)
        pipes[p]->process(data);


    cv::imshow("Ann",data->annotation_orig);
    cv::imshow("Image", data->image);
    //cv::waitKey();



    }while(dataset->next());

    //Need to do this to calculate the lengths. Perhaps to put into data
    codify->setDescriptorLengths(feat->lengths);

    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    //codify->readDictionary(data->dictionary, dictionary_filename);
    codify->readDictionary(data->dictionary, dictionary_filename);

    //codify->readWeights(data->featureWeights, data->featureBias, weights_filename);
    feat->initBagOfWords(data->dictionary);
    //feat->setWeightsAndBias(data->featureWeights,data->featureBias);


    dataset->startAgain();

    std::cout<<"Round2 - Encoding as BOW descriptor and classifying"<<std::endl;
    pipes.push_back(classify);

    i=0;
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
}

void test()
{
    std::cout << "TESTING DATASET CORE" << std::endl;

    //Initializing Data, Datasets
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
    feat_names.push_back(cvc::FeatureList::SIFT);
    feat_names.push_back(cvc::FeatureList::COLOR);
    feat_names.push_back(cvc::FeatureList::LOCATION);
    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    feat->initFeatures(feat_names);
    feat->setRegularizeOption(false);

    //Initializing BOW
    std::shared_ptr<cvc::codifyFeatures> codify(new cvc::codifyFeatures);
    codify->init(dictionary_size);
    codify->readDictionary(data->dictionary, dictionary_filename);
    //codify->readWeights(data->featureWeights, data->featureBias, weights_filename);
    codify->setRegularizeOption(false);
    codify->setUseContextDescriptorsToo(false);


    std::shared_ptr<cvc::linearClassifier> classify(new cvc::linearClassifier);
    classify->setSvmFilename(svm_filename);
    classify->setasTestingMode(true);
    classify->load_model_cv();


    std::shared_ptr<cvc::resultsManager> results(new cvc::resultsManager);

    //Setting inBOW mode
    feat->initBagOfWords(data->dictionary);

    //Setting the bias
    //feat->setWeightsAndBias(data->featureWeights, data->featureBias);

    dataset->startAgain();
    dataset->changeActiveMode(cvc::DatasetTypes::TEST);
    std::cout<<"Round2"<<std::endl;


    //...Setup all the in between pipes

    //Setting up pipes
    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    pipes.push_back(ann);
    //pipes.push_back(pre);
    pipes.push_back(sup);
    pipes.push_back(feat);
    pipes.push_back(codify);
    pipes.push_back(classify);
    pipes.push_back(results);

    int i=0;

    cv::namedWindow("Image", CV_GUI_EXPANDED);
    cv::namedWindow("Ann", CV_GUI_EXPANDED);
    cv::namedWindow("AnnPredicted", CV_GUI_EXPANDED);
    //cv::setMouseCallback("AnnPredicted", onMouse, 0);

    do
    {
        std::cout<<"On image : "<<++i<<std::endl;
    dataset->load(*data);

    cv::imshow("Image", data->image);

    /*
     * HERE is where we do some operations
     *
     */
    for(int i=0;i<pipes.size();i++)
        pipes[i]->process(data);

    ann_img = data->annotation_predicted;
    cv::imshow("Ann",data->annotation_orig);
    cv::imshow("AnnPredicted", data->annotation_predicted);
    //cv::imshow("AnnPredicted-Rtrees", data->annotation_predicted2);

    //Debug o/p
    cv::waitKey(0);
    //;


    }while(dataset->next());

    for(int i=0;i<pipes.size();i++)
        pipes[i]->finalOperations(data);
}

void gentestBOW()
{
    std::cout << "TESTING DATASET CORE" << std::endl;

    //Initializing Data, Datasets
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
    feat_names.push_back(cvc::FeatureList::SIFT);
    feat_names.push_back(cvc::FeatureList::COLOR);
    feat_names.push_back(cvc::FeatureList::LOCATION);
    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    feat->initFeatures(feat_names);
    feat->setRegularizeOption(false);

    //Initializing BOW
    std::shared_ptr<cvc::codifyFeatures> codify(new cvc::codifyFeatures);
    codify->init(dictionary_size);
    codify->readDictionary(data->dictionary, dictionary_filename);
    //codify->readWeights(data->featureWeights, data->featureBias, weights_filename);
    codify->setRegularizeOption(false);
    codify->setUseContextDescriptorsToo(true);


    std::shared_ptr<cvc::linearClassifier> classify(new cvc::linearClassifier);
    classify->setSvmFilename(svm_filename);
    classify->setasTestingMode(true);
    classify->load_model_cv();

    //Interim Results
    std::shared_ptr<cvc::saveProgress> progressify(new cvc::saveProgress);

    std::shared_ptr<cvc::resultsManager> results(new cvc::resultsManager);

    //Setting inBOW mode
    feat->initBagOfWords(data->dictionary);

    //Setting the bias
    //feat->setWeightsAndBias(data->featureWeights, data->featureBias);

    dataset->startAgain();
    dataset->changeActiveMode(cvc::DatasetTypes::TEST);
    std::cout<<"Round2"<<std::endl;


    //...Setup all the in between pipes

    //Setting up pipes
    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    pipes.push_back(ann);
    //pipes.push_back(pre);
    pipes.push_back(sup);
    pipes.push_back(feat);
    pipes.push_back(codify);
    pipes.push_back(progressify);

    int i=0;


    do
    {
        std::cout<<"On image : "<<++i<<std::endl;
    dataset->load(*data);

    /*
     * HERE is where we do some operations
     *
     */
    for(int i=0;i<pipes.size();i++)
        pipes[i]->process(data);
    ann_img = data->annotation_predicted;

    //Debug o/p
   // cv::waitKey(0);
    //;


    }while(dataset->next());

    for(int i=0;i<pipes.size();i++)
        pipes[i]->finalOperations(data);

    std::cout<<"Done!";
}

void gentrainBOW()
{
    std::cout << "TRAINING DATASET CORE" << std::endl;

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
    feat_names.push_back(cvc::FeatureList::SIFT);
    feat_names.push_back(cvc::FeatureList::COLOR);
    feat_names.push_back(cvc::FeatureList::LOCATION);
    std::shared_ptr<cvc::featureManager> feat(new cvc::featureManager);
    feat->initFeatures(feat_names);
    feat->setRegularizeOption(false);

    //Initializing BOW
    std::shared_ptr<cvc::codifyFeatures> codify(new cvc::codifyFeatures);
    codify->init(dictionary_size);
    codify->setDictionaryFilename(dictionary_filename, weights_filename);
    codify->setRegularizeOption(false);
    codify->setUseContextDescriptorsToo(true);

    //Classifier
    std::shared_ptr<cvc::linearClassifier> classify(new cvc::linearClassifier);
    classify->setSvmFilename(svm_filename);
    classify->setDescriptorFilename(descriptors_filename);
    classify->setTargetRows(5000);
    classify->setasTrainingMode(true);
    classify->setCrossValidate(true);
    //...Setup all the in between pipes


    //Interim Results
    std::shared_ptr<cvc::saveProgress> progressify(new cvc::saveProgress);

    //...Setup all the in between pipes

    //Setting up pipes
    std::vector<std::shared_ptr<cvc::cPipeModule>> pipes;
    pipes.push_back(ann);
    //pipes.push_back(pre);
    pipes.push_back(sup);
    pipes.push_back(feat);
    pipes.push_back(codify);


    int i=0;

    do
    {
        std::cout<<"On image : "<<++i<<std::endl;
    dataset->load(*data);
    for(int p=0;p<pipes.size();p++)
        pipes[p]->process(data);


    }while(dataset->next());

    codify->setDescriptorLengths(feat->lengths);

    for(int p=0;p<pipes.size();p++)
        pipes[p]->finalOperations(data);

    codify->readDictionary(data->dictionary, dictionary_filename);
    feat->initBagOfWords(data->dictionary);
    dataset->startAgain();

    std::cout<<"Round2 - Encoding as BOW descriptor and saving"<<std::endl;
    pipes.push_back(progressify);

    i=0;
    do
    {
        std::cout<<"On image : "<<++i<<std::endl;
    dataset->load(*data);


    for(int i=0;i<pipes.size();i++)
        pipes[i]->process(data);


    }while(dataset->next());

    for(int i=0;i<pipes.size();i++)
        pipes[i]->finalOperations(data);
}

int main(int ac, char *av[])
{

//    train();
//    test();
    gentrainBOW();
    gentestBOW();
    return 0;
}


