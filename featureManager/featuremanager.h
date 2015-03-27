#ifndef FEATUREMANAGER_H
#define FEATUREMANAGER_H
#include <memory>
#include <map>

#define OPENCV3 0

#if OPENCV3
#include <opencv2/features2d.hpp>
#else
#include <opencv2/features2d/features2d.hpp>
#endif

#include "../core/Pipeline/cvccore.h"
#include "cFeatures.h"

namespace cvc {

enum FeatureList{SIFT, SURF, TEXTON, COLOR, LOCATION, LBP};



class featureManager: public cPipeModule
{
public:
    std::vector<std::shared_ptr<cFeatures> > features_ptr;
    cv::Mat descriptors_concatenated;
    bool to_bow_descriptor;
    

    

    void setBagOfWords(bool option)
    {
        this->to_bow_descriptor = option;
        if(option)
        {
            for(int i = 0;i<features_ptr.size();i++)
                this->features_ptr[i]->computeBOW();
        }

    }

    void setDictionary(int feature_number, const cv::Mat &dictionary)
    {
        this->features_ptr[feature_number]->setDictionary(dictionary);
    }



    featureManager()
    {
        this->data_type=DataTypes::DATA_SINGLE;
        this->to_bow_descriptor = false;
        this->pipe_name="Feature Manager";
    }
    ~featureManager()
    {
        this->features_ptr.clear();
    }

   

    void initFeatures(std::vector<FeatureList> &features, std::vector<int> clusterCounts)
    {
        for(int i = 0;i<features.size();i++)
        {
            if(features[i]==FeatureList::SIFT)
                this->features_ptr.push_back(std::shared_ptr<cFeatures>(new SiftDescriptor));
            else if(features[i]==FeatureList::COLOR)
                this->features_ptr.push_back(shared_ptr<cFeatures>(new ColorDescriptor));
            else if(features[i]==FeatureList::LOCATION)
                this->features_ptr.push_back(shared_ptr<cFeatures>(new LocationDescriptor));
            else if(features[i]==FeatureList::LBP)
                this->features_ptr.push_back(shared_ptr<cFeatures>(new LBPDescriptor));
            else 
                continue;

            this->features_ptr[i]->setClusterCount(clusterCounts[i]);
        }


    }



    void calculateDescriptor(const cv::Mat &img, const cv::Mat &segments)
    {

        if(!this->to_bow_descriptor)
            for(int i=0;i<this->features_ptr.size();i++)
            {
                this->features_ptr[i]->calculateDescriptors(img, segments);
                std::cout<<"Dimensions of descriptor - "<<this->features_ptr[i]->descriptors.rows<<", "
                <<this->features_ptr[i]->descriptors.cols<<std::endl;
            }

        else
            for(int i=0;i<this->features_ptr.size();i++)
                this->features_ptr[i]->calculateCodedDescriptor(img, segments);            
            
    }

    void storeDescriptor()
    {
        for(int i=0;i<this->features_ptr.size();i++)
                this->features_ptr[i]->pushAndCollectDescriptor(true);
    }

    
    cv::Mat concatDescriptors()
    {
        for(int i=0;i<features_ptr.size();i++)
        {
            if(descriptors_concatenated.empty())
                descriptors_concatenated = features_ptr[i]->descriptors_all.clone();
            else
                cv::hconcat(descriptors_concatenated,features_ptr[i]->descriptors_all, descriptors_concatenated);
        }

        //std::cout<<"Dimensions of final desc are : "<<descriptors_concatenated.rows<<", "<<descriptors_concatenated.cols<<std::endl;

        return descriptors_concatenated;
    }




    void processData(std::shared_ptr<cData> data)
    {
        std::cout<<"Calculating Descriptor"<<std::endl;


        calculateDescriptor(data->image, data->superpixel_segments);


        if(to_bow_descriptor)
            concatDescriptors();
        else
        {
            storeDescriptor();
            data->descriptors_concat_pooled = this->descriptors_concatenated.clone();
        }


    }



};



}

#endif // FEATUREMANAGER_H
