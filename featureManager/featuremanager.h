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
#include <tbb/tbb.h>

namespace cvc {

enum FeatureList{SIFT, SURF, COLOR, LOCATION, LBP, HOG};



class featureManager: public cPipeModule
{
public:
    std::vector<std::shared_ptr<cFeatures> > features_ptr;
    cv::Mat descriptors_concatenated;
    bool to_bow_descriptor;
    std::string folder_cache;
    

    

    void clusterFeatures(bool option)
    {
        this->to_bow_descriptor = option;
        if(option)
        {
            for(int i = 0;i<features_ptr.size();i++)
            {
                this->features_ptr[i]->computeBOW();
                std::cout<<"Writing to cache.."<<std::endl;
                std::string filenm=folder_cache +"ftno_"+ this->features_ptr[i]->ftname + "_dict.xml";
                this->features_ptr[i]->saveDictionary(filenm);
            }
        }

    }

    void clearFeatures()
    {
        for(int i=0;i<features_ptr.size();i++)
        {
            this->features_ptr[i]->descriptors_all.release();
            this->features_ptr[i]->descriptors.release();
        }
    }

    void loadDictionaries()
    {
        for(int i = 0;i<features_ptr.size();i++)
        {
            std::string filenm=folder_cache +"ftno_"+ this->features_ptr[i]->ftname + "_dict.xml";
            this->features_ptr[i]->loadDictionary(filenm);
        }

        this->to_bow_descriptor=true;

    }

    void setDictionary(int feature_number, const cv::Mat &dictionary)
    {
        this->features_ptr[feature_number]->setDictionary(dictionary);
    }



    featureManager()
    {
        this->data_type=DataTypes::DATA_SINGLE;
        this->to_bow_descriptor = false;
        this->folder_cache = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/feature_cache/";
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
            else if(features[i]==FeatureList::HOG)
                this->features_ptr.push_back(shared_ptr<cFeatures>(new HOGDescriptor));
            else 
                continue;

            this->features_ptr[i]->setClusterCount(clusterCounts[i]);
        }


    }



    int calculateDescriptor(const cv::Mat &img, const cv::Mat &segments)
    {
        std::vector<int> row_sizes;

        if(!this->to_bow_descriptor)
            for(int i=0;i<this->features_ptr.size();i++)
            {
                std::cout<<"\t Calculating feature descriptor : "<<this->features_ptr[i]->ftname<<std::endl;
                int n=this->features_ptr[i]->calculateDescriptors(img, segments);
                row_sizes.push_back(n);
                //std::cout<<"Dimensions of descriptor - "<<this->features_ptr[i]->descriptors.rows<<", "
                //<<this->features_ptr[i]->descriptors.cols<<std::endl;
            }

        else
            for(int i=0;i<this->features_ptr.size();i++)
            {
                std::cout<<"Calculating Descriptor : "<<this->features_ptr[i]->ftname<<std::endl;
                int n = this->features_ptr[i]->calculateCodedDescriptor(img, segments);
                //std::cout<<this->features_ptr[i]->descriptors<<std::endl;
                //std::cout<<"\t Size of Features(data/ptr) ->"<<i<<" : "<<sizeof(*this->features_ptr[i])<<","<<sizeof(this->features_ptr[i])<<std::endl;
                row_sizes.push_back(n);
            }
        int elem =row_sizes[0];
        for(int i = 1;i<row_sizes.size();i++)
            if(row_sizes[i]!=elem)
            {
                std::cout<<"Descriptor : "<<i<<" has the wrong dimensions.Sorry"<<std::endl;
                return 0;
            }

        return elem;
            
    }

    //Does not Save! Only puts into the respective containers
    void storeDescriptor()
    {
        for(int i=0;i<this->features_ptr.size();i++)
                this->features_ptr[i]->pushAndCollectDescriptor(true);
    }

    
    void concatDescriptors()
    {
//        this->descriptors_concatenated.release();
//        cv::Mat desc_concated;
//        for(int i=0;i<features_ptr.size();i++)
//        {
//            if(desc_concated.empty())
//                desc_concated = features_ptr[i]->descriptors.clone();
//            else
//                cv::hconcat(desc_concated,features_ptr[i]->descriptors, desc_concated);
//        }
//        this->descriptors_concatenated = desc_concated.clone(); //return descriptors_concatenated;
//        desc_concated.release();
//        std::cout<<"Dimensions of final desc are : "<<this->descriptors_concatenated.rows<<", "<<this->descriptors_concatenated.cols<<std::endl;

        //Calculating total dimensions
        int l =0;
        int n =0;
        std::vector<int> lengths;
        lengths.push_back(0);
        for(int i=0;i<this->features_ptr.size();i++)
        {
            l+=this->features_ptr[i]->dictionary.rows;
            n=this->features_ptr[i]->nr_superpixels;
            lengths.push_back(this->features_ptr[i]->dictionary.rows);
            //std::cout<<this->features_ptr[i]->descriptors;
        }

        cv::Mat concatenated_descriptor = cv::Mat::zeros(n,l, CV_32FC1);
        for(int f = 0;f<this->features_ptr.size();f++)
        {
            for(int d=0;d<this->features_ptr[f]->descriptors.cols;d++)
                this->features_ptr[f]->descriptors.col(d).copyTo(concatenated_descriptor.col(d+f*lengths[f]));
        }
        //std::cout<<concatenated_descriptor<<std::endl; //All 0s here
        this->descriptors_concatenated = concatenated_descriptor.clone();
        concatenated_descriptor.release();

        std::cout<<"Dimensions of final desc are : "<<this->descriptors_concatenated.rows<<", "<<this->descriptors_concatenated.cols<<std::endl;

    }




    void processData(std::shared_ptr<cData> data)
    {
        std::cout<<"Calculating Descriptor"<<std::endl;


        int nr_rows=calculateDescriptor(data->image, data->superpixel_segments);

        if(nr_rows==data->gt_label.size() && nr_rows>0)
        {
            if(to_bow_descriptor)
                concatDescriptors();
            else
                storeDescriptor();
            data->descriptors_concat_pooled = this->descriptors_concatenated.clone();
            //std::cout<<data->descriptors_concat_pooled<<std::endl;// All 0s here
        }


    }

    void finalOperations(std::shared_ptr<cvc::cData> data)
    {
        std::cout<<"Final operation for : "<<this->pipe_name<<std::endl;

    }



};



}

#endif // FEATUREMANAGER_H
