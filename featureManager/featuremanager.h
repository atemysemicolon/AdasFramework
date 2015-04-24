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
    bool contextCapable;
    

    

    void clusterFeatures(bool option)
    {
        this->to_bow_descriptor = option;
        if(option)
        {
            for(int i = 0;i<features_ptr.size();i++)
            {
                if(this->features_ptr[i]->clusterCapable)
                {
                    this->features_ptr[i]->computeBOW();
                    std::cout<<"Writing to cache.."<<std::endl;
                    std::string filenm=folder_cache +"ftno_"+ this->features_ptr[i]->ftname + "_dict.xml";
                    this->features_ptr[i]->saveDictionary(filenm);
                }
            }
        }

    }

    void setContextDescriptors(bool option)
    {
        this->contextCapable=option;
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
            if(this->features_ptr[i]->clusterCapable)
            {
                std::string filenm=folder_cache +"ftno_"+ this->features_ptr[i]->ftname + "_dict.xml";
                this->features_ptr[i]->loadDictionary(filenm);
            }
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
                if(this->features_ptr[i]->clusterCapable)
                {
                    std::cout<<"Calculating Clustered Descriptor : "<<this->features_ptr[i]->ftname<<std::endl;
                    int n = this->features_ptr[i]->calculateCodedDescriptor(img, segments);
                    row_sizes.push_back(n);
                }
                else
                {
                    std::cout<<"Calculating Unclustered Descriptor : "<<this->features_ptr[i]->ftname<<std::endl;
                    int n = this->features_ptr[i]->calculateDescriptors(img, segments);
                    row_sizes.push_back(n);

                }
                //std::cout<<this->features_ptr[i]->descriptors<<std::endl;
                //std::cout<<"\t Size of Features(data/ptr) ->"<<i<<" : "<<sizeof(*this->features_ptr[i])<<","<<sizeof(this->features_ptr[i])<<std::endl;
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
            if(this->features_ptr[i]->clusterCapable)
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

            n=this->features_ptr[i]->nr_superpixels;

            lengths.push_back(this->features_ptr[i]->descriptors.cols);
            l+=this->features_ptr[i]->descriptors.cols;
            //std::cout<<this->features_ptr[i]->descriptors;
        }
        std::cout<<"Concatenating.."<<std::endl;
        cv::Mat concatenated_descriptor = cv::Mat::zeros(n,l, CV_32FC1);
        for(int f = 0;f<this->features_ptr.size();f++)
        {
            for(int d=0;d<this->features_ptr[f]->descriptors.cols;d++)
                this->features_ptr[f]->descriptors.col(d).copyTo(concatenated_descriptor.col(d+f*lengths[f]));
        }
        //std::cout<<concatenated_descriptor<<std::endl; //All 0s here
        this->descriptors_concatenated = concatenated_descriptor.clone();
        concatenated_descriptor.release();



    }

    void addContextualDescriptor(const cv::Mat &nbrs, const cv::Mat &segments)
    {
        std::cout<<"Adding contextual Descriptors"<<std::endl;
        cv::Mat descriptor_right=cv::Mat::zeros(this->descriptors_concatenated.rows,
                                                this->descriptors_concatenated.cols,
                                                CV_32FC1); //The descriptor to be concatenated to each descriptor

        cv::Mat descriptor_left = this->descriptors_concatenated;
        std::vector<int> sup_sizes;
        const int initVal=-1;
        sup_sizes.resize(this->descriptors_concatenated.rows, initVal);

        //Iterate through all superpixels
        for(int i=0;i<this->features_ptr[0]->nr_superpixels;i++)
        {
            cv::Mat nbrs_i = nbrs.row(i);
            int j=0;
            int nbrhood_size=0;
            cv::Mat desc = cv::Mat::zeros(1,descriptor_left.cols, CV_32FC1);
            //std::cout<<"\tSuperpixel number : "<<i<<":-"<<std::endl;
            //Iterate through each neighbour
            while(nbrs_i.at<int>(0,j)>=0)
            {

                cv::Mat sup_descriptor;
                int nbr_sup = nbrs_i.at<int>(0,j);
                //std::cout<<"\t\tNbr:"<<nbr_sup<<std::endl;
                j++;
                sup_descriptor=descriptor_left.row(nbr_sup);

                if(sup_sizes[nbr_sup]<0)
                    sup_sizes[nbr_sup] = this->features_ptr[0]->getPositions(nbr_sup,segments).size();

                sup_descriptor*=sup_sizes[nbr_sup];
                nbrhood_size +=sup_sizes[nbr_sup];

                desc+=sup_descriptor;


            }
            if(nbrhood_size>0)
                desc/=(2*nbrhood_size);
            else
                desc/=2;
            //std::cout<<"Generated Descriptor"<<desc<<std::endl;
            desc.copyTo(descriptor_right.row(i));

        }

        cv::hconcat(this->descriptors_concatenated, descriptor_right, this->descriptors_concatenated);


    }




    void processData(std::shared_ptr<cData> data)
    {
        std::cout<<"Calculating Descriptor"<<std::endl;


        int nr_rows=calculateDescriptor(data->image, data->superpixel_segments);

        if(nr_rows==data->gt_label.size() && nr_rows>0)
        {
            if(to_bow_descriptor)
            {
                concatDescriptors();
                if(this->contextCapable)
                    addContextualDescriptor(data->superpixel_neighbours, data->superpixel_segments);
            }
            else
                storeDescriptor();
            data->descriptors_concat_pooled = this->descriptors_concatenated.clone();
            std::cout<<"Dimensions of final desc are : "<<this->descriptors_concatenated.rows<<", "<<this->descriptors_concatenated.cols<<std::endl;
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
