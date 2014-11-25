#ifndef FEATUREMANAGER_H
#define FEATUREMANAGER_H
#include <memory>
#include <map>
#include "../core/Pipeline/cvccore.h"
#include "cFeatures.h"
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
namespace cvc {

enum FeatureList{SIFT, SURF, TEXTON};



class featureManager: public cPipeModule
{
public:
    std::vector<std::shared_ptr<cFeatures> > features_ptr;
    cv::Mat descriptors_concatenated;
    bool to_bow_descriptor;
    cv::Mat dictionary;


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

    void initFeatures(std::vector<FeatureList> &features)
    {
        for(int i = 0;i<features.size();i++)
        {
            if(features[i]==FeatureList::SIFT)
                this->features_ptr.push_back(std::shared_ptr<cFeatures>(new SiftDescriptor));
        }

    }

    void initBagOfWords(cv::Mat &dictionary)
    {
        this->dictionary = dictionary.clone();
        this->to_bow_descriptor = true;
    }


    void generate_keypoint_map(cv::Mat &superpixels, std::multimap< int, cv::KeyPoint> &keypoint_map, std::multimap<int, int> &keypoint_row_map, int descriptor_number)
    {
        //Pairing keypoint to superpixel and keypoint row to superpixel
        int x = 0;


        for(int i=0;i < this->features_ptr[descriptor_number]->keypoints.size(); i++)
            {
                cv::KeyPoint kp = features_ptr[descriptor_number]->keypoints[i];
                cv::Point p = cv::Point(kp.pt);
                int segment_number = (int)superpixels.at<float>(p);
                keypoint_map.insert(std::pair<int, cv::KeyPoint>(segment_number,kp));
                keypoint_row_map.insert(std::pair<int, int>(segment_number,i));
            }
    }

    //Change Average to BOW on test
    cv::Mat calcSuperpixelDescriptor(std::multimap<int, int> &keypoint_row_map, cv::Mat &descriptors, int superpixel_number)
    {
        cv::Mat pooledDescriptor, tempDescriptor;
        std::pair<std::multimap<int, int>::iterator, std::multimap<int, int>::iterator> pr2;
        pr2 = keypoint_row_map.equal_range(superpixel_number);
        int count = 0;

        for(std::multimap<int, int>::iterator it = pr2.first; it!=pr2.second;++it)
        {
            count++;
           // std::cout<<(*it).first<<", "<<descriptors.row((*it).second)<<std::endl;
            if (tempDescriptor.empty())
                tempDescriptor = descriptors.row((*it).second).clone();
            else
                tempDescriptor.push_back(descriptors.row((*it).second));
        }

        if(!to_bow_descriptor)
            pooledDescriptor = averageDescriptor(tempDescriptor);
        else
            pooledDescriptor = normalToBowDescriptor(tempDescriptor);

        return pooledDescriptor;

    }

    cv::Mat averageDescriptor(cv::Mat &descriptors)
    {
        cv::Mat tempDescriptor;
        for(int i = 0; i<descriptors.rows; i++)
        {
            if(tempDescriptor.empty())
                tempDescriptor = descriptors.row(i).clone();
            else
                tempDescriptor+=descriptors.row(i);
        }
        tempDescriptor = tempDescriptor/descriptors.rows;
        return tempDescriptor;
    }

    cv::Mat normalToBowDescriptor(cv::Mat &descriptors)
    {
        cv::Mat codedDescriptor;
        cv::Ptr<cv::DescriptorMatcher> flannMatcher = cv::DescriptorMatcher::create("FlannBased");
        cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor(new cv::BOWImgDescriptorExtractor(flannMatcher));
        bowExtractor->setVocabulary(this->dictionary);
        bowExtractor->compute(descriptors, codedDescriptor);
        return codedDescriptor;

    }



    void calculateDescriptor(const cv::Mat &img)
    {
        for(int i=0;i<this->features_ptr.size();i++)
        {
            this->features_ptr[i]->calculateKeypoints(img);
            this->features_ptr[i]->calculateDescriptors(img);
        }

    }



    //fancy function doing stuff when superpixels are pooled.Where is original annotation? Get from superpixels
    cv::Mat poolDescriptors(cv::Mat &segmented_image, int descriptor_number)
    {
        double min;double max;
        cv::minMaxIdx(segmented_image, &min, &max);
        int number_superpixels = max+1;
        cv::Mat superpixels;
        segmented_image.convertTo(superpixels, CV_32F);

        cv::Mat superpixel_descriptors;

        //KeyPoint map generator
        std::multimap< int, cv::KeyPoint> keypoint_map;
        std::multimap<int, int> keypoint_row_map; //From superpixel to keypoint row(same as descriptor rows)


        generate_keypoint_map(superpixels, keypoint_map, keypoint_row_map, descriptor_number);

        for(int i=0;i<number_superpixels;i++)
        {
            if(superpixel_descriptors.empty())
                    superpixel_descriptors= calcSuperpixelDescriptor(keypoint_row_map, features_ptr[0]->descriptors,i).clone();
            else
                    superpixel_descriptors.push_back(calcSuperpixelDescriptor(keypoint_row_map, features_ptr[0]->descriptors,i).clone());
        }

        return superpixel_descriptors;

    }

    cv::Mat concatDescriptors()
    {
        for(int i=0;i<features_ptr.size();i++)
        {
            if(descriptors_concatenated.empty())
                descriptors_concatenated = features_ptr[i]->descriptors.clone();
            else
                cv::hconcat(descriptors_concatenated,features_ptr[i]->descriptors, descriptors_concatenated);
        }

        //std::cout<<"Dimensions of final desc are : "<<descriptors_concatenated.rows<<", "<<descriptors_concatenated.cols<<std::endl;

        return descriptors_concatenated;
    }

    void processData(std::shared_ptr<cData> data)
    {
        calculateDescriptor(data->image);
        for(int i=0;i<features_ptr.size();i++)
            features_ptr[i]->descriptors=poolDescriptors(data->superpixel_segments, i).clone(); //Loop over to get every feature
        concatDescriptors();
        data->descriptors_concat_pooled=this->descriptors_concatenated.clone();

        this->descriptors_concatenated.release();

    }



};

class codifyFeatures : public cPipeModule
{
public:
    cv::Mat dictionary;
    cv::Mat descriptors_all;
    int clusterCount;
    bool isTrained;

    codifyFeatures()
    {
        isTrained = false;
        this->clusterCount = 100;
        this->pipe_name="Bag Of Words";

    }

    void init(int clusters)
    {
        this->clusterCount=clusters;
    }

    void aggregateDescriptors(cv::Mat &descriptors)
    {
        if(descriptors_all.empty())
            descriptors_all = descriptors.clone();
        else
            cv::vconcat(descriptors_all, descriptors, descriptors_all);
    }

    void clusterEmAll()
    {
            cv::BOWKMeansTrainer bowObj(this->clusterCount);
            bowObj.add(this->descriptors_all);
            this->dictionary =bowObj.cluster().clone();
            isTrained = true;
    }

    //This function will have to go to FeatureManager - Keep this class only to train clusters
    cv::Mat normalToBowDescriptor(cv::Mat &descriptors)
    {
        cv::Mat codedDescriptor;
        cv::Ptr<cv::DescriptorMatcher> flannMatcher = cv::DescriptorMatcher::create("FlannBased");
        cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor(new cv::BOWImgDescriptorExtractor(flannMatcher));
        bowExtractor->setVocabulary(this->dictionary);
        bowExtractor->compute(descriptors, codedDescriptor);
        return codedDescriptor;
    }

    void processData(std::shared_ptr<cData> data)
    {
        if(!isTrained)
            this->aggregateDescriptors(data->descriptors_concat_pooled);
        //std::cout<<"Dimensions of descriptors : "<<this->descriptors_all.rows<<", "<<this->descriptors_all.cols<<std::endl;
    }

    void writeDictionary(cv::Mat &dictionary, std::string filename)
    {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::WRITE);
        fs<<"Dictionary"<<dictionary;
        fs.release();
    }

    void readDictionary(cv::Mat &dictionary, std::string filename)
    {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        fs["Dictionary"]>>dictionary;
        fs.release();
    }


    void finalOperations(std::shared_ptr<cData> data)
    {
        std::cout<<"Final Operations at Bag of Words:-"<<std::endl<<"Clustering..."<<std::endl;
        if(!isTrained)
        {
            this->clusterEmAll();
            data->dictionary = this->dictionary.clone();
            this->writeDictionary(this->dictionary, "Dictionary.xml");
            isTrained=true;
        }

    }

};

}
#endif // FEATUREMANAGER_H
