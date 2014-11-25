#ifndef CFEATURES_H
#define CFEATURES_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree.hpp>
#include <opencv2/features2d.hpp>

namespace cvc
{


class cFeatures
{
public:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Mat descriptors;
    std::vector<int> descriptor_map; //Map from superpixels to descriptor rows
    std::vector<cv::KeyPoint> keypoints;

    virtual void calculateKeypoints(const cv::Mat &img){}
    virtual void calculateDescriptors(const cv::Mat &img){}
    virtual void loadDescriptorsFromFile(std::string file) {}

};


class SiftDescriptor : public cFeatures
{
public:

    SiftDescriptor()
    {
        detector  = cv::Ptr<cv::FeatureDetector>(new cv::DenseFeatureDetector());
        extractor = cv::Ptr<cv::DescriptorExtractor>(new cv::SiftDescriptorExtractor);
    }
    ~SiftDescriptor()
    {
        detector.release();
        extractor.release();
    }

    void calculateKeypoints(const cv::Mat &img)
    {
        detector->detect(img, this->keypoints);
    }

    void calculateDescriptors(const cv::Mat &img)
    {
        extractor->compute(img, this->keypoints, this->descriptors);
    }

};


class TextonDescriptor: public cFeatures
{
public:
    TextonDescriptor()
    {

    }
    ~TextonDescriptor()
    {

    }

    void loadDescriptorsFromFile(std::string file)
    {

    }
};

}

#endif // CFEATURES_H
