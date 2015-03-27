#ifndef CFEATURES_H
#define CFEATURES_H

#include <opencv2/opencv.hpp>

#define OPENCV3 0
#if OPENCV3
#include <opencv2/features2d.hpp>               //Opencv3
//#include <opencv2/xfeatures2d.hpp>              //Opencv3
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/nonfree/nonfree.hpp>        //Opencv2
#include <opencv2/features2d/features2d.hpp>  //Opencv2
#endif



#include "LBP/lbp.hpp"
#include "LBP/histogram.hpp"


namespace cvc
{


class cFeatures
{
public:
    
    cv::Mat descriptors;
    cv::Mat descriptors_all;
    int length;
    int repeat;
    cv::Mat dictionary;
    int nr_clusters;
    cv::Ptr<cv::DescriptorMatcher> flannMatcher;
    int nr_superpixels;


    

    //##############Utility Functions - Remove from inline later#####
    std::vector<Point> getPositions(const int& superpixel_index, const cv::Mat &superpixels)
    {
        std::vector<cv::Point> p;
        int temp_index = 0;
        for(int y=0;y<superpixels.rows;y++)
            for(int x=0;x<superpixels.cols;x++)
            {
                temp_index = superpixels.at<int>(y,x);
                if(temp_index==superpixel_index)
                {
                    cv::Point pt(y,x);
                    p.push_back(pt);
                }
            }
        return p;
    }
    cv::Mat meanDescriptor(const cv::Mat &desc)
    {
        cv::Mat d =desc.row(0).clone();
        for(int i=1;i<desc.rows;i++)
            d+=desc.row(i);
        d=d/desc.rows;

        return d.clone();

    }
    bool withinRange( int x, int y , cv::Mat &image)
    {
        return x >= 0 && y >= 0 && x < image.cols && y < image.rows;
    }

    void loadSuperpixelCount(const cv::Mat &segments)
    {
        double min;double max;
        cv::minMaxIdx(segments, &min, &max);
        this->nr_superpixels = max+1;
    }

    //######### END Utility Functions ##########

    
    //######ONE TIME CALL FUNCTIONS######
    void setClusterCount(int n)
    {
        this->nr_clusters = n;
    }

    void computeBOW()
    {
        cv::Mat desc;
        this->descriptors_all.convertTo(desc, CV_32F);
        std::cout<<"\t\tClustering..."<<std::endl;
        //cv::TermCriteria tc(1,1,2);
        //bowObj.add(desc);
        cv::Ptr<cv::BOWTrainer> bowObj;
        bowObj =  cv::Ptr<cv::BOWTrainer>(new cv::BOWKMeansTrainer(this->nr_clusters));
        this->dictionary =bowObj->cluster(desc).clone();
        bowObj.release();

        //Init Matcher as BOW is complete
        flannMatcher=cv::Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher);
        if(!this->descriptors_all.empty())
            this->descriptors_all.release();

    }
    void setDictionary(const cv::Mat& dict)
    {
        this->dictionary = dict.clone();

    }
    //######END ONE TIME CALL FUNCTIONS######

    cv::Mat encodeDescriptor(const cv::Mat &descriptor)
    {
        
        cv::Mat desc_uncoded;        
        descriptor.convertTo(desc_uncoded, CV_32FC1);
        cv::Mat codedDescriptor;


        if( this->dictionary.empty() )
        {
            //std::cout<<"Skipping Operations because of shitty error"<<std::endl;
            cv::Mat imgDescriptor = cv::Mat::zeros(1, desc_uncoded.rows, CV_32FC1);
            return imgDescriptor.clone();
        }

        std::vector<cv::DMatch> matches;
        this->flannMatcher->match( desc_uncoded, this->dictionary, matches );
        cv::Mat imgDescriptor;
        imgDescriptor.create(1, this->dictionary.rows, CV_32FC1);
        imgDescriptor.setTo(cv::Scalar::all(0));
        float *dptr = imgDescriptor.ptr<float>();
        for( size_t i = 0; i < matches.size(); i++ )
        {
            int queryIdx = matches[i].queryIdx;
            int trainIdx = matches[i].trainIdx; // cluster index
            CV_Assert( queryIdx == (int)i );

            dptr[trainIdx] = dptr[trainIdx] + 1.f;
        }

        // Normalize image descriptor.
        imgDescriptor /= desc_uncoded.size().height;
        //std::cout<<imgDescriptor<<std::endl; -Fixed


        //std::cout<<"Returning size : "<<imgDescriptor.rows<<", "<<imgDescriptor.cols<<std::endl;

        if(!imgDescriptor.empty())
            return imgDescriptor.clone();
        else
            return cv::Mat::zeros(1,this->dictionary.rows, CV_32F);

    }

    void pushAndCollectDescriptor(bool should_i)
    {
        if(should_i)
            if(this->descriptors_all.empty())
                this->descriptors_all = this->descriptors.clone();
            else
                this->descriptors_all.push_back(this->descriptors);

    }

    

    virtual void calculateDescriptors(const cv::Mat &img, const cv::Mat &segments){}
    virtual void calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments) {}
    virtual void loadDescriptorsFromFile(std::string file) {}

};




class SiftDescriptor : public cFeatures
{
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    std::vector<int> descriptor_map; //Map from superpixels to descriptor rows
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat denseDescriptor;
public:

    SiftDescriptor()
    {

        detector = cv::Ptr<cv::FeatureDetector>(new cv::DenseFeatureDetector);
        extractor = cv::Ptr<cv::DescriptorExtractor>(new cv::SiftDescriptorExtractor); //Opencv2
        this->length = 128;

    }
    ~SiftDescriptor()
    {
        detector.release();
        extractor.release();
    }

    void calculateKeypoints(const cv::Mat &img)
    {
        this->keypoints.clear();
        detector->detect(img, this->keypoints);
    }

    void calculateDenseDescriptor(const cv::Mat &img)
    {

        calculateKeypoints(img);
        cv::Mat localImg;
        cv::cvtColor(img,localImg, cv::COLOR_BGR2GRAY);
        extractor->compute(localImg, this->keypoints,this->denseDescriptor);
        this->length = this->descriptors.cols;

    }


    cv::Mat calculateSuperpixelDescriptor(const cv::Mat &superpixels, const int superpixel_index)
    {

        cv::Mat desc = cv::Mat::zeros(1,this->denseDescriptor.cols, this->denseDescriptor.type());
        cv::Point pt;
        int counter = 0;
        //Iterate through all keypoints to see which ones belong to sup 'n'
        for(int i=0;i<this->keypoints.size(); i++)
        {
            pt = keypoints[i].pt;

            //This operation would possibly change for bag of words
            if(superpixels.at<int>(pt) == superpixel_index)
            {
                if(desc.empty())
                    desc  = this->denseDescriptor.row(i).clone();
                else
                    desc.push_back(denseDescriptor.row(i).clone());
               
            }

        }


        return desc.clone();

    }

    

    void calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
    {
        
        cv::Mat superpixels;
        cv::Mat desc_all;

        cFeatures::loadSuperpixelCount(segments);
        calculateDenseDescriptor(img);
        segments.convertTo(superpixels, CV_32SC1);

        //Iterating through all superpixels
        for (int n = 0;n<this->nr_superpixels;n++)
        {
            cv::Mat d = calculateSuperpixelDescriptor(segments, n);            
            cv::Mat desc = meanDescriptor(d);

            if(desc_all.empty())
                desc_all=desc.clone();
            else
                desc_all.push_back(desc.clone());

        }
        //cout<<desc;

        this->descriptors = desc_all.clone();

        //Get desc per superpixel
    }


    void calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments) 
    {
        calculateDenseDescriptor(img);
        cFeatures::loadSuperpixelCount(segments);

        cv::Mat superpixels;
        segments.convertTo(superpixels, CV_32SC1);

        cv::Mat desc_all;
        //Iterating through all superpixels
        for (int n = 0;n<this->nr_superpixels;n++)
        {
             cv::Mat d = calculateSuperpixelDescriptor(segments, n);            
            cv::Mat desc = encodeDescriptor(d);

            if(desc_all.empty())
                desc_all=desc.clone();
            else
                desc_all.push_back(desc.clone());

        }
        //cout<<desc;

        this->descriptors = desc_all.clone();

    }

};


class ColorDescriptor : public cFeatures
{
public:
    ColorDescriptor()
    {
        this->length = 3;
    }
    void calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
    {
        cFeatures::loadSuperpixelCount(segments);
        cv::Mat descriptor;
        cv::Mat img_conv;
        cv::cvtColor(img, img_conv, CV_BGR2Lab);

        for(int i=0;i<this->nr_superpixels;i++)
        {
            //std::cout<<"Calculating for superpixel : "<<i<<std::endl;
            std::vector<cv::Point> pts = getPositions(i, segments);
            //std::cout<<"Got Positions"<<std::endl;
            cv::Mat desc = cv::Mat::zeros(pts.size(),3,CV_32F);
            //std::cout<<"Made desc container"<<std::endl;
            cv::Vec3i colour;

            for(int c=0;c<pts.size();c++)
            {

                //std::cout<<"Points .. "<<pts[c]<<std::endl;

                colour = img_conv.at<cv::Vec3b>(pts[c]);
                desc.at<float>(c,0) = colour[0]/pts.size();
                desc.at<float>(c,1) = colour[1]/pts.size();
                desc.at<float>(c,2) = colour[2]/pts.size();
            }

            cv::Mat desc_mean = meanDescriptor(desc);
            

            if(descriptor.empty())
                descriptor = desc_mean;
            else
                descriptor.push_back(desc_mean);

        }


        this->descriptors =  descriptor.clone();

    }

    void calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments) 
    {
        cFeatures::loadSuperpixelCount(segments);
        cv::Mat descriptor;
        cv::Mat img_conv;
        cv::cvtColor(img, img_conv, CV_BGR2Lab);

        for(int i=0;i<this->nr_superpixels;i++)
        {
            
            std::vector<cv::Point> pts = getPositions(i, segments);
            cv::Mat desc = cv::Mat::zeros(pts.size(),3,CV_32F);
            cv::Vec3i colour;
            for(int c=0;c<pts.size();c++)
            {
                colour = img_conv.at<cv::Vec3b>(pts[c]);
                desc.at<float>(c,0) = colour[0]/pts.size();
                desc.at<float>(c,1) = colour[1]/pts.size();
                desc.at<float>(c,2) = colour[2]/pts.size();
            }

            cv::Mat desc_mean = encodeDescriptor(desc);
            

            if(descriptor.empty())
                descriptor = desc_mean;
            else
                descriptor.push_back(desc_mean);

        }


        this->descriptors =  descriptor.clone();

    }

};

class LocationDescriptor : public cFeatures
{

public:
    LocationDescriptor()
    {
        this->length = 2;
    }
    void calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
    {
        cFeatures::loadSuperpixelCount(segments);
        cv::Mat descriptor;
        cv::Mat img_conv;
        cv::cvtColor(img, img_conv, CV_BGR2Lab);

        for(int i=0;i<this->nr_superpixels;i++)
        {
            
            std::vector<cv::Point> pts = getPositions(i, segments);
            cv::Mat desc = cv::Mat::zeros(pts.size(),2,CV_32F);
            cv::Vec3i colour;
            for(int c=0;c<pts.size();c++)
            {
                
                desc.at<float>(c,0) = pts[c].x/img.cols;
                desc.at<float>(c,1) = pts[c].y/img.rows;
                
            }

            cv::Mat desc_mean = meanDescriptor(desc);
         
            if(descriptor.empty())
                descriptor = desc_mean;
            else
                descriptor.push_back(desc_mean);

        }


        this->descriptors =  descriptor.clone();

    }
    void calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments) 
    {
        cFeatures::loadSuperpixelCount(segments);
        cv::Mat descriptor;
        cv::Mat img_conv;
        cv::cvtColor(img, img_conv, CV_BGR2Lab);

        for(int i=0;i<this->nr_superpixels;i++)
        {
            
            std::vector<cv::Point> pts = getPositions(i, segments);
            cv::Mat desc = cv::Mat::zeros(pts.size(),2,CV_32F);
            cv::Vec3i colour;
            for(int c=0;c<pts.size();c++)
            {
                
                desc.at<float>(c,0) = pts[c].x/img.cols;
                desc.at<float>(c,1) = pts[c].y/img.rows;
                
            }

            cv::Mat desc_mean = encodeDescriptor(desc);
         
            if(descriptor.empty())
                descriptor = desc_mean;
            else
                descriptor.push_back(desc_mean);

        }


        this->descriptors =  descriptor.clone();
    }
    
};

class LBPDescriptor : public cFeatures
{

    public:
        LBPDescriptor()
        {
            this->length = 256; //or255 last one could be zero-np
        }

        cv::Mat feature_lbp(const cv::Mat &img)
        {
            int radius = 1;
            int neighbors = 8;
            cv::Mat dst;
            cv::Mat lbp;
            cvtColor(img, dst, cv::COLOR_BGR2GRAY);
            dst.convertTo(dst, CV_32SC1); //Converting to <int>
            lbp::ELBP(dst, lbp, radius, neighbors);
            normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
            return lbp.clone();
        }

        cv::Mat feature_lbp_cluster(const cv::Mat &lbp_img, const cv::Mat &superpixels)
        {
            

            cv::Mat descriptor;

            float avg_superpixel_size  = (lbp_img.cols*lbp_img.rows)/this->nr_superpixels;


            for(int i=0;i<this->nr_superpixels;i++)
            {
                std::vector<cv::Point> pts = getPositions(i, superpixels);
                
                std::vector<uchar> histogram(100,0);
                cv::Mat desc = cv::Mat::zeros(pts.size(),this->length, CV_32FC1);

               for( int p=0; p<pts.size(); ++p )
                {
                    int pixel_value = lbp_img.at<uchar>(pts[p]);
                    desc.at<float>(p, pixel_value) += 1.0f;
                }
                cv::Mat desc_mean = meanDescriptor(desc);

                if(descriptor.empty())
                    descriptor = desc_mean.clone();
                else
                    descriptor.push_back(desc_mean);

            }

            descriptor = descriptor / (avg_superpixel_size);
            return descriptor.clone();
        }

        cv::Mat feature_lbp_grid(const cv::Mat &lbp_image)
        {
            cv::Mat descriptor;
            int dx[9] = {0, -1, -1, 0, 1, 1, 1, 0, -1 };
            int dy[9] = {0, 0, -1, -1, -1, 0, 1, 1, 1 };

            cv::Mat lbp;
            lbp_image.convertTo(lbp, CV_32F);

            for(int r = 0;r<lbp.rows; r++)
                for(int c=0;c<lbp.cols;c++)
                {
                    int p=0;
                    std::vector<float> values;
                    cv::Mat desc = cv::Mat::zeros(1,255,CV_32F);
                    //Iterate through GRID
                    for (int i =0 ;i< 9;i++)
                    {
                        if(withinRange(c+dx[i], r+dy[i], lbp) && lbp.at<float>(c+dx[i],r+dy[i])< 256)
                            values.push_back(lbp.at<float>(c+dx[i],r+dy[i]));

                    }

                    //Iterate through Vector
                    for(int i=0;i<values.size();i++)
                    {

                        int v = values[i];
                        desc.at<float>(0,v) +=1;
                    }

                    if(descriptor.empty())
                        descriptor=desc.clone();
                    else
                        descriptor.push_back(desc);

                }
            return descriptor.clone();

        }
        cv::Mat feature_lbp_codify(const cv::Mat &lbp_grid_features, const cv::Mat &superpixels)
        {

            cv::Mat descriptor;
            std::cout<<"LBP grid feature size : "<<lbp_grid_features.rows<<", "<<lbp_grid_features.cols<<std::endl;
            for(int i=0;i<this->nr_superpixels;i++)
            {
                //std::cout<<"Encoding superpixel : "<<i<<std::endl;
                std::vector<cv::Point> pos = getPositions(i, superpixels);
                //std::cout<<"Nr pixels : "<<pos.size()<<std::endl;
                cv::Mat desc;
                for(int p = 0; p<pos.size();p++)
                {
                    

                    //Encoding point to row index which will hold descriptor
                    int index = (pos[p].y*superpixels.cols + pos[p].x);
                    if(index>=lbp_grid_features.rows)
                    {
                        //std::cout<<"Out OF RANGE - Index: "<<index<<" -Point : "<<pos[p]<<std::endl;
                        continue;
                    }
                    else
                        //std::cout<<"Reading from LBP grid..."<<p<<" -> i :"<<index<<std::endl;
                    if(desc.empty())
                        desc = lbp_grid_features.row(index).clone();
                    else
                    {
                        desc.push_back(lbp_grid_features.row(index).clone());
                        //std::cout<<"Pusing onto descriptor - "<<desc.rows<<std::endl;
                    }



                }
                //std::cout<<"Accumulated descriptors...."<<std::endl;

                cv::Mat desc_coded = encodeDescriptor(desc);
                if(descriptor.empty())
                    descriptor=desc_coded.clone();
                else
                    descriptor.push_back(desc_coded.clone());

            }

            return descriptor.clone();
        }

        void calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
        {
            cFeatures::loadSuperpixelCount(segments);
            cv::Mat lbp = feature_lbp(img);
            this->descriptors = feature_lbp_cluster(lbp, segments);

        }

        void calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments) 
        {
            cFeatures::loadSuperpixelCount(segments);
            cv::Mat lbp = feature_lbp(img);
            cv::Mat lbp_grid = feature_lbp_grid(lbp);
            this->descriptors = feature_lbp_codify(lbp_grid,segments);
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
