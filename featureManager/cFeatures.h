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
    
    cv::Mat descriptors; //descriptors of one image
    cv::Mat descriptors_all; //Across all images in dataset-Not concatenated features!
    int length;
    int repeat;
    cv::Mat dictionary;
    int nr_clusters;
    cv::Ptr<cv::DescriptorMatcher> flannMatcher;
    int nr_superpixels;
    std::string ftname;
    bool clusterCapable;

    

    //##############Utility Functions - Remove from inline later#####

    //Move later to FileModule class
    bool saveDescriptors(const std::string file_name)
    {
        //std::string new_file_name = file_name+"_"+std::to_string(feature_number);
        std::string fieldnm= "Descriptor";
        return saveMat(file_name, fieldnm, this->descriptors);

    }
    bool saveDictionary(const std::string file_name)
    {
        //std::string new_file_name = file_name+"_"+std::to_string(feature_number);
        std::string fieldnm= "Dictionary";
        return saveMat(file_name, fieldnm, this->dictionary);

    }
    bool loadDescriptors(const std::string file_name)
    {
        //std::string new_file_name = file_name+"_"+std::to_string(feature_number);
        std::string fieldnm= "Descriptor";
        this->descriptors = loadMat(file_name, fieldnm);
        return !this->descriptors.empty();
    }
    bool loadDictionary(const std::string file_name)
    {

        //std::string new_file_name = file_name+"_"+std::to_string(feature_number);
        std::string fieldnm= "Dictionary";
        this->dictionary = loadMat(file_name, fieldnm);
        return !this->dictionary.empty();
    }
    bool saveMat(const std::string &file_name, const std::string &field_name, const cv::Mat &mat_obj)
    {
        cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
        if(!fs.isOpened())
            return false;
        fs<<field_name<<mat_obj;
        fs.release();
        return true;
    }
    cv::Mat loadMat(const std::string &file_name, const std::string &field_name)
    {
        cv::FileStorage fs(file_name,cv::FileStorage::READ);
        cv::Mat m;
        if(!fs.isOpened())
            return m.clone();
        fs[field_name]>>m;
        fs.release();
        return m.clone();
    }

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
                    cv::Point pt(x,y);
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
        if(this->flannMatcher.empty())
            this->flannMatcher=cv::Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher);

        cv::Mat desc_uncoded;//=descriptor;
        descriptor.convertTo(desc_uncoded, CV_32FC1);
        //cv::Mat codedDescriptor;


        if( this->dictionary.empty() )
        {
            //std::cout<<"Skipping Operations because of shitty error"<<std::endl;
            cv::Mat imgDescriptor = cv::Mat::zeros(1, desc_uncoded.rows, CV_32FC1);
            return imgDescriptor.clone();
        }

        std::vector<cv::DMatch> matches;
        this->flannMatcher->match( desc_uncoded, this->dictionary, matches );


        cv::Mat imgDescriptor(1, this->dictionary.rows, CV_32FC1);
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

    

    virtual int calculateDescriptors(const cv::Mat &img, const cv::Mat &segments){}
    virtual int calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments) {}
    virtual int loadDescriptorsFromFile(std::string file) {}

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
        this->ftname = "SIFT";
        this->clusterCapable = true;

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

        cv::Mat desc;// = cv::Mat::zeros(1,this->denseDescriptor.cols, this->denseDescriptor.type());
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
                    desc  = this->denseDescriptor.row(i);
                else
                    desc.push_back(denseDescriptor.row(i));
               
            }

        }


        return desc.clone();

    }

    

    int calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
    {
        
        cv::Mat superpixels=segments;
        cv::Mat desc_all;

        cFeatures::loadSuperpixelCount(segments);
        calculateDenseDescriptor(img);
        //segments.convertTo(superpixels, CV_32SC1);

        //Iterating through all superpixels
        for (int n = 0;n<this->nr_superpixels;n++)
        {

            cv::Mat d = calculateSuperpixelDescriptor(segments, n);            
            if(d.empty())
                d=cv::Mat::zeros(2, denseDescriptor.cols, CV_32FC1);
            cv::Mat desc = meanDescriptor(d);

            if(desc_all.empty())
                desc_all=desc;
            else
                desc_all.push_back(desc);

            desc.release();
            d.release();

        }
        //cout<<desc;

        this->descriptors = desc_all.clone();
        desc_all.release();
        superpixels.release();
        return this->descriptors.rows;

        //Get desc per superpixel
    }


    int calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments)
    {
        //std::cout<<"\tEncoding Sift..."<<std::endl;
        calculateDenseDescriptor(img);
        cFeatures::loadSuperpixelCount(segments);

        cv::Mat superpixels;
        segments.convertTo(superpixels, CV_32SC1);

        cv::Mat desc_all=cv::Mat::zeros(this->nr_superpixels, this->dictionary.rows, CV_32FC1);

        //Iterating through all superpixels
        for (int n = 0;n<this->nr_superpixels;n++)
        {
             cv::Mat d = calculateSuperpixelDescriptor(segments, n);            
             cv::Mat m = encodeDescriptor(d);
             m.copyTo(desc_all.row(n));
             //std::cout<<m<<std::endl;

            d.release();
        }

        //std::cout<<desc_all<<std::endl;

        this->descriptors = desc_all.clone();
        desc_all.release();
        superpixels.release();
        //std::cout<<"\tDone!"<<std::endl;
        return this->descriptors.rows;

    }

};


class ColorDescriptor : public cFeatures
{
public:
    ColorDescriptor()
    {
        this->length = 3;
        this->ftname = "Color";
        this->clusterCapable = true;
    }
    int calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
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
        descriptor.release();
        img_conv.release();
        return this->descriptors.rows;

    }

    int calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments)
    {
        cFeatures::loadSuperpixelCount(segments);
        cv::Mat descriptor=cv::Mat::zeros(this->nr_superpixels, this->dictionary.rows, CV_32FC1);
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


            cv::Mat m = encodeDescriptor(desc);

            m.copyTo(descriptor.row(i));

        }


        this->descriptors =  descriptor.clone();
        descriptor.release();
        img_conv.release();
        //std::cout<<this->descriptors<<std::endl;
        return this->descriptors.rows;

    }

};

class LocationDescriptor : public cFeatures
{

public:
    LocationDescriptor()
    {
        this->length = 2;
        this->ftname = "Location";
        this->clusterCapable = true;
    }
    int  calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
    {
        cFeatures::loadSuperpixelCount(segments);
        cv::Mat descriptor;
        cv::Mat img_conv;
        cv::cvtColor(img, img_conv, CV_BGR2Lab);

        for(int i=0;i<this->nr_superpixels;i++)
        {
            
            std::vector<cv::Point> pts = getPositions(i, segments);
            cv::Mat desc = cv::Mat::zeros(pts.size(),2,CV_32F);
            for(int c=0;c<pts.size();c++)
            {
                
                desc.at<float>(c,0) = (float)pts[c].x/img.cols;
                desc.at<float>(c,1) = (float)pts[c].y/img.rows;
                
            }

            cv::Mat desc_mean = meanDescriptor(desc);
         
            if(descriptor.empty())
                descriptor = desc_mean;
            else
                descriptor.push_back(desc_mean);

        }


        this->descriptors =  descriptor.clone();
        descriptor.release();
        img_conv.release();
        return this->descriptors.rows;


    }
    int calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments)
    {
        std::cout<<"\tEncoding Colour..."<<std::endl;
        cFeatures::loadSuperpixelCount(segments);
        cv::Mat descriptor=cv::Mat::zeros(this->nr_superpixels, this->dictionary.rows, CV_32FC1);
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


         
            cv::Mat m= encodeDescriptor(desc);
            m.copyTo(descriptor.row(i));
            //std::cout<<m<<std::endl;



        }


        this->descriptors =  descriptor.clone();
        descriptor.release();
        img_conv.release();
        std::cout<<"\tDone!"<<std::endl;
        return this->descriptors.rows;
    }
    
};

class LBPDescriptor : public cFeatures
{

    public:
        LBPDescriptor()
        {
            this->length = 256; //or255 last one could be zero-np
            this->ftname = "LBP";
            this->clusterCapable = false;
        }
        bool withinRange(const cv::Mat &img, const cv::Point &p)
        {
            if(p.x<img.cols && p.x>= 0 && p.y< img.rows && p.y >=0)
                return true;
            return false;
        }


        cv::Mat feature_lbp(const cv::Mat &img)
        {
            cv::Mat lbp_im;
            cv::Mat dst;
            cv::cvtColor(img,dst, CV_BGR2GRAY);
            int radius=1;
            lbp::ELBP(dst,lbp_im, radius);
            normalize(lbp_im, lbp_im, 0, 255, NORM_MINMAX, CV_8UC1);
            return lbp_im.clone();
        }
        cv::Mat genDescriptorPtLbp(const cv::Point &p, const cv::Mat &lbp_image)
        {
            //Lbp Image is smaller than image
            const int dx[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
            const int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

            cv::Mat desc = cv::Mat::zeros(1,256, CV_32FC1);

            //Iterate through 8-nbrhood and create descriptor
            for(int i=0;i<8;i++)
            {
                cv::Point compare_pt = p-cv::Point(dx[i],dy[i]);
                if(withinRange(lbp_image, compare_pt))
                {
                    int lbp_label = lbp_image.at<uchar>(compare_pt);
                    desc.at<float>(0,lbp_label)+=1.0f;
                }
            }

            return desc.clone();
        }
        cv::Mat genSuperpixelDescriptorLbp(std::vector<cv::Point> &pts, const cv::Mat &segments, const cv::Mat &lbp_image)
        {
            //Need to simple transformation as points on image!=pts on lbp image
            std::vector<cv::Point> lbp_pts;
            //Holds difference
            cv::Point p;
            p.x=1;
            p.y=1;
            for(auto pt : pts)
            {
                cv::Point lbp_pt = pt-p;
                if(withinRange(lbp_image, lbp_pt))
                    lbp_pts.push_back(lbp_pt);

            }
            cv::Mat superpixel_desc;
            for (auto pt : lbp_pts)
                superpixel_desc.push_back(genDescriptorPtLbp(pt, lbp_image));
            return superpixel_desc.clone();
        }

        //Trying to bin LBP directly without BOW
        cv::Mat superpixelLbp(std::vector<cv::Point> &pts, const cv::Mat &lbp_image)
        {
            std::vector<cv::Point> lbp_pts;
            //Holds difference
            cv::Point p;
            p.x=1;
            p.y=1;
            for(auto pt : pts)
            {
                cv::Point lbp_pt = pt-p;
                if(withinRange(lbp_image, lbp_pt))
                    lbp_pts.push_back(lbp_pt);

            }

            cv::Mat superpixel_desc = cv::Mat::zeros(1, 256, CV_32FC1);
            int c=0;
            for(auto pt:lbp_pts)
            {
                    if(withinRange(lbp_image, pt))
                    {
                        int lbp_label = lbp_image.at<uchar>(pt);
                        superpixel_desc.at<float>(0,lbp_label)+=1.0f;
                        c++;
                    }
            }
            if(c!=0)
                superpixel_desc = superpixel_desc/c;

            return superpixel_desc.clone();

        }

        int calculateDescriptorsForBOW(const cv::Mat &img, const cv::Mat &segments)
        {
            cFeatures::loadSuperpixelCount(segments);

            //LBP Image
            cv::Mat lbp_im = feature_lbp(img);


            cv::Mat desc = cv::Mat::zeros(this->nr_superpixels, 256, CV_32FC1);
            //LBP Descriptor
            for(int i =0;i<this->nr_superpixels;i++)
            {
                std::vector<cv::Point> pts;
                pts = getPositions(i, segments);
                cv::Mat d = genSuperpixelDescriptorLbp(pts,segments, lbp_im);
                if(d.empty())
                    d=cv::Mat::zeros(1, 256, CV_32FC1);
                cv::Mat superpixel_desc = this->meanDescriptor(d);
                superpixel_desc.copyTo(desc.row(i));
            }

            this->descriptors = desc.clone();
            desc.release();
            lbp_im.release();
            return this->descriptors.rows;

        }
        int calculateDescriptorsForNoBOW(const cv::Mat &img, const cv::Mat &segments)
        {
            cFeatures::loadSuperpixelCount(segments);

            //LBP Image
            cv::Mat lbp_im = feature_lbp(img);


            cv::Mat desc = cv::Mat::zeros(this->nr_superpixels, 256, CV_32FC1);
            //LBP Descriptor
            for(int i =0;i<this->nr_superpixels;i++)
            {
                std::vector<cv::Point> pts;
                pts = getPositions(i, segments);
                cv::Mat d = superpixelLbp(pts,lbp_im);
                if(d.empty())
                    d=cv::Mat::zeros(1, 256, CV_32FC1);
                d.copyTo(desc.row(i));
            }

            this->descriptors = desc.clone();
            desc.release();
            lbp_im.release();
            return this->descriptors.rows;
        }


        int calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
        {
           int n = calculateDescriptorsForNoBOW(img, segments);
           return n;
        }

        int calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments)
        {
            cFeatures::loadSuperpixelCount(segments);

            //LBP Image
            cv::Mat lbp_im = feature_lbp(img);


            cv::Mat desc = cv::Mat::zeros(this->nr_superpixels, this->dictionary.rows, CV_32FC1);
            //LBP Descriptor
            for(int i =0;i<this->nr_superpixels;i++)
            {
                std::vector<cv::Point> pts = getPositions(i, segments);
                cv::Mat d = genSuperpixelDescriptorLbp(pts,segments, lbp_im);
                if(d.empty())
                    d=cv::Mat::zeros(1, 256, CV_32FC1);
                cv::Mat superpixel_desc = this->encodeDescriptor(d);
                //std::cout<<superpixel_desc<<std::endl; //Debugging
                superpixel_desc.copyTo(desc.row(i));
            }

            this->descriptors = desc.clone();
            desc.release();
            lbp_im.release();
            return this->descriptors.rows;
        }

    

};

class HOGDescriptor: public cFeatures
{

public:
    cv::Mat desc_hog;
    HOGDescriptor()
    {
        this->length=9;//?
        this->ftname = "HOG";
        this->clusterCapable = true;

    }
    ~HOGDescriptor()
    {

    }
    //Utility
    void getPositionsAndCells(int superpixel_index, const cv::Mat &segments, std::vector<cv::Point> &p, std::vector<int> &p_x_map, std::vector<int> &p_y_map, const int cell_width_x, const int cell_width_y)
    {
        int temp_index = 0;
        for(int y=0;y<segments.rows;y++)
            for(int x=0;x<segments.cols;x++)
            {
                temp_index = segments.at<int>(y,x);
                if(temp_index==superpixel_index)
                {
                    cv::Point pt(x,y);
                    p.push_back(pt);
                    p_x_map.push_back((int)x/cell_width_x);
                    p_y_map.push_back((int)y/cell_width_y);
                }
            }
        //return p;
    }

    cv::Mat genDescriptorSingleHog(const cv::Mat &segments, const cv::Mat &descriptors_hog, int n_superpixel, int cell_width_x, int cell_width_y)
    {
       std::vector<cv::Point> pts;
       std::vector<int> cell_map_x;
       std::vector<int> cell_map_y;
       getPositionsAndCells(n_superpixel,segments, pts,cell_map_x,cell_map_y,cell_width_x,cell_width_y);
       cv::Mat desc=cv::Mat::zeros(pts.size(), 9, CV_32FC1);
       for(int i=0;i<pts.size();i++)
       {
           //std::cout<<pts[i]<<"->"<<cell_map_x[i]<<","<<cell_map_y[i]<<std::endl;
           cv::Mat tempDesc = descriptors_hog.row(cell_map_y[i]*cell_width_x + cell_map_x[i]);
           //std::cout<<tempDesc<<endl;
           //cin.get();
           if(!tempDesc.empty())
            tempDesc.copyTo(desc.row(i));
       }

       return desc.clone();
    }


    //Unencoded descriptor
    void convertToHOGDescriptor(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size, cv::Mat &desc, int cellSize )
    {
        const int DIMX = size.width;
        const int DIMY = size.height;
        float zoomFac = 1.5;
        //Mat visu;
        //resize(color_origImg, visu, Size( (int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac) ) );

        //int cellSize        = 8;
        int gradientBinSize = 9;
        //float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

        // prepare data structure: 9 orientation / gradient strenghts for each cell
        int cells_in_x_dir = DIMX / cellSize;
        int cells_in_y_dir = DIMY / cellSize;
        float*** gradientStrengths = new float**[cells_in_y_dir];
        int** cellUpdateCounter   = new int*[cells_in_y_dir];
        for (int y=0; y<cells_in_y_dir; y++)
        {
            gradientStrengths[y] = new float*[cells_in_x_dir];
            cellUpdateCounter[y] = new int[cells_in_x_dir];
            for (int x=0; x<cells_in_x_dir; x++)
            {
                gradientStrengths[y][x] = new float[gradientBinSize];
                cellUpdateCounter[y][x] = 0;

                for (int bin=0; bin<gradientBinSize; bin++)
                    gradientStrengths[y][x][bin] = 0.0;
            }
        }

        // nr of blocks = nr of cells - 1
        // since there is a new block on each cell (overlapping blocks!) but the last one
        int blocks_in_x_dir = cells_in_x_dir - 1;
        int blocks_in_y_dir = cells_in_y_dir - 1;

        // compute gradient strengths per cell
        int descriptorDataIdx = 0;
        int cellx = 0;
        int celly = 0;

        for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
        {
            for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
            {
                // 4 cells per block ...
                for (int cellNr=0; cellNr<4; cellNr++)
                {
                    // compute corresponding cell nr
                    cellx = blockx;
                    celly = blocky;
                    if (cellNr==1) celly++;
                    if (cellNr==2) cellx++;
                    if (cellNr==3)
                    {
                        cellx++;
                        celly++;
                    }

                    for (int bin=0; bin<gradientBinSize; bin++)
                    {
                        float gradientStrength = descriptorValues[ descriptorDataIdx ];
                        descriptorDataIdx++;

                        gradientStrengths[celly][cellx][bin] += gradientStrength;

                    } // for (all bins)


                    // note: overlapping blocks lead to multiple updates of this sum!
                    // we therefore keep track how often a cell was updated,
                    // to compute average gradient strengths
                    cellUpdateCounter[celly][cellx]++;

                } // for (all cells)


            } // for (all block x pos)
        } // for (all block y pos)

        desc = cv::Mat::zeros(cells_in_y_dir*cells_in_x_dir, 9, CV_32FC1);

        // compute average gradient strengths
        for (celly=0; celly<cells_in_y_dir; celly++)
        {
            for (cellx=0; cellx<cells_in_x_dir; cellx++)
            {

                float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

                // compute average gradient strenghts for each gradient bin direction
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
                    desc.at<float>(celly*cells_in_x_dir + cellx,bin) = gradientStrengths[celly][cellx][bin];
                }
            }
        }

        //cout<<"Stuff:"<<desc<<std::endl;





        // don't forget to free memory allocated by helper data structures!
        for (int y=0; y<cells_in_y_dir; y++)
        {
            for (int x=0; x<cells_in_x_dir; x++)
            {
                delete[] gradientStrengths[y][x];
            }
            delete[] gradientStrengths[y];
            delete[] cellUpdateCounter[y];
        }
        delete[] gradientStrengths;
        delete[] cellUpdateCounter;

        //return desc.clone();

    }

    void calculateGradients(const cv::Mat &img, const cv::Mat &segments)
    {
        cv::Mat img_gray;
        cv::cvtColor(img, img_gray, CV_RGB2GRAY);
        cv::HOGDescriptor hog;
        std::vector<float> descriptorsValues;
        std::vector<Point> locations;

        hog.compute( img_gray, descriptorsValues, Size(0,0), Size(0,0), locations);
        hog.cellSize=Size(2,2);

        //std::cout << "HOG descriptor size is " << hog.getDescriptorSize() << std::endl;
        //std::cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << std::endl;
        //std::cout << "Found " << descriptorsValues.size() << " descriptor values" << std::endl;
        //std::cout << "Nr of locations specified : " << locations.size() << std::endl;


        convertToHOGDescriptor(img,descriptorsValues, cv::Size(img.cols, img.rows), desc_hog, hog.cellSize.height);

        //genDescriptorSingleHog(img, segments, desc, 2, 8,8);
    }
    int calculateDescriptors(const cv::Mat &img, const cv::Mat &segments)
    {
        //Calculates HOG image and reshapes as descriptors
        cFeatures::loadSuperpixelCount(segments);
        calculateGradients(img, segments);
        //int nr_superpixels = 1000;//Change!!
        cv::Mat desc_all = cv::Mat::zeros(this->nr_superpixels, 9,CV_32FC1);
        for(int i=0;i<this->nr_superpixels;i++)
        {
            cv::Mat d = genDescriptorSingleHog(segments, desc_hog, i, 8,8);
            cv::Mat d_sum = this->meanDescriptor(d);
            if(!d_sum.empty())
                d_sum.copyTo(desc_all.row(i));
        }

        this->descriptors = desc_all.clone();
        desc_all.release();
        return this->descriptors.rows;
    }

    int calculateCodedDescriptor(const cv::Mat &img, const cv::Mat &segments)
    {
        //Calculates HOG image and reshapes as descriptors
        cFeatures::loadSuperpixelCount(segments);
        calculateGradients(img, segments);
        //int nr_superpixels = 1000;//Change!!
        cv::Mat desc_all = cv::Mat::zeros(this->nr_superpixels, this->dictionary.rows,CV_32FC1);
        for(int i=0;i<this->nr_superpixels;i++)
        {
            cv::Mat d = genDescriptorSingleHog(segments, desc_hog, i, 8,8);
            cv::Mat d_sum = this->encodeDescriptor(d);
            if(!d_sum.empty())
                d_sum.copyTo(desc_all.row(i));
        }

        this->descriptors = desc_all.clone();
        desc_all.release();
        return this->descriptors.rows;

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

    int loadDescriptorsFromFile(std::string file)
    {
        return 0;

    }
};


}

#endif // CFEATURES_H
