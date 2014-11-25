#ifndef CSUPERPIXEL_H
#define CSUPERPIXEL_H
#include "slicsuperpixel.h"
#include <opencv2/opencv.hpp>
#include "../core/Pipeline/cvccore.h"
#include <iostream>
namespace cvc
{
    class cSuperpixelManager:public cPipeModule
    {
    public:
        SLICSuperpixel slic; /**< TODO */
        int number_superpixels; /**< TODO */
        int number_superpixels_after; /**< TODO */
        /**
         * @brief
         *
         */
        cSuperpixelManager()
        {
            this->data_type=DATA_SINGLE;
            this->pipe_name = "cSuperpixel";

        }
        /**
         * @brief
         *
         * @param number
         */
        void init(int number)
        {
            this->number_superpixels = number;

        }

        std::vector<cv::Point> getPositions(int superpixel_index, cv::Mat &clusters)
        {
            std::vector<cv::Point> p;
            int temp_index = 0;
            for(int y=0;y<clusters.rows;y++)
                for(int x=0;x<clusters.cols;x++)
                {
                    temp_index = clusters.at<int>(y,x);
                    if(temp_index==superpixel_index)
                    {
                        cv::Point pt(y,x);
                        p.push_back(pt);
                    }
                }
            return p;
        }

        std::vector<uchar> getPixelValues(std::vector<cv::Point> &pts, cv::Mat &ann_img)
        {
            std::vector<uchar> pixel_vals;
            for(int i=0;i<pts.size();i++)
            {
                pixel_vals.push_back(ann_img.at<uchar>(pts[i]));
            }
            return pixel_vals;
        }

        void setPixelValues(std::vector<cv::Point> &pts, std::vector<uchar> &pixel_values, cv::Mat &ann_img)
        {
            for(int i=0;i<std::min(pts.size(), pixel_values.size());i++)
            {
                ann_img.at<uchar>(pts[i]) = pixel_values[i];
            }

        }

        int findMode(std::vector<uchar>pixel_vals)
        {
            std::vector<int> histogram(this->number_superpixels_after+1,0);
            for( int i=0; i<pixel_vals.size(); ++i )
              ++histogram[ pixel_vals[i] ];
            int mode = (std::max_element( histogram.begin(), histogram.end() ) - histogram.begin());
            histogram.clear();
            return mode;

        }

        std::vector<int> modifyAnnotations(Mat &ann_img, bool isTrain, cv::Mat &clusters)
        {
            std::vector<int> ground_truth;

            if(isTrain)
            {
                for(int i=0;i<this->number_superpixels_after; i++)
                {
                    std::vector<cv::Point> pts = this->getPositions(i, clusters);

                    if(pts.size()>0)
                    {
                        std::vector<uchar> pixels = this->getPixelValues(pts, ann_img);
                        int best_label=this->findMode(pixels);
                        ground_truth.push_back(best_label);
//                        pixels = std::vector<uchar>(pixels.size(), (uchar)best_label);
//                        this->setPixelValues(pts,pixels, ann_img);
//                        pixels.clear();
                    }
                    else
                        ground_truth.push_back(-1);


                }

            }
            return ground_truth;


        }

        /**
         * @brief
         *
         * @param segments
         * @return int
         */
        int findNumberOfSuperpixels(cv::Mat &segments)
        {
            double min, max;
            cv::minMaxLoc(segments, &min, &max);
            int number_of_superpixels = max+1;
            return number_of_superpixels;
        }
        /**
         * @brief
         *
         * @param x
         * @param y
         * @param image
         * @return bool
         */
        bool withinRange( int x, int y, cv::Mat &image )
        {
            return x >= 0 && y >= 0 && x < image.cols && y < image.rows;
        }

        /**
         * @brief
         *
         * @param number
         * @param nbrs
         */
        void showNeigbhours(int number,cv::Mat &nbrs)
        {
            cv::Mat rowMat = nbrs.row(number);
            for(int i=0;i<rowMat.cols;i++)
                if(rowMat.at<int>(0,i)<0)
                    break;
            else
                    std::cout<<rowMat.at<int>(0,i)<<", ";

            std::cout<<std::endl;

        }

        /**
         * @brief
         *
         * @param contours
         * @param segments
         * @param num_superpixels
         * @return cv::Mat
         */
        cv::Mat getNeighbours(std::vector<cv::Point2i> &contours, cv::Mat &segments, int num_superpixels)
        {

           cv::Mat nbrs = cv::Mat( num_superpixels,100, CV_32SC1, Scalar(-1));

           int dx[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
           int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

           for(int i=0;i<contours.size();i++)
           {
               int actual_val = segments.at<int>(contours[i]);
               if(actual_val>=0)
                   for(int j=0;j<8;j++)
                   {
                       int x = contours[i].x+dx[j];
                       int y = contours[i].y+dy[j];
                       if(withinRange(x,y,segments))
                       {
                           int compare_val = segments.at<int>(y,x);
                           if(compare_val!=actual_val)
                           {
                               bool should_push = true;


                               int pos = 0;
                               cv::Mat rowMat = nbrs.row(actual_val).clone();
                               for(int w=0;w<rowMat.cols;w++)
                               {
                                   if(rowMat.at<int>(0,w)<0)
                                   {
                                       pos=w;
                                       break;
                                   }
                                   if(rowMat.at<int>(0,w)==compare_val)
                                   {
                                       should_push=false;
                                       break;
                                   }

                               }



                               if(should_push)
                                   nbrs.at<int>(actual_val,pos) = compare_val;




                           }
                       }
                   }
           }

           return nbrs;

        }

        /**
         * @brief
         *
         * @param data
         */
        void processData(std::shared_ptr<cData> data)
        {
            std::vector<cv::Point2i> contours;
            cv::Mat image = data->image;
            cv::Mat ann = data->annotation_indexed.clone();
            slic.init( image, number_superpixels);
            slic.generateSuperPixels();

            data->superpixel_segments = slic.clusters.clone();
            data->number_of_superpixels = findNumberOfSuperpixels(data->superpixel_segments);
            this->number_superpixels_after = data->number_of_superpixels;
            contours = slic.getContours();
            data->superpixel_neighbours =this->getNeighbours(contours, data->superpixel_segments, data->number_of_superpixels).clone();
            data->gt_label.clear();
            data->gt_label=this->modifyAnnotations(data->annotation_indexed,data->mode==DatasetTypes::TRAIN, data->superpixel_segments);

        }


    };

}
#endif // CSUPERPIXEL_H
