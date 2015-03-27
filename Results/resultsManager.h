#ifndef RESULTSMANAGER_H
#define RESULTSMANAGER_H

#include "../core/Pipeline/cvccore.h"
#include <opencv2/opencv.hpp>
namespace cvc
{
class resultsManager: public cPipeModule
{
public:

    resultsManager()
    {
        this->data_type=cvc::DataTypes::DATA_SINGLE;
        this->pipe_name="Calculating Results";
    }

    bool withinRange( int x, int y, cv::Mat &image ) {
        return x >= 0 && y >= 0 && x < image.cols && y < image.rows;
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
                    cv::Point pt(x,y);
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
            int x = pts[i].x;
            int y = pts[i].y;
            if(withinRange(x,y,ann_img))
                ann_img.at<uchar>(y,x) = pixel_values[i];
        }

    }

    cv::Mat generatePredictedLabelImage(cv::Mat &predicted_labels, cv::Mat &clusters)
    {
        cv::Mat predictedIndexImage = cv::Mat(clusters.size(), CV_8UC1, cv::Scalar(11)); //Void class
        for(int i=0;i<predicted_labels.rows;i++)
        {
            std::vector<cv::Point> pts = getPositions(i,clusters);
            uchar predicted_label = (uchar)predicted_labels.at<float>(i,0);
            if(predicted_label>11 || predicted_label<0)
                predicted_label=11;
            std::vector<uchar> pixel_vals(pts.size(), predicted_label);
            setPixelValues(pts, pixel_vals, predictedIndexImage);

        }
        return predictedIndexImage;
    }

    cv::Mat loadIndicestoImage(cv::Mat &index_image, std::vector<cv::Vec3b> &label_colours)
    {

//        cv::Mat colorImage;
//        colorImage = cv::Mat::zeros(index_image.size(),CV_8UC3);
//        for(int x=0;x<index_image.cols;x++)
//            for(int y=0;y<index_image.rows;y++)
//            {
//                uchar index = index_image.at<uchar>(y,x);

//                if( (index >=0) && (index <label_colours.size()) )
//                    colorImage.at<cv::Vec3b>(y,x) = label_colours[index];

//            }

//        return colorImage;

        //cv::Mat_<uchar> templatedImage = (cv::Mat_<uchar> &) index_image;
        //cv::Mat_<cv::Vec3b> colorImage(index_image.rows, index_image.cols);
        cv::Mat colorImage;
        colorImage.create(index_image.rows, index_image.cols, CV_8UC3);
        //cv::Mat tempImage = index_image.clone();
        for(int x=0;x<index_image.cols;x++)
            for(int y=0;y<index_image.rows;y++)
            {
                uchar index = index_image.at<uchar>(y,x);

                if( (index >=0) && (index <label_colours.size()) )
                    colorImage.at<cv::Vec3b>(y,x) = label_colours[index];

            }

        return colorImage;

     }


    void processData(std::shared_ptr<cData> data)
    {
        if(data->mode==TEST)
        {

            cv::Mat annPredicted=generatePredictedLabelImage(data->labelsPredicted, data->superpixel_segments);//Do Something. No need otherwise
            //cv::imwrite("/home/prassanna/Development/DataTest/sample_predicted_gt.png",predictedIndexImage);
            data->annotation_predicted=annPredicted.clone();
            //cv::Mat predictedColorImage=cv::Mat::zeros(predictedIndexImage.rows, predictedIndexImage.cols, CV_8UC3);
            data->annotation_predicted=loadIndicestoImage(annPredicted, data->class_data.class_colours).clone();
            //data->annotation_orig=loadIndicestoImage(data->annotation_indexed, data->class_data.class_colours);

            //For R-Trees prediction
            {
                cv::Mat annPredicted2=generatePredictedLabelImage(data->labelsPredicted2, data->superpixel_segments);//Do Something. No need otherwise
                //cv::imwrite("/home/prassanna/Development/DataTest/sample_predicted_gt.png",predictedIndexImage);
                data->annotation_predicted2=annPredicted2.clone();
                //cv::Mat predictedColorImage=cv::Mat::zeros(predictedIndexImage.rows, predictedIndexImage.cols, CV_8UC3);
                data->annotation_predicted2=loadIndicestoImage(annPredicted2, data->class_data.class_colours).clone();
                //data->annotation_orig=loadIndicestoImage(data->annotation_indexed, data->class_data.class_colours);
            }

        }

    }

};


}

#endif // RESULTSMANAGER_H
