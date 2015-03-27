#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#include <opencv2/opencv.hpp>

#include "../core/Pipeline/cvccore.h"
namespace cvc
{
enum PREPROCESS{COLOR_HIST,BLUR};

class preProcessing : public cPipeModule
{
public:
    std::vector<PREPROCESS> list;
    preProcessing()
    {
            this->data_type=cvc::DataTypes::DATA_SINGLE;
            this->pipe_name="Preprocessing operations";

    }

    void initProcessing(std::vector<PREPROCESS> processing_list)//Change to vector Eventually
    {
        this->list=processing_list;
    }
    void initProcessing()
    {
        this->list.push_back(COLOR_HIST);
        this->list.push_back(BLUR);
    }
    void calcThreeChannelHistogram(cv::Mat &img)
    {
        std::vector<cv::Mat> channels_img;
        cv::split(img, channels_img);
        cv::equalizeHist(channels_img[0], channels_img[0]);
        cv::equalizeHist(channels_img[1], channels_img[1]);
        cv::equalizeHist(channels_img[2], channels_img[2]);
        cv::merge(channels_img, img);
    }



    void processData(std::shared_ptr<cData> data)
    {
        for(int i=0;i<this->list.size();i++)
        {
            if(list[i]==PREPROCESS::BLUR)
            {
                cv::Mat img=data->image.clone();
                cv::bilateralFilter(data->image, img,5,11,21);
                data->image=img.clone();
            }
            if(list[i]==PREPROCESS::COLOR_HIST)
                calcThreeChannelHistogram(data->image);

        }
    }

};




}
#endif // PREPROCESSING_H
