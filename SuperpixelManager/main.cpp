#include <iostream>
#include <opencv2/opencv.hpp>
#include "csuperpixel.h"


/**
 * @brief
 *
 * @return int
 */
int main()
{
       Mat image = imread( "/home/prassanna/Development/Datasets/KITTI_SEMANTIC2/Training_00/GT/1.png" );
       Mat ann_image_t =  imread( "/home/prassanna/Development/DataTest/sample_gt.png" );
       Mat ann_img;
       ann_image_t.convertTo(ann_img, CV_8U);

       std::shared_ptr<cvc::cData> dt(new cvc::cData);
       dt->image = image.clone();
       dt->annotation_indexed=ann_img.clone();
       dt->mode=cvc::DatasetTypes::TRAIN;

       cvc::cSuperpixelManager sup;
       sup.init(400);

       std::shared_ptr<cvc::cPipeModule> pip= std::make_shared<cvc::cSuperpixelManager>(sup);
       pip->process(dt);

       //Test
       sup.showNeigbhours(3,dt->superpixel_neighbours);
       sup.showNeigbhours(50,dt->superpixel_neighbours);
       sup.showNeigbhours(100,dt->superpixel_neighbours);
       sup.showNeigbhours(340,dt->superpixel_neighbours);

       return 0;

}

