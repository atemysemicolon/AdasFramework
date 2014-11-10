#ifndef DATAMODULES_H
#define DATAMODULES_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

namespace cvc
{
//Structs to put together information
/**
 * @brief
 * Generic Structures for holding data. imageStruct is for a single image
 */
struct imageStruct
{
    bool exists;

    cv::Mat img; /**< Image File */
    std::string fileName; /**< TODO */
    std::string folderName; /**< TODO */

    //PointCloudT::Ptr cloud;

};



struct annotationStruct
{
    cv::Mat annotation;
    cv::Mat votedAnnotation;
    cv::Mat simplifiedAnnotation;

//    PointCloudT::Ptr annotation_cloud;
//    PointCloudT::Ptr annotation_simplified_cloud;
//    PointCloudT::Ptr annotation_voted_cloud;
};

/**
 * @brief
 * To hold a single type of feature descriptors and their associated data
 */
struct singleFeatureStruct
{
    std::vector<cv::KeyPoint> kps; /**< Keypoints */
    cv::Mat desc;/**< Descriptor */
    bool isBow; /**< Is it a bag of words Descriptor? */
    std::string descriptor_name; /**< eg. names = SIFT; SURF; */
    std::string pooling_method; /**< eg. Adding */

};


struct featureContainer
{
    std::vector<singleFeatureStruct> descriptors; //Multiple kind of descriptors (Different names)
    cv::Mat concatenatedDescriptor;

};



/**
 * @brief
 * To hold superpixel structures.
 */
struct singleSuperpixelStruct
{
    bool exists;
    int superpixel_label; /**< TODO */
    std::vector<cv::Point2i> positions; /**< Position with respect to image*/
    int area; /**< Number of pixels in area */
    std::vector<int> neighbours; /**< Neighbourhood labels for each superpixel */
    int gt_label; /**< Ground truth label */
    int predicted_label; /**< Predicted Label */
    featureContainer features; /**< superpixel descriptor */
};


/**
 * @brief
 *
 */
struct superpixelContainer
{
    std::vector<singleSuperpixelStruct> nodes;
    int number;
    std::string method;
};

/**
 * @brief
 *
 */
struct classdataStruct
{
    std::string dataset;
    bool exists;
    int number_of_classes; /**< Number of semantic classes */
    std::vector<int> class_map; /**< If semantic classes need to be mapped to other semantic classes */
    std::vector<cv::Vec3b> class_colours; /**< colours on annotation of each class */
    std::vector<std::string> class_names; /**< Names of each class */

};




//MOVE TO A DIFFERENT FILE LATER
enum DataTypes{DATA_SINGLE, DATA_SET};
class cData
{
public:
    DataTypes data_type;
    cData()
    {
        data_type = DATA_SINGLE;
    }

};

class cDataset : public cData
{
public:
    int number_of_files;
    cDataset()
    {
        data_type = DATA_SET;
        number_of_files = 100;
    }

};

class cParams
{
public:
    std::vector<std::string> keys;
    std::vector<std::string> values;

};

}


#endif // DATAMODULES_H
