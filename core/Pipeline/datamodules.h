#ifndef DATAMODULES_H
#define DATAMODULES_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../DatasetReader/datasetReader.h"
#include "../AnnotationManager/annotationManager.h"

namespace cvc
{


/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
struct classdataStruct
{
    std::string dataset; /**< TODO */
    bool exists; /**< TODO */
    int number_of_classes; /**< Number of semantic classes */ /**< TODO */
    std::vector<int> class_map; /**< If semantic classes need to be mapped to other semantic classes */ /**< TODO */
    std::vector<cv::Vec3b> class_colours; /**< colours on annotation of each class */ /**< TODO */
    std::vector<std::string> class_names; /**< Names of each class */ /**< TODO */

};




//MOVE TO A DIFFERENT FILE LATER
/**
 * @brief
 *
 */
enum DataTypes{DATA_SINGLE, DATA_SET};
/**
 * @brief
 *
 */
enum DatasetTypes{TRAIN, TEST, ALL};

/**
 * @brief
 *
 */
class cData
{
public:

    DataTypes data_type; /**< TODO */
    DatasetTypes mode; /**< TODO */
    cv::Mat image; /**< TODO */
    cv::Mat annotation_orig; /**< TODO */
    cv::Mat annotation_indexed; /**< TODO */

    cv::Mat superpixel_segments; /**< TODO */
    cv::Mat superpixel_neighbours; /**< TODO */
    int number_of_superpixels; /**< TODO */

    std::vector<cv::Mat> descriptors; //Different kinds of descriptors too /**< TODO */
    std::vector<cv::Mat> descriptors_pooled; //Pooled Descriptors for each superpixel /**< TODO */
    cv::Mat descriptors_concat_pooled; //Containing all features /**< TODO */
    std::vector<int> gt_label; //Ground truth label for each row in descriptors_concat_pooled /**< TODO */

    classdataStruct class_data; /**< TODO */



    /**
     * @brief
     *
     */
    cData()
    {
        this->data_type = DATA_SINGLE;
    }

};


/**
 * @brief
 *
 */
class cDataset : public cData
{
    /**
     * @brief
     *
     */
    struct indexStruct
    {
        int train; /**< TODO */
        int test; /**< TODO */
        DatasetTypes active_index; /**< TODO */
        /**
         * @brief
         *
         * @return int
         */
        int currentIndex()
        {
            if(active_index == TRAIN)
                return train;
            else if (active_index == TEST)
                return test;
            else return 0;
        }
    };

public:
    int number_of_files; /**< TODO */
    std::vector<std::string> files_images; /**< TODO */
    std::vector<std::string> files_labels; /**< TODO */

    indexStruct index; /**< TODO */
    std::shared_ptr<generalDataset> datasetObj; /**< TODO */

    /**
     * @brief
     *
     */
    cDataset()
    {
        this->data_type = DATA_SET;
        this->index.test=0;
        this->index.train=0;
        this->index.active_index = TRAIN;
        this->number_of_files = 0;
        this->mode = TRAIN;

    }

    //Init Functions
    /**
     * @brief
     *
     * @param dataset
     * @param ds_type
     * @return bool
     */
    bool loadDataset(std::shared_ptr<generalDataset> dataset, DatasetTypes ds_type)
    {
        this->datasetObj = dataset;
        dataset->loadDataset();
        if(ds_type==TRAIN)
        {
            files_images=dataset->files_images_train;
            files_labels=dataset->files_labels_train;
        }
        else if (ds_type == TEST)
        {
            files_images=dataset->files_images_test;
            files_labels=dataset->files_labels_test;
        }
        else
            return false;

        this->number_of_files = std::min(files_labels.size(), files_images.size());
        return true;
    }

    /**
     * @brief
     *
     */
    void debugDataset()
    {
        std::cout<<"Number of files : "<<this->number_of_files<<std::endl;
        std::cout<<"File Names : -"<<std::endl;
        for(int i = 0;i<this->number_of_files; i++)
        {
            std::cout<<files_images[i]<<", "<<files_labels[i]<<std::endl;
        }
    }

    /**
     * @brief
     *
     * @param cl_data
     * @return bool
     */
    bool initClassData(std::shared_ptr<ClassData> cl_data)
    {
        this->class_data.dataset = cl_data->dataset_name;
        this->class_data.class_names = cl_data->label_names;
        this->class_data.class_colours = cl_data->label_colours;
        this->class_data.number_of_classes = cl_data->label_colours.size();
        this->class_data.class_map = cl_data->map_indices;
        this->class_data.exists=true;

    }

    /**
     * @brief
     *
     */
    void doInitOperations()
    {
        //NOTHING TO DO HERE

    }

    //Iterate functions


    /**
     * @brief
     *
     * @param index_shifted
     * @return bool
     */
    bool exists(int index_shifted)
    {
        if(index.active_index==TRAIN)
            return((index.train + index_shifted) < number_of_files);
        else if(index.active_index==TEST)
            return((index.test + index_shifted) < number_of_files);
        else
            return false;

    }

    /**
     * @brief
     *
     * @param data_new
     */
    void load(cData &data_new)
    {
        if(this->exists(0))
        {
            data_new.annotation_orig = cv::imread(this->files_labels[index.currentIndex()]);
            data_new.image = cv::imread(this->files_images[index.currentIndex()]);
            data_new.data_type = DATA_SINGLE;
            data_new.mode = this->mode;
        }
    }

    /**
     * @brief
     *
     */
    void next()
    {
        if(index.active_index==TRAIN)
               index.train++;
         else if(index.active_index==TEST)
               index.test++;
        //Write code to load data from dataset file pair
    }


    /**
     * @brief
     *
     * @param ds_type
     */
    void changeActiveMode(DatasetTypes ds_type)
    {
        index.active_index=ds_type;
        this->loadDataset(this->datasetObj, ds_type);
    }

    /**
     * @brief
     *
     * @param data_new
     */
    void push(cData &data_new)
    {
        //Write code to copy in descriptors and gt_label
        //DO I HAVE TO? CAN I DO THIS IN SOME RESULT CLASS?
    }

    //Persistence
    /**
     * @brief
     *
     */
    void writeFile()
    {

        //USE OPENCV FILE STORAGE TO DO EVERYTHING HERE

    }
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void readFile()
    {

    }


};

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class cParams
{
public:
    std::vector<std::string> keys; /**< TODO */ /**< TODO */
    std::vector<std::string> values; /**< TODO */ /**< TODO */

};

}


#endif // DATAMODULES_H
