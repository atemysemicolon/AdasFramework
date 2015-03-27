#ifndef DATAMODULES_H
#define DATAMODULES_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../DatasetReader/datasetReader.h"
//#include "../AnnotationManager/annotationManager.h"

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
/**
 * @brief
 *
 */
struct classdataStruct
{
    std::string dataset; /**< TODO */ /**< TODO */
    bool exists; /**< TODO */ /**< TODO */
    int number_of_classes; /**< Number of semantic classes */ /**< TODO */ /**< TODO */
    std::vector<int> class_map; /**< If semantic classes need to be mapped to other semantic classes */ /**< TODO */ /**< TODO */
    std::vector<cv::Vec3b> class_colours; /**< colours on annotation of each class */ /**< TODO */ /**< TODO */
    std::vector<std::string> class_names; /**< Names of each class */ /**< TODO */ /**< TODO */

};




//MOVE TO A DIFFERENT FILE LATER
/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
enum DataTypes{DATA_SINGLE, DATA_SET};
/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
enum DatasetTypes{TRAIN, TEST, ALL};

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class cData
{
public:

    DataTypes data_type; /**< TODO */ /**< TODO */
    DatasetTypes mode; /**< TODO */ /**< TODO */
    cv::Mat image; /**< TODO */ /**< TODO */
    cv::Mat annotation_orig; /**< TODO */ /**< TODO */
    cv::Mat annotation_indexed; /**< TODO */ /**< TODO */
    cv::Mat annotation_predicted; /**< TODO */
    cv::Mat annotation_predicted2;

    cv::Mat superpixel_segments; /**< TODO */ /**< TODO */
    cv::Mat superpixel_neighbours; /**< TODO */ /**< TODO */
    int number_of_superpixels; /**< TODO */ /**< TODO */

    std::vector<cv::Mat> descriptors; //Different kinds of descriptors too /**< TODO */ /**< TODO */
    std::vector<cv::Mat> descriptors_pooled; //Pooled Descriptors for each superpixel /**< TODO */ /**< TODO */
    cv::Mat descriptors_concat_pooled; //Containing all features /**< TODO */ /**< TODO */
    std::vector<int> gt_label; //Ground truth label for each row in descriptors_concat_pooled /**< TODO */ /**< TODO */
    cv::Mat dictionary; /**< TODO */
    cv::Mat labelsPredicted; /**< TODO */
    cv::Mat labelsPredicted2;

    cv::Mat featureWeights;
    cv::Mat featureBias;

    classdataStruct class_data; /**< TODO */ /**< TODO */

    std::string filename_current;




    /**
     * @brief
     *
     */
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
    /**
     * @brief
     *
     */
    struct indexStruct
    {
        int train; /**< TODO */ /**< TODO */
        int test; /**< TODO */ /**< TODO */
        DatasetTypes active_index; /**< TODO */ /**< TODO */
        /**
         * @brief
         *
         * @return int
         */
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
    int number_of_files; /**< TODO */ /**< TODO */
    std::vector<std::string> files_images; /**< TODO */ /**< TODO */
    std::vector<std::string> files_labels; /**< TODO */ /**< TODO */

    indexStruct index; /**< TODO */ /**< TODO */
    std::shared_ptr<generalDataset> datasetObj; /**< TODO */ /**< TODO */

    /**
     * @brief
     *
     */
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
//    bool initClassData(std::shared_ptr<ClassData> cl_data)
//    {
//        this->class_data.dataset = cl_data->dataset_name;
//        this->class_data.class_names = cl_data->label_names;
//        this->class_data.class_colours = cl_data->label_colours;
//        this->class_data.number_of_classes = cl_data->label_colours.size();
//        this->class_data.class_map = cl_data->map_indices;
//        this->class_data.exists=true;

//    }

    /**
     * @brief
     *
     */
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
            data_new.class_data=this->class_data;
            data_new.filename_current = this->files_images[index.currentIndex()];
        }
    }

    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     * @return bool
     */
    bool next()
    {
        int indx;
        if(index.active_index==TRAIN)
        {
               index.train++;
               indx=index.train;
        }
         else if(index.active_index==TEST)
        {
               index.test++;
               indx=index.test;
        }
        //Write code to load data from dataset file pair

        return( indx < std::min(files_images.size(), files_labels.size()) );

    }

    /**
     * @brief
     *
     */
    void startAgain()
    {
        if(index.active_index==TRAIN)
        {
               index.train=0;

        }
         else if(index.active_index==TEST)
        {
               index.test=0;

        }

    }


    /**
     * @brief
     *
     * @param ds_type
     */
    /**
     * @brief
     *
     * @param ds_type
     */
    void changeActiveMode(DatasetTypes ds_type)
    {
        index.active_index=ds_type;
        this->mode=ds_type;
        this->loadDataset(this->datasetObj, ds_type);
    }

    /**
     * @brief
     *
     * @param data_new
     */
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
/**
 * @brief
 *
 */
class cParams
{
public:
    std::vector<std::string> keys; /**< TODO */ /**< TODO */ /**< TODO */
    std::vector<std::string> values; /**< TODO */ /**< TODO */ /**< TODO */

};

}


#endif // DATAMODULES_H
