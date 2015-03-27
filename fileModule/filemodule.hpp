#ifndef FILEMODULE_HPP
#define FILEMODULE_HPP
#include "../core/Pipeline/cvccore.h"
#include <opencv2/core/core.hpp>
#include "../core/DatasetReader/datasetReader.h"

namespace cvc {

class fileModule{

public:




    static bool writeData(std::string filename, cvc::cData &data)
    {

        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if(!fs.isOpened())
            return false;
        //What else do we write?

        fs<<"features"<<data.descriptors_concat_pooled;
        fs<<"labels"<<data.gt_label;

        fs<<"superpixels"<<data.superpixel_segments;
        fs<<"superneighbours"<<data.superpixel_neighbours;

        //Classdata will have to be o/ped by datasetwriter

        fs.release();
        return true;
    }

    static bool readData(std::string filename, cvc::cData &data)
    {

        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if(!fs.isOpened())
            return false;
        fs["features"]>>data.descriptors_concat_pooled;
        fs["labels"]>>data.gt_label;
        fs["superpixels"]>>data.superpixel_segments;
        fs["superneighbours"]>>data.superpixel_neighbours;

        //Classdata will have to be read by datasetwriter

        //std::cout<<data.descriptors_concat_pooled<<std::endl;

        fs.release();
        return true;
    }


    static bool writeDataset(std::string filename, cvc::cDataset &dataset)
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if(!fs.isOpened())
            return false;

        //ClassData
        fs<<"name"<<dataset.class_data.dataset;
        fs<<"numberOfClasses"<<dataset.class_data.number_of_classes;
        fs<<"class_map"<<dataset.class_data.class_map;
        fs<<"class_colours"<<dataset.class_data.class_colours;
        fs<<"class_names"<<dataset.class_data.class_names;

        fs<<"index_active"<<(int)dataset.index.active_index;
        fs<<"index_current"<<dataset.index.currentIndex();

        std::string rootNameImage = "ImageFiles";
        std::string rootNameLabel= "LabelFiles";
        int n = dataset.files_images.size();
        fs<<"numberOfImages"<<n;
        for(int i=0;i<dataset.files_images.size();i++)
        {
            std::string temp = rootNameImage + std::to_string(i);
            fs<<temp<<dataset.files_images[i];
            temp = rootNameLabel + std::to_string(i);
            fs<<temp<<dataset.files_labels[i];

        }


        fs.release();
        return true;
    }

    static bool readDataset(std::string filename, cvc::cDataset &dataset)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if(!fs.isOpened())
            return false;

        //ClassData
        fs["name"]>>dataset.class_data.dataset;
        fs["numberOfClasses"]>>dataset.class_data.number_of_classes;
        fs["class_map"]>>dataset.class_data.class_map;
        fs["class_colours"]>>dataset.class_data.class_colours;
        //fs["class_names"]>>dataset.class_data.class_names;



        int temp;
        fs["index_active"]>>temp;
        dataset.index.active_index = cvc::DatasetTypes(temp);
        if(dataset.index.active_index==cvc::DatasetTypes::TRAIN)
            fs["index_current"]>>dataset.index.train;
        else
            fs["index_current"]>>dataset.index.test;


        dataset.files_images.clear();
        dataset.files_labels.clear();

        int n;
        fs["numberOfImages"]>>n;
        for(int i=0;i<n;i++)
        {
            std::string name ="ImageFiles"+std::to_string(i);
            std::string temporary_file;
            fs[name]>>temporary_file;
            dataset.files_images.push_back(temporary_file);
            name ="LabelFiles"+std::to_string(i);
            fs[name]>>temporary_file;
            dataset.files_labels.push_back(temporary_file);
        }


        //Debug
        std::cout<<"Dataset name : "<<dataset.class_data.dataset<<std::endl;
        std::cout<<"NumberOfClasses"<<dataset.class_data.number_of_classes<<std::endl;
        std::cout<<"clasS_map : "<<std::endl;
        for(int i = 0;i<dataset.class_data.class_map.size();i++)
            std::cout<<dataset.class_data.class_map[i]<<", ";

        std::cout<<"class_colours : ";
        for(int i = 0;i<dataset.class_data.class_map.size();i++)
            std::cout<<dataset.class_data.class_map[i]<<", ";

        for(int i = 0;i<dataset.files_images.size();i++)
            std::cout<<dataset.files_images[i]<<std::endl<<dataset.files_labels[i]<<std::endl<<std::endl;




        fs.release();
        return true;
    }





};

}
#endif // FILEMODULE_HPP
