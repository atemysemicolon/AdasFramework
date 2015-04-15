#ifndef SAVEPROGRESS_H
#define SAVEPROGRESS_H
#include <opencv2/opencv.hpp>
#include "../core/Pipeline/cvccore.h"
#include <iostream>

namespace cvc
{

    /*
     * Class to Dump selected data types
     * Into select files
     * Will read from python and rectify it
     * */
    class saveProgress:public cPipeModule
    {
    public:
        std::string folder_dump;

        saveProgress()
        {
            this->data_type = DATA_SINGLE;
            this->pipe_name = "SaveIntermediateResults";

        }

        void initFolderLocation(const std::string &folder_name)
        {
            this->folder_dump = folder_name;
        }



        /**
         * @brief
         *
         * @param data
         */
        void processData(std::shared_ptr<cData> data)
        {
            std::string local_filename =  data->filename_current.substr(data->filename_current.find_last_of("\\/")+1,data->filename_current.length()) ;

            std::string filename_sup = local_filename + "_sup.xml";
            std::string filename_desc = local_filename + "_desc.xml";
            std::string filename_ann = local_filename + "_ann.xml";
            std::string filename_img = local_filename + "_im.xml";
            std::string filename_nbr = local_filename + "_nbrs.xml";

            std::cout<<"\t Writing superpixels to "<<(this->folder_dump+filename_sup)<<std::endl;
            writeSegments(data->superpixel_segments,this->folder_dump+filename_sup);

            std::cout<<"\t Writing Descriptors to "<<(this->folder_dump+filename_desc)<<std::endl;
            writeDescriptors(data->descriptors_concat_pooled,this->folder_dump+filename_desc);

            std::cout<<"\t Writing Superneighbours to "<<(this->folder_dump+filename_nbr)<<std::endl;
            writeMat(data->superpixel_neighbours,this->folder_dump+filename_nbr, "Neighbours");

            std::cout<<"\t Writing Annotations to "<<(this->folder_dump+filename_ann)<<std::endl;
            writeMat(data->annotation_indexed,this->folder_dump+filename_ann, "Ann");

            std::cout<<"\t Writing Images to "<<(this->folder_dump+filename_img)<<std::endl;
            writeMat(data->image,this->folder_dump+filename_img, "Img");


            //Clearing data
            data->superpixel_segments.release();
            data->superpixel_neighbours.release();
            data->image.release();
            data->annotation_indexed.release();
            data->annotation_orig.release();
            data->descriptors_concat_pooled.release();

        }


        bool writeSegments(cv::Mat &segments, std::string filename_out)
        {
            cv::FileStorage fs(filename_out, cv::FileStorage::WRITE);
            if(!fs.isOpened())
                return false;

            fs<<"Segments"<<segments;

            fs.release();
            return true;

        }

        bool writeDescriptors(cv::Mat &descriptors, std::string filename_out)
        {
            cv::FileStorage fs(filename_out, cv::FileStorage::WRITE);
            if(!fs.isOpened())
                return false;

            fs<<"Descriptors"<<descriptors;

            fs.release();
            return true;

        }


        bool writeMat(cv::Mat &descriptors, std::string filename_out, std::string field)
        {
            cv::FileStorage fs(filename_out, cv::FileStorage::WRITE);
            if(!fs.isOpened())
                return false;

            fs<<field<<descriptors;

            fs.release();
            return true;

        }



        virtual void finalOperations(std::shared_ptr<cvc::cData> data)
        {
            std::cout<<"Final operation for : "<<this->pipe_name<<std::endl;
        }


    };

}

#endif // SAVEPROGRESS_H
