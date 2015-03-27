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
            this->folder_dump = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump2/";
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

            std::cout<<"\t Writing superpixels to "<<(this->folder_dump+filename_sup)<<std::endl;
            writeSegments(data->superpixel_segments,this->folder_dump+filename_sup);

            std::cout<<"\t Writing Descriptors to "<<(this->folder_dump+filename_desc)<<std::endl;
            writeDescriptors(data->descriptors_concat_pooled,this->folder_dump+filename_desc);

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



        virtual void finalOperations(std::shared_ptr<cvc::cData> data)
        {
            std::cout<<"Final operation for : "<<this->pipe_name<<std::endl;
        }


    };

}

#endif // SAVEPROGRESS_H
