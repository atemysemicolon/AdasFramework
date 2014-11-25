#ifndef UNARYCLASSIFIER_H
#define UNARYCLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "../core/Pipeline/cvccore.h"
#include <string>
#include <utility>
namespace cvc
{
enum CLASSIFIER_TYPES {SVM, RTREES};
    class unaryClassifier:public cPipeModule
    {
    public:

        cv::Ptr<cv::StatModel> classifyer;
        cv::Mat descriptors_stored;
        std::vector<int> gt_labels;
        cv::Ptr<cv::TrainData> trainingData;
        CLASSIFIER_TYPES classifier_type;
        bool load_descriptors;

        unaryClassifier()
        {
            this->pipe_name="Classifier";
            this->data_type=cvc::DataTypes::DATA_SINGLE;
        }

        void initClassifier(CLASSIFIER_TYPES model_types, bool descriptors_to_load)
        {
            this->classifier_type = model_types;
            this->load_descriptors = descriptors_to_load;
        }



        void train()
        {
            //Set up training Data before

            //Call the right Classifier here

        }

        void pushDescriptors(cv::Mat &descriptors, std::vector<int> gt)
        {
            if(this->descriptors_stored.empty())
                descriptors_stored.push_back(descriptors);
            else
                cv::vconcat(this->descriptors_stored, descriptors, this->descriptors_stored);

            this->gt_labels.insert(gt_labels.end(), gt.begin(), gt.end());
            //std::cout<<"Classifer : "<<descriptors_stored.rows<<", "<<gt_labels.size()<<std::endl;

            //record ground Truth too!
        }

        void testClassifier()
        {
            //set up a training procedure, to immediately check results -Quantitative
        }

        cv::Mat predict_as_image(std::vector<int> predicted_labels, bool save_image, std::string &location)
        {
            //generate annotations from results

            //Have option to save the image. If so, create a folder in the Dataset directory called results, and inside that with some name


        }

        std::vector<int> superpixelNeigbhours(int number,cv::Mat &nbrs)
        {
            cv::Mat rowMat = nbrs.row(number);
            std::vector<int> neybrow;
            for(int i=0;i<rowMat.cols;i++)
                if(rowMat.at<int>(0,i)<0)
                    break;
            else
                    neybrow.push_back(rowMat.at<int>(0,i));

            return neybrow;

        }

        void writeData(std::string filename)
        {

            cv::FileStorage fs;
            fs.open(filename, cv::FileStorage::WRITE);
            int max_index = std::min(this->descriptors_stored.rows, (int)this->gt_labels.size());
            cv::write(fs, "Descriptors", this->descriptors_stored);
            cv::write(fs,"GroundTruth", gt_labels);
            //cv::write(fs,"Length",max_index);



            fs.release();

        }

        void readData(std::string filename)
        {



            cv::FileStorage fs;
            fs.open(filename, cv::FileStorage::READ);
            int max_index;
            cv::FileNode fn = fs["Descriptors"];
            cv::read(fn, this->descriptors_stored);
            fn = fs["GroundTruth"];
            cv::read(fn, this->gt_labels);


            fs.release();

        }


        void processData(std::shared_ptr<cData> data)
        {
            std::cout<<"Final Operation of : Classifier"<<std::endl;
           this->pushDescriptors(data->descriptors_concat_pooled, data->gt_label);

        }

        void finalOperations(std::shared_ptr<cData> data)
        {
            train();
            writeData("descriptors.xml");

        }

    };

}
#endif // UNARYCLASSIFIER_H
