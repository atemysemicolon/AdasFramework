#ifndef UNARYCLASSIFIER_H
#define UNARYCLASSIFIER_H


#include "../core/Pipeline/cvccore.h"
#include <string>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

namespace cvc
{
enum CLASSIFIER_TYPES {SVM, RTREES};
    class unaryClassifier:public cPipeModule
    {
    public:

        cv::Ptr<cv::StatModel> classifyer;
        cv::Mat descriptors_stored;
        std::vector<int> gt_labels;
        std::vector<float> weights;
        cv::Mat labelsMat;
        cv::Mat weightsMat;

        cv::Ptr<cv::TrainData> trainingData;
        CLASSIFIER_TYPES classifier_type;
        bool load_descriptors;
        int number_classes;
        std::string filename_descriptor_out;

        CvSVMParams params;
        CvSVM svm;
        std::string filename_svm;

        bool mode_training;
        bool mode_pedicting;




        unaryClassifier()
        {
            this->pipe_name="Classifier";
            this->data_type=cvc::DataTypes::DATA_SINGLE;
        }

        void initClassifier(CLASSIFIER_TYPES model_types, bool descriptors_to_load, int number_classes)
        {
            this->classifier_type = model_types;
            this->load_descriptors = descriptors_to_load;
            this->number_classes=number_classes;

            //Default
            this->mode_training=true;

            //Default Parameters
            params.svm_type    = CvSVM::C_SVC;
            params.kernel_type = CvSVM::LINEAR;
            params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 60, 1e-5);
        }

        void setSVMParams(CvSVMParams &parameters)
        {
            this->params = parameters;
        }

        void setSVMFileName(std::string name)
        {
            this->filename_svm=name;
        }
        void setDescriptorOutFilename(std::string name)
        {
            this->filename_descriptor_out = name;
        }
        void loadSVMfromfilename()
        {
            this->svm.load(filename_svm.c_str());
            this->params = this->svm.get_params();
        }

        void setasTrainingMode(bool choice)
        {
            this->mode_training = choice;
            this->mode_pedicting = !choice;
        }

        void setasPredictMode(bool choice)
        {
            this->mode_pedicting = choice;
            this->mode_training = !choice;
        }


        std::vector<float> getGroundTruthStatistics(std::vector<int> &labels, int number_of_classes)
        {
            std::vector<float> histogram(number_of_classes, 0);
            long int sum=0;
            for(int i=0;i<labels.size();i++)
            {
                if(labels[i]>=0 && labels[i]<number_of_classes)
                {
                    histogram[labels[i]]++;
                    sum++;
                }

            }

//            for(int k=0;k<number_of_classes;k++)
//            {
//                //histogram[k]=1-(histogram[k]);
//                std::cout<<"\t\t"<<k<<". "<<histogram[k]<<std::endl;
//            }

            float min = 10000;

            for(int k=0;k<number_of_classes;k++)
            {
                histogram[k]=(1-(histogram[k]/(float)sum))*10;
                if(min>histogram[k])
                    min = histogram[k];

            }
            std::cout<<"\tClass Weights :-"<<std::endl;
            for(int k=0;k<number_of_classes;k++)
            {
                histogram[k]=histogram[k]/min;
                std::cout<<"\t\t"<<k<<". "<<histogram[k]<<std::endl;
            }
//            std::vector<float> class_weights;
//            for(int i=0;i<labels.size();i++)
//            {
//                if(labels[i]>=0 && labels[i]<number_of_classes)
//                {
//                    class_weights.push_back(histogram[labels[i]]);
//                }

//            }

            return histogram;

        }

        void genTrainingData(cv::Mat &descriptors, std::vector<int> &labels, std::vector<float> &weights)
        {
//            std::vector<float> sample_weights(labels.size(),0.0f);
//            for(int i=0;i<labels.size();i++)
//            {
//                if(labels[i]>=0 &&labels[i]<class_weights.size())
//                    sample_weights[i]=class_weights[labels[i]];
//            }

            //TODO -> ADd class_weights

            this->descriptors_stored = descriptors.clone();
            cv::Mat labelsMattemp(labels, true);
            cv::Mat weightsMattemp(weights, true);
            //cv::Mat temp2;

            //weightsMattemp.convertTo(temp2, CV_32SC1);
            this->weightsMat=weightsMattemp.clone();
            this->labelsMat=labelsMattemp.clone();

            weightsMattemp.release();
            labelsMattemp.release();
            //temp2.release();



        }



        void train()
        {

            //Set up training Data before
            std::cout<<"\tTraining the Classifier..."<<std::endl;
            this->genTrainingData(this->descriptors_stored, this->gt_labels, this->weights);
            //std::cout<<labelsMat<<std::endl;
            cv::Mat mat_temp=this->weightsMat.reshape(0,1);
            CvMat cvmat = mat_temp.clone();

            this->params.class_weights=&cvmat;

            this->svm.train(this->descriptors_stored, this->labelsMat, cv::Mat(), cv::Mat(), this->params);
            //this->svm.train(this->descriptors_stored, this->labelsMat, cv::Mat(), this->weightsMat, this->params);
           //this->svm.train_auto(this->descriptors_stored, this->labelsMat, cv::Mat(), cv::Mat(), this->params, 200);

            this->svm.save(this->filename_svm.c_str());

            std::cout<<"\tSVM Trained!"<<std::endl;

            cv::Mat I = cv::Mat::zeros(512,512, CV_8UC3);
            int x     = svm.get_support_vector_count();
            for(int i = 0; i < x; ++i)
            {
               const float* v = svm.get_support_vector(i);
               cv::circle( I,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), 3, 8);
            }


           cv::imshow("SVM for Non-Linear Training Data", I);
           cv::waitKey(0);
           cv::destroyWindow("SVM for Non-Linear Training Data");

            //Call the right Classifier here

        }



        void pushDescriptors(cv::Mat &descriptors, std::vector<int> gt)
        {
            if(this->descriptors_stored.empty())
                descriptors_stored.push_back(descriptors);
            else
                cv::vconcat(this->descriptors_stored, descriptors, this->descriptors_stored);
            for(int i=0;i<gt.size();i++)
                if(gt[i]<0 || gt[i]>11)
                    gt[i]=11;

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

        cv::Mat predictedLabels(cv::Mat &descriptors)
        {
            cv::Mat labels_predicted;
            this->svm.predict(descriptors, labels_predicted);
            //std::cout<<labels_predicted;
            return labels_predicted;
        }

        void writeData(std::string filename)
        {
            if(!filename.empty())
            {
                cv::FileStorage fs;
                fs.open(filename, cv::FileStorage::WRITE);
                int max_index = std::min(this->descriptors_stored.rows, (int)this->gt_labels.size());
                cv::write(fs, "Descriptors", this->descriptors_stored);
                cv::write(fs,"GroundTruth", gt_labels);
                //cv::write(fs,"Length",max_index);
                fs.release();
            }

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
            if(this->mode_training)
            {
               this->pushDescriptors(data->descriptors_concat_pooled, data->gt_label);

            }
            if(this->mode_pedicting)
            {
                std::cout<<"Predicting..."<<std::endl;
                data->labelsPredicted=this->predictedLabels(data->descriptors_concat_pooled).clone();
            }

        }

        void finalOperations(std::shared_ptr<cData> data)
        {
            if(this->mode_training)
            {
                std::cout<<"Final Operation of : Classifier"<<std::endl;
                writeData(filename_descriptor_out);
                this->weights= this->getGroundTruthStatistics(this->gt_labels,data->class_data.number_of_classes );
                //Classweight should somehow go into train()
                train();
                if(!this->descriptors_stored.empty())
                    this->descriptors_stored.release();
            }

        }

    };

}
#endif // UNARYCLASSIFIER_H
