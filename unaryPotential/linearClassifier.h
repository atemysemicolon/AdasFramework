#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H
//#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#include "linear.h"

#include <opencv2/core/core.hpp>
#include "../core/Pipeline/cvccore.h"

#include <time.h>
#include <stdlib.h>
#define OPENCV3 0

#if OPENCV3
#include<opencv2/ml.hpp> //Opencv3
#else
#include <opencv2/ml/ml.hpp> //Opencv2
#endif

namespace cvc
{
#if OPENCV3
namespace cvml=cv::ml; //Opencv3
#endif



class linearClassifier:public cPipeModule
{


public:


    std::string filename_svm;
    bool to_train;
    bool to_test;
    int target_descriptor_rows;
    std::string filename_descriptors;

    cv::Mat descriptors_all;
    std::vector<int> labels_all;
    cv::Mat labelsMat;

    cv::Mat predictedLabels;
    cv::Mat predictedLabels2;


    //Opencv3
#if OPENCV3
    cv::Ptr<cvml::SVM> svm; //Opencv3
    cvml::SVM::Params params_cv;
    cv::Ptr<cvml::TrainData> train_data;

    cv::Ptr<cvml::RTrees> rtrees;
    cvml::RTrees::Params params_rtrees;
#else
    //Opencv2
    cv::Ptr<cv::SVM> svm;
    cv::SVMParams params_cv;
    bool cross_train;
#endif


    linearClassifier()
    {
        this->pipe_name="Classifier";
        this->data_type=cvc::DataTypes::DATA_SINGLE;
        loadDefaultParams();
    }

    void loadDefaultParams()
    {
        this->target_descriptor_rows=1000;
        this->filename_svm = "Kitti_svm.svm";
        this->filename_descriptors="Kitti_descriptors.xml";
        this->cross_train = false;

    }

    void setCrossValidate(bool option)
    {
        this->cross_train=option;
    }


    //utility function - for all libraries
    void StretchorSqueezeSamples(cv::Mat &desc,std::vector<int> &labels, int target_length)
    {
        srand(time(NULL));
        cv::Mat tempDesc;
        std::vector<int> tempLabels;
        while(tempDesc.rows!=target_length)
        {
            int n = rand()%desc.rows;
            tempLabels.push_back(labels[n]);
            if(tempDesc.empty())
                tempDesc = desc.row(n).clone();
            else
                tempDesc.push_back(desc.row(n));

        }

        desc=tempDesc.clone();
        labels=tempLabels;


    }

    void resizeSamples(cv::Mat&desc, std::vector<int> &labels, int target_length)
    {
        srand(time(NULL));
        cv::Mat targetDesc;

        std::vector<int> filled(desc.rows, 0);
        std::vector<int> tempLabels;
        bool transfer_complete = false;
        int i=0;
        do
        {
            int target_row = rand()%desc.rows;

            if(filled[target_row] && (i<desc.rows))
                continue;
            else
            {
                tempLabels.push_back(labels[target_row]);
                targetDesc.push_back(desc.row(target_row).clone());
                filled[target_row]=1;
                i++;
            }

            if(targetDesc.rows==target_length)
                transfer_complete=true;


        }while(!transfer_complete);

        desc = targetDesc.clone();
        labels = tempLabels;
    }

    void randomizeSamples(cv::Mat &desc, std::vector<int> &labels)
    {
        cv::Mat tempDesc;
        std::vector<int> tempLabels;


        //Generating 0,1,2,3......N(Number of samples)
        std::vector<int> targets;
        targets.resize(labels.size());
        for(int i=0;i<labels.size();i++)
            targets[i]=i;

        //Shuffling targets
        auto engine = std::default_random_engine{};
        std::shuffle(std::begin(targets), std::end(targets), engine);

        //Transferring Row by Row - Could be a super-expensive operation
        for(int i=0;i<labels.size();i++)
        {
            if(tempDesc.empty())
                tempDesc=desc.row(targets[i]).clone();
            else
                tempDesc.push_back(desc.row(targets[i]).clone());

            tempLabels.push_back(labels[targets[i]]);
        }

        desc=tempDesc.clone();
        labels=tempLabels;



    }

    //Utility function - for all libraries
    void equalizeSamples(cv::Mat &desc, std::vector<int> &labels, int classes_nr, int target, cv::Mat &netDescriptors, std::vector<int> &netLabels)
    {

        std::multimap<int, int> label_row_map;
        std::pair<std::multimap<int,int>::iterator, std::multimap<int,int>::iterator> pr2;
        std::vector<int> label_counts;

        //Inserting elements into the Map
        for(int i=0;i<labels.size();i++)
            label_row_map.insert(std::pair<int,int>(labels[i],i));

        //Find the count of all classes
        for(int i=0;i<classes_nr;i++)
            label_counts.push_back(label_row_map.count(i));

        //Display Counts -Debug
//        for(int i=0;i<classes_nr;i++)
//            std::cout<<"nr_classes["<<i<<"] = "<<label_counts[i]<<std::endl;



        //cv::Mat netDescriptors;
        //std::vector<int> netLabels;
        //int target=1000;

        //Construct Descriptors for a particular class (Looping over for all classes)
        for(int class_i=0;class_i<classes_nr;class_i++)
        {
            cv::Mat tempDesc;
            std::vector<int> tempLabels;
            pr2 = label_row_map.equal_range(class_i);
            for(std::multimap<int, int>::iterator it = pr2.first; it!=pr2.second;++it)
            {
                tempLabels.push_back(labels[(*it).second]);
                //std::cout<<(*it).first<<", "<<((*it).second)<<std::endl;
                if (tempDesc.empty())
                    tempDesc = desc.row((*it).second).clone();
                else
                    tempDesc.push_back(desc.row((*it).second));
            }
            //std::cout<<"Descriptors constructed for class_i="<<class_i<<std::endl<<"Number of rows : "<<tempDesc.rows<<","<<tempLabels.size()<<std::endl;
            resizeSamples(tempDesc, tempLabels, target);
            //std::cout<<"After Stretching -> Descriptors constructed for class_i="<<class_i<<std::endl<<"Number of rows : "<<tempDesc.rows<<","<<tempLabels.size()<<std::endl;

            //Putting it together
            if(netDescriptors.empty())
                netDescriptors=tempDesc.clone();
            else
                netDescriptors.push_back(tempDesc);

            netLabels.insert(netLabels.end(), tempLabels.begin(), tempLabels.end());

        }

    }

    //Utility function -for all libraries
    void pushDescriptors(cv::Mat &descriptors, std::vector<int> &gt)
    {
        if(this->descriptors_all.empty())
            descriptors_all.push_back(descriptors);
        else
            cv::vconcat(this->descriptors_all, descriptors, this->descriptors_all);
        for(int i=0;i<gt.size();i++)
            if(gt[i]<0 || gt[i]>11)
                gt[i]=11;

        this->labels_all.insert(labels_all.end(), gt.begin(), gt.end());
        //std::cout<<"Classifer : "<<descriptors_stored.rows<<", "<<gt_labels.size()<<std::endl;

        //record ground Truth too!
    }

    //Utility function - To read descriptors&labels
    void readData(std::string filename)
    {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::FileNode fn = fs["Descriptors"];
        cv::read(fn, this->descriptors_all);
        fn = fs["GroundTruth"];
        cv::read(fn, this->labels_all);
        fs.release();
    }

    void writeData(std::string filename)
            {
                if(!filename.empty())
                {
                    cv::FileStorage fs;
                    fs.open(filename, cv::FileStorage::WRITE);
                    cv::write(fs, "Descriptors", this->descriptors_all);
                    cv::write(fs,"GroundTruth", this->labels_all);
                    fs.release();
                }

            }

    void setSvmFilename(std::string filename)
    {
        this->filename_svm = filename;

    }

    void setDescriptorFilename(std::string filename)
    {
        this->filename_descriptors = filename;
    }

    void setasTrainingMode(bool option)
    {
        this->to_train = option;
        this->to_test = !option;
    }

    void setasTestingMode(bool option)
    {
        this->to_test=option;
        this->to_train=!option;
    }

    void setTargetRows(int target)
    {
        this->target_descriptor_rows=target;
    }


//LibLinear
#pragma region REGION_LIBLINEAR

    struct parameter param_linear;		// set by parse_command_line
    struct problem prob;		// set by read_problem
    struct feature_node *x_space;
    struct model *model;

    //LIBLINEAR PARAMS
    void generateLinearParams()
    {

        param_linear.solver_type = L2R_LR;
        param_linear.eps = 1e-3;
        param_linear.nr_weight=0;
        //param.weight_label=weights_label.data();
        //param.weight = weights.data();

        //LibSVm
        //            param.svm_type = C_SVC;
        //            param.kernel_type = LINEAR;
        //            param.degree = 3;
        //            param.gamma = 0;	// 1/num_features
        //            param.coef0 = 0;
        //            param.nu = 0.5;
        //            param.cache_size = 1000;
        //            param.C = 1;
        //            param.eps = 1e-3;
        //            param.p = 0.1;
        //            param.shrinking = 1;
        //            param.probability = 1;
        //            param.nr_weight =0; //12;
        //            param.weight_label = weights_label.data();
        //            param.weight = weights.data();

    }
    //LIBLINEAR FORMAT EDITOR
    void convertTolibsvmFormatTrain(cv::Mat &descriptors, std::vector<int> &gt_labels)
    {

        int elements = (descriptors.cols+1)*descriptors.rows;
        prob.l  = descriptors.rows;
        prob.y = Malloc(double, prob.l);
        prob.x = Malloc(struct feature_node * , prob.l);
        x_space = Malloc(struct feature_node, elements);
        int j=0;
        for(int i=0;i<descriptors.rows;i++)
        {
            prob.x[i]=&x_space[j];
            prob.y[i]=gt_labels[i];
            for(int k=0;k<descriptors.cols;k++)
            {
                x_space[j].index=k+1;
                x_space[j].value=descriptors.at<float>(i,k);
                ++j;
            }
            x_space[j++].index=-1;
        }

    }
    //LIBLINEAR CROSS VALIDATION
    void do_cross_validation(int nr_fold)
    {
        int i;
        int total_correct = 0;
        double total_error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double *target = Malloc(double, prob.l);

        cross_validation(&prob,&param_linear,nr_fold,target);
        if(param_linear.solver_type == L2R_L2LOSS_SVR ||
           param_linear.solver_type == L2R_L1LOSS_SVR_DUAL ||
           param_linear.solver_type == L2R_L2LOSS_SVR_DUAL)
        {
            for(i=0;i<prob.l;i++)
            {
                double y = prob.y[i];
                double v = target[i];
                total_error += (v-y)*(v-y);
                sumv += v;
                sumy += y;
                sumvv += v*v;
                sumyy += y*y;
                sumvy += v*y;
            }
            printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
            printf("Cross Validation Squared correlation coefficient = %g\n",
                    ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
                    ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
                  );
        }
        else
        {
            for(i=0;i<prob.l;i++)
                if(target[i] == prob.y[i])
                    ++total_correct;
            printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
        }

        free(target);
    }
    //LIBLINEAR TRAINING
    bool trainlinear(cv::Mat &descriptors, std::vector<int> &labels, std::string &filename_model)
    {
//        std::string filename_model="test_kitti2.svm";
//        cv::Mat descriptors;
//        std::vector<int> labels;
//        readData("/home/prassanna/Development/workspace/builds/AdasFramework/PipeLine/kitti_descriptors.xml", descriptors, labels);

        convertTolibsvmFormatTrain(descriptors, labels);
//        std::vector<float> wts_float = getGroundTruthStatistics(labels, 12);
//        std::vector<int> wt_label={0,1,2,3,4,5,6,7,8,9,10,11};
//        std::vector<double> wts(wts_float.begin(), wts_float.end());

        generateLinearParams();

        if(check_parameter(&prob, &param_linear) ==NULL)
            std::cout<<"Parameters right."<<std::endl;
        else
            return false;
        std::cout<<"Prepared Data. Training.."<<std::endl;

        //do_cross_validation(10);
        model=train(&prob, &param_linear);
        std::cout<<"Trained Data. Saving..."<<std::endl;



        //svm_check_parameter(&prob, &param);

        if(save_model(filename_model.c_str(),model))
        {
            std::cout<<"Can't write file!"<<std::endl;
            exit(1);
        }
        free_and_destroy_model(&model);
        free(prob.y);
        free(prob.x);
        free(x_space);

        return true;
    }
    //FOR LIBLINEAR
    void convertTolibsvmFormatTest(cv::Mat &descriptors)
    {

        int elements = (descriptors.cols+1)*descriptors.rows;
    //    prob.l  = descriptors.rows;
    //    prob.y = Malloc(double, prob.l);
    //    prob.x = Malloc(struct svm_node * , prob.l);
        x_space = Malloc(struct feature_node, elements);
        int j=0;


            for(int k=0;k<descriptors.cols;k++)
            {
                x_space[j].index=k+1;
                x_space[j].value=descriptors.at<float>(0,k);
                ++j;
            }
            x_space[j++].index=-1;


    }
    //TESTROW-LIBLINEAR
    int testRow(cv::Mat &descriptors, struct model *model_, int nr_class, double *prob_estimates)
    {
        //std::string filename_model="test_kitti.svm";
        convertTolibsvmFormatTest(descriptors);

        predict_probability(model_, x_space, prob_estimates);
        //std::cout<<"Class-wise distribution of this node :-"<<std::endl;
        double max=0;
        int maxIndex=0;
        for(int i=0;i<nr_class;i++)
        {
            //std::cout<<prob_estimates[i]<<" , ";
            if(max<prob_estimates[i])
            {
                max=prob_estimates[i];
                maxIndex=i;
            }
        }

        //std::cout<<std::endl;

        return maxIndex;

    }
    //TEST LIBLINEAR
    cv::Mat testLinear(cv::Mat &descriptors)
    {
//        std::string filename_model="test_kitti.svm";
//        cv::Mat descriptors;
//        std::vector<int> labels;
//        readData("/home/prassanna/Development/workspace/builds/AdasFramework/PipeLine/kitti_descriptors.xml", descriptors, labels);
       // model = svm_load_model(filename_model.c_str());

        int nr_class=get_nr_class(model);
        int i=0;
        cv::Mat predictedLabels(descriptors.rows,1, CV_32F);
        for(int i=0;i<descriptors.rows;i++)
        {
            //std::cout<<"\n Row number : "<<i<<std::endl;
            cv::Mat rowMat =descriptors.row(i);
            double *prob_estimates = (double *) malloc(nr_class*sizeof(double));
            int predicted_label = testRow(rowMat, model, nr_class, prob_estimates);
            predictedLabels.at<float>(i, 0) = predicted_label;
        }

        free(x_space);

        return predictedLabels.clone();

    }
    void loadmodellinear(std::string filename)
    {
        setSvmFilename(filename);
        model = load_model(filename.c_str());

    }
    void destroyModelLinear()
    {
        free_and_destroy_model(&model);
    }

#pragma endregion REGION_LIBLINEAR





#pragma region REGION_OPENCV

    void load_model_cv(std::string filename=std::string())
    {
        if(!filename.empty())
            this->filename_svm=filename;
#if OPENCV3
        this->svm=cvml::StatModel::load<cvml::SVM>(this->filename_svm);
        this->rtrees=cvml::StatModel::load<cvml::RTrees>(this->filename_svm + ".rtrees");
#else
        this->svm = (cv::Ptr<cv::SVM>(new cv::SVM));
        this->svm->load(this->filename_svm.c_str());
#endif

    }
    void destroy_model_cv()
    {
        if(!this->svm.empty())
            this->svm->clear();
#if OPENCV3
        if(!this->rtrees.empty())
            this->rtrees->clear();
#endif
    }

    void save_model_cv()
    {
#if OPENCV3
        this->svm->save(this->filename_svm);
        this->rtrees->save(this->filename_svm + ".rtrees");
#else
        this->svm->save(this->filename_svm.c_str());
#endif
    }

    void constructTrainData()
    {

//        this->params_cv.C=1;
//        //this->train_data = cvml::TrainData::create(this->descriptors_all, cvml::ROW_SAMPLE, this->labelsMat);


#if OPENCV3
        cv::Mat labelsMat_temp(this->labels_all, true);
        labelsMat_temp.convertTo(this->labelsMat, CV_32SC1);
        this->params_cv.svmType=cvml::SVM::C_SVC;
        this->params_cv.kernelType=cvml::SVM::LINEAR;
        //this->params_cv.termCrit=(cv::TermCriteria::MAX_ITER, 1000, 1e-6);
        this->params_cv.termCrit.maxCount = 1000;
        this->params_cv.termCrit.type=cv::TermCriteria::MAX_ITER;
        this->params_cv.termCrit.epsilon=1e-6;
        this->params_cv.C=12.5; //found from cross validation
        this->params_rtrees = cvml::RTrees::Params(10,150,0,false, 15, Mat(),true,0,cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 0.01f));
        this->train_data = cvml::TrainData::create(this->descriptors_all.clone(),cvml::ROW_SAMPLE, this->labelsMat.clone());
#else
        //Opencv2
        cv::Mat labelsMat_temp(labels_all,true);
        //this->labelsMat = labelsMat_temp.clone();
        labelsMat_temp.convertTo(this->labelsMat, CV_32SC1);
        this->params_cv.svm_type=cv::SVM::C_SVC;
        this->params_cv.kernel_type=cv::SVM::LINEAR;
        this->params_cv.term_crit = cv::TermCriteria( cv::TermCriteria::MAX_ITER,
                                                   1000,
                                                   1e-6 );
        this->params_cv.C =0.5;
#endif
    }

    void train_svm_cv()
    {
        std::cout<<"Training SVM...."<<std::endl;




#if OPENCV3
        //Initializing Grids for cross validation..
        cvml::ParamGrid cgrid=cvml::SVM::getDefaultGrid(cvml::SVM::C);
        cvml::ParamGrid gammagrid=cvml::SVM::getDefaultGrid(cvml::SVM::GAMMA);
        cvml::ParamGrid pgrid=cvml::SVM::getDefaultGrid(cvml::SVM::POLY);
        cvml::ParamGrid nugrid=cvml::SVM::getDefaultGrid(cvml::SVM::NU);
        cvml::ParamGrid coeffgrid=cvml::SVM::getDefaultGrid(cvml::SVM::COEF);
        cvml::ParamGrid degreegrid=cvml::SVM::getDefaultGrid(cvml::SVM::DEGREE);

        //Grid traversal only in C
        gammagrid.logStep=0;
        pgrid.logStep=0;
        nugrid.logStep=0;
        degreegrid.logStep=0;
        coeffgrid.logStep=0;


        //cvml::SVM::Params p = this->svm->getParams();
        //std::cout<<"Training Auto SVM "<<std::endl;
        this->svm=cvml::SVM::create(this->params_cv);
        this->svm->train(train_data);
        //this->svm->trainAuto(this->train_data, 1000, cgrid, gammagrid,pgrid, nugrid,degreegrid,pgrid );

        cvml::SVM::Params p = this->svm->getParams();
        std::cout<<"C=["<<p.C<<"]"<<std::endl;
        std::cout<<"....done"<<std::endl;       
#else
        //OPENCV 2
        std::cout<<"Training Auto..."<<std::endl;
        this->svm = (cv::Ptr<cv::SVM>(new cv::SVM));

        //Initializing Grids for cross validation..
        cv::ParamGrid cgrid=cv::SVM::get_default_grid(cv::SVM::C);
        cv::ParamGrid gammagrid=cv::SVM::get_default_grid(cv::SVM::GAMMA);
        cv::ParamGrid pgrid=cv::SVM::get_default_grid(cv::SVM::POLY);
        cv::ParamGrid nugrid=cv::SVM::get_default_grid(cv::SVM::NU);
        cv::ParamGrid coeffgrid=cv::SVM::get_default_grid(cv::SVM::COEF);
        cv::ParamGrid degreegrid=cv::SVM::get_default_grid(cv::SVM::DEGREE);

        //Grid traversal only in C
        gammagrid.step=0;
        pgrid.step=0;
        nugrid.step=0;
        degreegrid.step=0;
        coeffgrid.step=0;

        //Training with Cross Validation
        if(this->cross_train)
            svm->train_auto(this->descriptors_all, this->labelsMat,
                            cv::Mat(), cv::Mat(), this->params_cv, 10,
                            cgrid, gammagrid, pgrid, nugrid, coeffgrid, degreegrid);
        else
            svm->train(this->descriptors_all, this->labelsMat, cv::Mat(), cv::Mat(), this->params_cv);
#endif

    }

    void train_rtrees_cv()
    {
#if OPENCV3
        std::cout<<"Training R-Trees"<<std::endl;
        this->rtrees = cvml::RTrees::create(this->params_rtrees);
        this->rtrees->train(train_data);
        //this->rtrees->save(this->filename_svm + ".rtrees");
        //this->rtrees->clear();
        std::cout<<"...done"<<std::endl;
#endif
    }

    void test_svm_cv(cv::Mat &testDesc)
    {

        //Opencv2 - Debug
//        cv::SVM svm;
//        std::cout<<"Testing Opencv SVM..."<<std::endl;
//        svm.load(this->filename_svm.c_str());
//        cv::SVMParams p = svm.get_params();


//        std::cout<<"C=["<<p.C<<"]"<<std::endl;

//        //sample prediction
//        int rno = 10;
//        while(rno>=0)
//        {
//        int gt_label = this->labels_all[rno];
//        int predicted_label=svm.predict(this->descriptors_all.row(rno));



//        std::cout<<gt_label<<"(Actual)->(Predicted)"<<predicted_label<<std::endl;
//        std::cout<<"Enter another number:-"<<std::endl;
//        std::cin>>rno;
//        }

//        svm.clear();
//        std::cout<<"....done"<<std::endl;

        //OPENCV 2 and 3!
        this->svm->predict(testDesc, this->predictedLabels);
        std::cout<<"Done!"<<std::endl;
    }

    void test_rtrees_cv(cv::Mat &testDesc)
    {
#if OPENCV3
        cv::Mat prob;
        this->rtrees->predict(testDesc,  this->predictedLabels2);
        //this->rtrees->predict()
        std::cout<<"Done!"<<std::endl;
#endif
    }




    void displayPredictions(cv::Mat &predictions)
    {

        for(int i=0;i<predictions.rows;i++)
                    std::cout<<"prediction at row "<<i<<" is : "<<predictions.at<float>(i,0)<<std::endl;
    }

#pragma endregion REGION_OPENCV







    void trainEverything(int class_nr)
    {
        std::cout<<"Training..."<<std::endl;
        std::vector<int> finalLabels;
        cv::Mat finalDesc;

        //Step1 - Sample construction
        this->equalizeSamples(this->descriptors_all, this->labels_all, class_nr, this->target_descriptor_rows, finalDesc, finalLabels); //Step 1
        this->labels_all=finalLabels;
        this->descriptors_all=finalDesc.clone();
        this->randomizeSamples(this->descriptors_all, this->labels_all);

        //Debug
        std::cout<<"Final Descriptor size : "<<this->descriptors_all.rows<<", "<<this->descriptors_all.cols<<", "<<this->labels_all.size()<<std::endl;

        //Step 2 - Copying to Opencv's structure
        this->constructTrainData();

        // Step 3 - Actual Training
        this->train_svm_cv();
        this->train_rtrees_cv();

        //Final operations
        this->save_model_cv();
        this->destroy_model_cv();

    }





    void processData(std::shared_ptr<cData> data)
    {
        if(this->to_train)
        {
           this->pushDescriptors(data->descriptors_concat_pooled, data->gt_label);

        }
        if(this->to_test)
        {
            std::cout<<"Predicting..."<<std::endl;

            if(this->svm.empty())
                this->load_model_cv(this->filename_svm);
            this->test_svm_cv(data->descriptors_concat_pooled);
            this->test_rtrees_cv(data->descriptors_concat_pooled);
            data->labelsPredicted=this->predictedLabels.clone();
            this->predictedLabels.release();

            if(!this->predictedLabels2.empty())
            {
                data->labelsPredicted2=this->predictedLabels2.clone();
                this->predictedLabels2.release();
            }

        }

    }

    void finalOperations(std::shared_ptr<cData> data)
    {
        if(this->to_train)
        {
            std::cout<<"Final Operation of : Classifier"<<std::endl;
            writeData(this->filename_descriptors);
            trainEverything(data->class_data.number_of_classes);
            if(!this->descriptors_all.empty())
                this->descriptors_all.release();
        }

        if(!this->svm.empty())
            this->svm->clear();

    }


}; //End of class





} //End of namespace

#endif // LINEARCLASSIFIER_H
