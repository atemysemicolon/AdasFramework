#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
using namespace std;

namespace cvml=cv::ml;

void readData(std::string filename, cv::Mat &descriptors, std::vector<int> &gt_labels)
{



    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    int max_index;
    cv::FileNode fn = fs["Descriptors"];
    cv::read(fn, descriptors);
    fn = fs["GroundTruth"];
    cv::read(fn, gt_labels);


    fs.release();

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
        else
            std::cout<<labels[i]<<"->"<<i<<std::endl;

    }


    float min = 10000;

    for(int k=0;k<number_of_classes;k++)
    {
        histogram[k]=(1-(histogram[k]/(float)sum));
        if(min>histogram[k])
            min = histogram[k];

    }
    std::cout<<"\tClass Weights :-"<<std::endl;
    for(int k=0;k<number_of_classes;k++)
    {
        //histogram[k]=histogram[k]/min;
        std::cout<<"\t\t"<<k<<". "<<histogram[k]<<std::endl;
    }

    return histogram;

}


void train()
{
    cv::Mat desc;
    std::vector<int> labels;





    //Load Descriptors
    readData("/home/prassanna/Development/workspace/builds/AdasFramework/PipeLine/kitti_descriptors.xml", desc, labels);
    //TODO:Make option to get descriptors from data/descriptors_stored
    std::vector<float> weights = getGroundTruthStatistics(labels, 12); //TODO:Replace with number of classes from DATA

    cv::Mat labelsMat(labels,true);



    cv::Mat weightsMat(weights, true);
    //Remove - only Debug
    std::cout<<desc.rows<<", "<<desc.cols<<std::endl;
    std::cout<<labelsMat.rows<<", "<<labelsMat.cols<<", "<<labelsMat.channels()<<std::endl;
    std::cout<<weightsMat.rows<<", "<<weightsMat.cols<<", "<<labelsMat.channels()<<std::endl;
    weightsMat.reshape(1,1);

    cvml::SVM::Params params;

//    params.classWeights = weightsMat.clone();
    params.svmType = cvml::SVM::C_SVC;
    params.kernelType=cvml::SVM::LINEAR;
    params.termCrit = cv::TermCriteria( cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100000, FLT_EPSILON );




    cv::Ptr<cvml::TrainData> tdata = cvml::TrainData::create(desc, cvml::ROW_SAMPLE, labelsMat);


    cvml::ParamGrid  Cgrid = cvml::SVM::getDefaultGrid(cvml::SVM::C);
    cvml::ParamGrid  gammagrid = cvml::SVM::getDefaultGrid(cvml::SVM::GAMMA);
    gammagrid.logStep=0;
    cvml::ParamGrid  pgrid = cvml::SVM::getDefaultGrid(cvml::SVM::P);
    pgrid.logStep=0;
    cvml::ParamGrid  nugrid = cvml::SVM::getDefaultGrid(cvml::SVM::NU);
    nugrid.logStep=0;
    cvml::ParamGrid  coeffgrid = cvml::SVM::getDefaultGrid(cvml::SVM::C);
    coeffgrid.logStep=0;
    cvml::ParamGrid  degreegrid = cvml::SVM::getDefaultGrid(cvml::SVM::DEGREE);
    degreegrid.logStep=0;
    //svm->trainAuto(tdata,20000,Cgrid, gammagrid, pgrid, nugrid,coeffgrid,degreegrid);

for(params.C=1;params.C<100;params.C=params.C+10)
{
    std::cout<<"Training for C = "<<params.C<<std::flush;;
    cv::Ptr<cvml::SVM> svm = cvml::SVM::create(params);
    std::string name = "trained_" + std::to_string(params.C) + ".svm";
    svm->train(tdata);
    svm->save(name);
    svm->clear();
    std::cout<<"....done"<<std::endl;
}
}

/*
 * C++: bool SVM::trainAuto(const Ptr<TrainData>& data, int kFold=10, ParamGrid Cgrid=SVM::getDefaultGrid(SVM::C), ParamGrid gammaGrid=SVM::getDefaultGrid(SVM::GAMMA), ParamGrid pGrid=SVM::getDefaultGrid(SVM::P), ParamGrid nuGrid=SVM::getDefaultGrid(SVM::NU), ParamGrid coeffGrid=SVM::getDefaultGrid(SVM::COEF), ParamGrid degreeGrid=SVM::getDefaultGrid(SVM::DEGREE), bool balanced=false)
Parameters:
data – the training data that can be constructed using TrainData::create or TrainData::loadFromCSV.
kFold – Cross-validation parameter. The training set is divided into kFold subsets. One subset is used to test the model, the others form the train set. So, the SVM algorithm is executed kFold times.
*Grid – Iteration grid for the corresponding SVM parameter.
balanced – If true and the problem is 2-class classification then the method creates more balanced cross-validation subsets that is proportions between classes in subsets are close to such proportion in the whole train dataset.
The method trains the SVM model automatically by choosing the optimal parameters C, gamma, p, nu, coef0, degree from SVM::Params. Parameters are considered optimal when the cross-validation estimate of the test set error is minimal.

If there is no need to optimize a parameter, the corresponding grid step should be set to any value less than or equal to 1. For example, to avoid optimization in gamma, set gammaGrid.step = 0, gammaGrid.minVal, gamma_grid.maxVal as arbitrary numbers. In this case, the value params.gamma is taken for gamma.

And, finally, if the optimization in a parameter is required but the corresponding grid is unknown, you may call the function SVM::getDefaulltGrid(). To generate a grid, for example, for gamma, call SVM::getDefaulltGrid(SVM::GAMMA).
*/

void test()
{
    cv::Mat desc;
    std::vector<int> labels;


    //Load Descriptors
    readData("/home/prassanna/Development/workspace/builds/AdasFramework/PipeLine/kitti_descriptors.xml", desc, labels);
    //TODO:Make option to get descriptors from data/descriptors_stored
    //Remove - only Debug
    std::cout<<desc.rows<<", "<<desc.cols<<std::endl;
  //  std::cout<<labelsMat.rows<<", "<<labelsMat.cols<<", "<<labelsMat.channels()<<std::endl;


    std::string file_name="trained.svm";
    cv::Ptr<cvml::SVM> svm=cvml::StatModel::load<cvml::SVM>(file_name);



    int i=0;
    for(i=0;i<desc.rows;i=i+10)
    {
        std::cout<<"At descriptor row : "<<i<<std::endl;
        cv::Mat predicted;

        for(int j=0;j<10;j++)
        {
            cv::Mat rowMat  = desc.row( i+j);
            float predictedLabel = svm->predict(rowMat);
            std::cout<<labels[i+j]<<" (Actual) -> (Predicted)"<<predictedLabel<<std::endl;
        }

        cv::waitKey(0);


    }





}


int main()
{
    train();
    test();



    return 0;
}

