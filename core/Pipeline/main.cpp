#include <iostream>
#include <string.h>
#include <tbb/tbb.h>
#include <memory>

#include "datamodules.h"


namespace cvc
{
    using namespace cvc;
    using namespace std;


    class cDataset;
    class cData;
    class cParams;

    enum PipeModes {PIPE_DATA, PIPE_DATASET};
    class cPipeModule
    {
    public:
        int position; //1,2,3 ... Depending on when this module has to be executed
        bool isCompleted; //true when it gets done
        bool isRunning; //true when it is running
        DataTypes data_type;
        std::string pipe_name;


        virtual float process(std::shared_ptr<cData> data)
        {

            //All the looking good stuff
            tbb::tick_count begin = tbb::tick_count::now();
            std::cout<<"Currently on : "<<pipe_name<<std::endl;
            std::cout<<"-----------module specific output---------"<<std::endl;
            if(this->data_type==DATA_SINGLE)
                processData(data);
            else
                processDataset(std::static_pointer_cast<cDataset>(data));
            tbb::tick_count now = tbb::tick_count::now();
            std::cout<<"time taken : "<<(now-begin).seconds()<<std::endl<<std::endl;
            return (now-begin).seconds();

        }

        virtual void processData(std::shared_ptr<cData> data)
        {
            cout<<"Im in Process-data"<<endl;


        }

        virtual void processDataset(std::shared_ptr<cDataset> dataset)
        {
            cout<<"Im in process-dataset"<<"- Proof: "<<dataset->number_of_files<<endl;
        }

        virtual void loadParams(cParams &parameters)
        {

        }

        virtual void loadDefaultParams()
        {

        }

    };





}



int main()
{
    std::shared_ptr<cvc::cData> data(new cvc::cData);
    cvc::cPipeModule bla;
    bla.data_type = cvc::DATA_SINGLE;
    bla.process(data);
    std::shared_ptr<cvc::cData> dataset(new cvc::cDataset);
    bla.data_type=cvc::DATA_SET;
    bla.process(dataset);
    std::cout << "Hello World!" << std::endl;
    return 0;
}

