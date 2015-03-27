#ifndef PIPECORE_H
#define PIPECORE_H
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

    /**
     * @brief
     *
     */
    enum PipeModes {PIPE_DATA, PIPE_DATASET};
    /**
     * @brief
     *
     */
    class cPipeModule
    {
    public:
        int position; //1,2,3 ... Depending on when this module has to be executed /**< TODO */
        bool isCompleted; //true when it gets done /**< TODO */
        bool isRunning; //true when it is running /**< TODO */
        DataTypes data_type; /**< TODO */
        std::string pipe_name; /**< TODO */


        /**
         * @brief
         *
         * @param data
         * @return float
         */
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



        /**
         * @brief
         *
         * @param data
         */
        virtual void processData(std::shared_ptr<cData> data)
        {
            cout<<"Im in Process-data"<<endl;


        }

        /**
         * @brief
         *
         * @param dataset
         */
        virtual void processDataset(std::shared_ptr<cDataset> dataset)
        {
            cout<<"Im in process-dataset"<<"- Proof: "<<dataset->number_of_files<<endl;
        }

        /**
         * @brief
         *
         * @param parameters
         */
        virtual void loadParams(cParams &parameters)
        {

        }

        /**
         * @brief
         *
         */
        virtual void loadDefaultParams()
        {

        }

        virtual void finalOperations(std::shared_ptr<cvc::cData> data)
        {
            std::cout<<"Final operation for : "<<this->pipe_name<<std::endl;
        }

    };





}

#endif // PIPECORE_H
