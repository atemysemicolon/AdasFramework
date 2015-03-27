#include <iostream>
#include <resultsManager.h>

int main()
{
    cvc::resultsManager rm;
    rm.processData(new cvc::cDataHandler);

    return 0;
}

