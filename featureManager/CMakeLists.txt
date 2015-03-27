cmake_minimum_required(VERSION 3.1)
project(featureManager)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    ../core/Pipeline/cvccore.h
    ../core/Pipeline/datamodules.h
    ../core/Pipeline/PipeCore.h
    LBP/histogram.cpp
    LBP/histogram.hpp
    LBP/lbp.cpp
    LBP/lbp.hpp
    cFeatures.h
    featuremanager.h
    main.cpp)

# configure opencv
find_package( OpenCV REQUIRED )

add_executable(featureManager ${SOURCE_FILES})
target_link_libraries( featureManager ${OpenCV_LIBS} )
