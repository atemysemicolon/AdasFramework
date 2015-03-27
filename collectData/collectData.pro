TEMPLATE = lib
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

HEADERS += \
    ../core/Pipeline/PipeCore.h \
    ../core/Pipeline/datamodules.h \
    ../core/Pipeline/cvccore.h


#OpenCV stuff
LIBS += `pkg-config opencv --libs`

#Enabling c++ '11 standard
QMAKE_CXXFLAGS += -std=c++11

#Enabling TBB
LIBS += -ltbb

#Enable Boost (if required)
LIBS += -L/usr/lib/ -lboost_filesystem -lboost_system -lboost_thread -lboost_program_options -lboost_python
LIBS += `python-config --ldflags`
QMAKE_CXXFLAGS += `python-config --cflags`

INCLUDEPATH += /home/prassanna/Development/workspace/numpy-opencv-converter
LIBS += -L/home/prassanna/Development/workspace/numpy-opencv-converter/build
