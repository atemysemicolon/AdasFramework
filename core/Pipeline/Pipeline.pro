TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    ../../SuperpixelManager/slicsuperpixel.cpp


#OpenCV stuff
LIBS += `pkg-config opencv --libs`

#Enable Boost (if required)
LIBS += -L/usr/lib/ -lboost_filesystem -lboost_system -lboost_thread -lboost_program_options

#Enabling c++ '11 standard
QMAKE_CXXFLAGS += -std=c++11

#Enabling TBB
LIBS += -ltbb


HEADERS += \
    datamodules.h \
    PipeCore.h \
    cvccore.h \
    ../DatasetReader/datasetReader.h \
    ../AnnotationManager/annotationManager.h \
    ../../SuperpixelManager/slicsuperpixel.h \
    ../../SuperpixelManager/csuperpixel.h \
    ../../featureManager/featuremanager.h \
    ../../featureManager/cFeatures.h \
    ../../fileModule/filemodule.hpp \
    ../../Results/resultsManager.h \
    ../../preProcesing/preprocessing.h \
    ../../unaryPotential/linearClassifier.h \
    ../../InterimResultsSaver/saveProgress.h


# remove possible other optimization flags
QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2

# add the desired -O3 if not present
QMAKE_CXXFLAGS_RELEASE += -O3
