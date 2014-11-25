TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    ../../SuperpixelManager/slicsuperpixel.cpp


#OpenCV stuff
LIBS += `pkg-config opencv --libs`

#Enabling c++ '11 standard
QMAKE_CXXFLAGS += -std=c++11

#Enabling TBB
QMAKE_CXXFLAGS += -ltbb

#Enable Boost (if required)
LIBS += -L/usr/lib/ -lboost_filesystem -lboost_system -lboost_thread

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
    ../../unaryClassifier/unaryclassifier.h
