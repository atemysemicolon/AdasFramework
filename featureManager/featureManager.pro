TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

#OpenCV stuff
LIBS += `pkg-config opencv --libs`

#Enabling c++ '11 standard
QMAKE_CXXFLAGS += -std=c++11

#Enabling TBB
QMAKE_CXXFLAGS += -ltbb

HEADERS += \
    ../core/Pipeline/cvccore.h \
    ../core/Pipeline/datamodules.h \
    ../core/Pipeline/PipeCore.h \
    cFeatures.h \
    featuremanager.h
