TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    slicsuperpixel.cpp

#OpenCV stuff
LIBS += `pkg-config opencv --libs`

#Enabling c++ '11 standard
QMAKE_CXXFLAGS += -std=c++11

#Enabling TBB
QMAKE_CXXFLAGS += -ltbb

#Enable Boost (if required)
LIBS += -L/usr/lib/ -lboost_filesystem -lboost_system -lboost_thread

HEADERS += \
    slicsuperpixel.h \
    ../core/Pipeline/datamodules.h \
    ../core/Pipeline/PipeCore.h \
    ../core/Pipeline/cvccore.h \
    csuperpixel.h
