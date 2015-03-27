TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    svm.cpp \
    linear.cpp \
    tron.cpp \
    blas/dscal.c \
    blas/dnrm2.c \
    blas/ddot.c \
    blas/daxpy.c

HEADERS += \
    svm.h \
    ../core/Pipeline/PipeCore.h \
    ../core/Pipeline/datamodules.h \
    ../core/Pipeline/cvccore.h \
    linear.h \
    tron.h \
    blas/blasp.h \
    blas/blas.h \
    linearClassifier.h

#OpenCV stuff
LIBS += `pkg-config opencv --libs`

#Enabling c++ '11 standard
QMAKE_CXXFLAGS += -std=c++11

#Enabling TBB
LIBS += -ltbb

#Enable Boost (if required)
LIBS += -L/usr/lib/ -lboost_filesystem -lboost_system -lboost_thread -lboost_program_options

#OPENMP
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp





