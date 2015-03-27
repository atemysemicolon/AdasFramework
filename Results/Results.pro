TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

SUBDIRS += \
    ../core/Pipeline/Pipeline.pro

HEADERS += \
    ../core/Pipeline/PipeCore.h \
    ../core/Pipeline/datamodules.h \
    ../core/Pipeline/cvccore.h \
    resultsManager.h

