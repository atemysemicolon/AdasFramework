#-------------------------------------------------
#
# Project created by QtCreator 2014-12-18T14:13:15
#
#-------------------------------------------------

QT       -= core gui

TARGET = dataInterop
TEMPLATE = lib

DEFINES += DATAINTEROP_LIBRARY

SOURCES += datainterop.cpp

HEADERS += datainterop.h\
        datainterop_global.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}
