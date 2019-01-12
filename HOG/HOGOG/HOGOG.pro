#-------------------------------------------------
#
# Project created by QtCreator 2018-01-16T16:38:48
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = HOGOG
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    command.h \
    para.h

FORMS    += mainwindow.ui

LIBS += `pkg-config opencv --libs`
#INCLUDEPATH += /home/tan/opencv3.4.0/include
#INCLUDEPATH += /home/tan/opencv3.4.0/include/opencv
#LIBS    += /home/tan/opencv3.4.0/lib/*.so
