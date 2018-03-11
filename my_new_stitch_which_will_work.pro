#-------------------------------------------------
#
# Project created by QtCreator 2018-03-05T09:47:06
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = my_new_stitch_which_will_work
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

LIBS += `pkg-config opencv --libs`
