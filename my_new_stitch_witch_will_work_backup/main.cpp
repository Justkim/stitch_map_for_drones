#include "mainwindow.h"
#include <QApplication>
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <stdlib.h>
#include <iostream>
using namespace std;
using namespace cv;
void symmetryTest(const std::vector<cv::DMatch> &matches1,const std::vector<cv::DMatch> &matches2,std::vector<cv::DMatch>& symMatches)
{
    symMatches.clear();
    for (vector<DMatch>::const_iterator matchIterator1= matches1.begin();matchIterator1!= matches1.end(); ++matchIterator1)
    {
        for (vector<DMatch>::const_iterator matchIterator2= matches2.begin();matchIterator2!= matches2.end();++matchIterator2)
        {
            if ((*matchIterator1).queryIdx ==(*matchIterator2).trainIdx &&(*matchIterator2).queryIdx ==(*matchIterator1).trainIdx)
            {
                symMatches.push_back(DMatch((*matchIterator1).queryIdx,(*matchIterator1).trainIdx,(*matchIterator1).distance));
                break;
            }
        }
    }
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
 //   MainWindow w;
     Mat map;
   // w.show();
    vector<KeyPoint> kp1;
    vector<KeyPoint> kp2;
    int flag=0;
    Mat hom=Mat::ones(3,3,CV_64F);
    Mat revhom=Mat::ones(3,3,CV_64F);
    cv::Mat m = cv::Mat::eye(3, 3, CV_64F);

    double offsetX = 0.0;
    double offsetY = 0.0;

    vector<Mat> savedHoms;
    vector<Mat> savedrevHoms;


    for(int i=1;i<4;i++)
    {


    stringstream conversion;
     stringstream conversion1;
    //conversion <<"/home/atena/Desktop/n1.jpg";
    conversion <<"/home/kim/Desktop/tests/"<<i<<".jpg";
    string name=conversion.str();
    conversion1 <<"/home/kim/Desktop/tests/"<<i+1<<".jpg";
    if(flag==1)
    {
        map=imread("/home/kim/Desktop/tests/result1.jpg");
    }
    string name1=conversion1.str();

    Mat frame1=imread(name);
    Mat frame2=imread(name1);
    if(frame1.empty())
    {
        qDebug()<<"no frame1";
    }
    if(frame2.empty())
    {
        qDebug()<<"no frame2";
    }

    ORB orb(80000,1.2f,8,31,0,3,ORB::HARRIS_SCORE,31);
//    CV_WRAP explicit ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
//        int firstLevel = 0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31 );
    Mat des1;
    Mat des2;
    Mat matchMask;
    vector<Point2f> points1, points2;
    /*
     *
     *
     *
     *
     *
     *
     * detect noises part
     *
     *
     *
     *
     * */

//    medianBlur(frame1,frame1,5);
//    medianBlur(frame2,frame2,5);
    if(flag==0)
    {
        map=frame1;

    }
    GaussianBlur(frame1,frame1,Size(5,5),20);
    GaussianBlur(frame2,frame2,Size(5,5),20);
//    namedWindow("setFilter",WINDOW_NORMAL);
//    resizeWindow("setFilter", 600,600);

//    imshow("matchR",matchResult);
//    createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );


    orb.detect(frame1,kp1);
    orb.detect(frame2,kp2);
    orb.compute(frame1,kp1,des1);
    orb.compute(frame2,kp2,des2);
    BFMatcher bf(NORM_HAMMING2,false);
    vector<vector<DMatch> > matches1;
    vector<vector<DMatch> > matches2;
    vector<cv::DMatch> good_matches1;
    vector<cv::DMatch> good_matches2;
    vector<cv::DMatch> symMatches;
    bf.knnMatch(des1,des2,matches1,5,matchMask,true);
    bf.knnMatch(des2,des1,matches2,5,matchMask,true);

    good_matches1.clear();
    for (int i = 0; i < matches1.size(); ++i)
    {
        const float ratio = 0.7;
        if (matches1[i][0].distance < ratio * matches1[i][1].distance)
        {
            good_matches1.push_back(matches1[i][0]);
        }
    }

    good_matches2.clear();
    for (int i = 0; i < matches2.size(); ++i)
    {
        const float ratio = 0.7;
        if (matches2[i][0].distance < ratio * matches2[i][1].distance)
        {
            good_matches2.push_back(matches2[i][0]);
        }
    }


    Mat matchResult;




    points1.clear();
    points2.clear();
    symmetryTest(good_matches1,good_matches2,symMatches);
//    for(size_t q = 0; q < good_matches1.size(); q++)
//    {
//        points1.push_back(kp1[good_matches1[q].queryIdx].pt);
//        points2.push_back(kp2[good_matches1[q].trainIdx].pt);
//    }
    for(size_t q = 0; q < symMatches.size(); q++)
    {
        points1.push_back(kp1[symMatches[q].queryIdx].pt);
        points2.push_back(kp2[symMatches[q].trainIdx].pt);
    }
    cv::drawMatches(frame1,kp1,frame2,kp2,symMatches,matchResult);
    namedWindow("matchR",WINDOW_NORMAL);
    resizeWindow("matchR", 600,600);

    imshow("matchR",matchResult);



    Mat homMask;
    if(flag==0)
    {
    hom=findHomography(points2,points1,CV_RANSAC,2,homMask);

    revhom=findHomography(points1,points2,CV_RANSAC,2,homMask);
    }
    else
    {
        hom=m*hom*findHomography(points2,points1,CV_RANSAC,2,homMask);
        //revhom=hom.inv();
    }

    Mat warpResult;
    double offsetX;
    double offsetY;
    double maxX;
    double maxY;


        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(0, 0);
        corners[1] = cv::Point2f(0, frame2.rows);
        corners[2] = cv::Point2f(frame2.cols,0);
        corners[3] = cv::Point2f(frame2.cols, frame2.rows);

        std::vector<cv::Point2f> cornersTransform(4);


        cv::perspectiveTransform(corners, cornersTransform, hom);
//        upRightCorner=cornersTransform[0];

        for(size_t i = 0; i < 4; i++) {
            std::cout << "cornersTransform[" << i << "]=" << cornersTransform[i] << std::endl;
            std::cout << "corners[" << i << "]=" << corners[i] << std::endl;
            if(abs(cornersTransform[i].x) > maxX) {

                maxX = abs(cornersTransform[i].x);
            }

            if(abs(cornersTransform[i].y) > offsetY) {
                maxY = abs(cornersTransform[i].y);
            }


        }
//        offsetX=abs(offsetX);
//        offsetY=abs(offsetY);
        for(size_t i = 0; i < 4; i++) {
            std::cout << "cornersTransform[" << i << "]=" << cornersTransform[i] << std::endl;
            std::cout << "corners[" << i << "]=" << corners[i] << std::endl;
            if(cornersTransform[i].x < offsetX) {

                offsetX = cornersTransform[i].x;
            }

            if(cornersTransform[i].y < offsetY) {
                offsetY = cornersTransform[i].y;
            }


        }
        offsetX=-offsetX;
        offsetY=-offsetY;
        double warpX=-INT64_MAX,warpY=-INT64_MAX;
        for(size_t i = 0; i < 4; i++) {

            if(cornersTransform[i].x + offsetX> warpX ) {

                warpX = cornersTransform[i].x;
            }

            if(cornersTransform[i].y + offsetY> warpY) {
                warpY = cornersTransform[i].y;
            }


        }

    Size size_warp(warpX,warpY);
    qDebug()<<"size_warp";
    qDebug()<<(double)map.cols<<warpX;
    qDebug()<<(double)map.rows<<warpY;

    m.at<double>(0,2) = offsetX;
    m.at<double>(1,2) = offsetY;


//    //Get max offset outside of the image
//    qDebug()<<"max?";
//    qDebug()<<map.total();


    // offsetY = 100;



    qDebug()<<"///////////////////////";
    //    for(int i=0;i<3;i++)
    //    {

    //        for(int j=0;j<3;j++)
    //            qDebug()<<m.at<double>(i,j);
    //    }
    qDebug()<<"result Homography"<<endl;
    for(int i=0;i<hom.rows;i++)
    {
        for(int j=0;j<hom.cols;j++)
        {
            qDebug()<<(double)hom.at<double>(i,j)<<"  ";
        }
                        qDebug()<<endl;
    }


    qDebug()<<"result Homography"<<endl;
    for(int i=0;i<revhom.rows;i++)
    {
        for(int j=0;j<revhom.cols;j++)
        {
            qDebug()<<(double)revhom.at<double>(i,j)<<"  ";
        }
          qDebug()<<endl;
    }








//     savedHoms.push_back(hom);
//     savedrevHoms.push_back(revhom);
//    namedWindow("afterwarpnom",WINDOW_NORMAL);
//    resizeWindow("afterwarp", 600,600);
//     imshow("afterwarpnom",warpResult);

//    Mat invwarp;

     warpPerspective(frame2,warpResult,hom,size_warp);
   //  warpPerspective(warpResult,invwarp,revhom,size_warp);


    namedWindow("afterwarp",WINDOW_NORMAL);
    resizeWindow("afterwarp", 600,600);
     imshow("afterwarp",warpResult);




    flag=1;
    cv::Mat m1 = cv::Mat::eye(2, 3, CV_64F);
    m1.at<double>(0,2) = offsetX;
    m1.at<double>(1,2) = offsetY;
    namedWindow("mapbefore",WINDOW_NORMAL);
    resizeWindow("mapberfore", 600,600);
    imshow("mapbefore",map);




    warpAffine(map,map,m1,Size(max(map.cols+offsetX,warpX),max(map.rows+offsetY,warpY)));
    namedWindow("mapafter",WINDOW_NORMAL);
    resizeWindow("mapafter", 600,600);
    imshow("mapafter",map);







     Mat newmask,oldmask;
     Mat newmap(std::max((double)map.rows+offsetY,warpY),std::max((double)map.cols+offsetX,warpX),frame1.type());

     newmap.setTo(0);


     cv::Mat oldpart(newmap,cv::Rect(0,0,map.cols,map.rows));
     //cv::Mat oldpart(map1,cv::Rect(offsetX,offsetY,map.cols,map.rows));
     //this block of code worked last night:/ today it just make garbeges.
     inRange(warpResult,Scalar(1,0,0),Scalar(255,255,255),newmask);
     inRange(map,Scalar(1,0,0),Scalar(255,255,255),oldmask);
     map.copyTo(newmap,oldmask);
     warpResult.copyTo(newmap,newmask);

//the dumbest way to do the blending:)
//    for(int i=0;i<warpResult.rows;i++)
//    {
//        for(int j=0;j<warpResult.cols;j++)
//        {
//            cv::Vec3b & pixel = warpResult.at<Vec3b>(i,j);
//            uchar & B = pixel[0];
//            uchar & G = pixel[1];
//            uchar & R = pixel[2];


//            if(R==0 && G==0 && B==0 )
//            {
//                if (i<map.rows && j<map.cols)
//                     warpResult.at<Vec3b>(i,j)=map.at<Vec3b>(i,j);




//            }

//        }

   // }
    qDebug()<<"over";
     imwrite("/home/kim/Desktop/tests/result1.jpg",newmap);
     //Mat newmap1;
//    if((i+1)%2==1)
//    {
//        qDebug()<<"on loop";
//        Mat correctionHom=savedHoms[i-1];

//        Mat correctionrevHom=correctionHom.inv();
//        Size size_correctedMap(1000,1000);//this is wrong !
//        warpPerspective(newmap1,newmap,correctionrevHom,size_correctedMap);
//        hom=hom/correctionHom;
//    }
   //  imwrite("/home/kim/Desktop/tests/result2.jpg",newmap1);





    namedWindow("blendR",WINDOW_NORMAL);
    resizeWindow("blendR", 600,600);

    imshow("blendR",newmap);

    }









  return a.exec();
}

