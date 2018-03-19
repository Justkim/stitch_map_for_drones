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
    Mat map;
    vector<KeyPoint> kp1;
    vector<KeyPoint> kp2;
    int flag=0;
    Mat hom=Mat::ones(3,3,CV_64F);
    Mat revhom=Mat::ones(3,3,CV_64F);
    cv::Mat m = cv::Mat::eye(3, 3, CV_64F);

    double offsetX = 0.0;
    double offsetY = 0.0;
    double savedOffsetX=0;
    double savedOffsetY=0;

    vector<Mat> savedHoms;
    vector<Mat> savedrevHoms;
    Mat revhom1;
    Mat revhom2;
    std::vector<cv::Point2f> cornersTransform(4);



    for(int i=1;i<16;i++)
    {


    stringstream conversion;
    stringstream conversion1;
    conversion <<"/home/kim/Desktop/normal_pics/"<<i<<".jpg";
    string name=conversion.str();
    conversion1 <<"/home/kim/Desktop/normal_pics/"<<i+1<<".jpg";
    if(flag==1)
    {
        map=imread("/home/kim/Desktop/r/result1.jpg");
    }
    string name1=conversion1.str();
    Mat frame1=imread(name);
    Mat frame2=imread(name1);


    if(frame1.empty())
    {
        qDebug()<<"no frame1";
         //throw an exception here

    }
    if(frame2.empty())
    {
        qDebug()<<"no frame2";
         //throw an exception here
    }

    ORB orb(60000,1.2f,8,31,0,3,ORB::HARRIS_SCORE,31);
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


    }
    else
    {

        hom=m*hom*findHomography(points2,points1,CV_RANSAC,2,homMask);
        if(i==2)
        {

            revhom2=hom.inv();

        }


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




        cv::perspectiveTransform(corners, cornersTransform, hom);


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


        savedOffsetX=offsetX;
        savedOffsetY=offsetY;

        offsetX=-offsetX;
        offsetY=-offsetY;

        if(offsetX<0)
        {
            offsetX=0;
        }
        if(offsetY<0)
        {
            offsetY=0;

        }
        double warpX=-INT64_MAX,warpY=-INT64_MAX;
        for(size_t i = 0; i < 4; i++) {

            if(cornersTransform[i].x> warpX ) {

                warpX = cornersTransform[i].x;
            }

            if(cornersTransform[i].y> warpY) {
                warpY = cornersTransform[i].y;
            }


        }

    Size size_warp(warpX+offsetX,warpY+offsetY);
//    qDebug()<<"size_warp";
//    qDebug()<<(double)map.cols<<warpX;
//    qDebug()<<(double)map.rows<<warpY;


    m.at<double>(0,2) = offsetX;
    m.at<double>(1,2) = offsetY;



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
//    Mat hom1=hom;
//    hom1.at<double>(0,2)=0;
//    hom1.at<double>(1,2)=0;





     warpPerspective(frame2,warpResult,m*hom,size_warp);
   //  warpPerspective(warpResult,invwarp,revhom,size_warp);


    namedWindow("afterwarp",WINDOW_NORMAL);
    resizeWindow("afterwarp", 600,600);
   imwrite("/home/kim/Desktop/r/warpresult.jpg",warpResult);




    flag=1;
    cv::Mat m1 = cv::Mat::eye(2, 3, CV_64F);
    m1.at<double>(0,2) = offsetX;
    m1.at<double>(1,2) = offsetY;
    //namedWindow("mapbefore",WINDOW_NORMAL);
   // resizeWindow("mapberfore", 600,600);
   // imshow("mapbefore",map);




    warpAffine(map,map,m1,Size((double)map.cols+offsetX,(double)map.rows+offsetY));
//    m1rev=m1.inv();
    namedWindow("mapafter",WINDOW_NORMAL);
    resizeWindow("mapafter", 600,600);
    imwrite("/home/kim/Desktop/r/map.jpg",map);







     Mat newmask,oldmask;

//     if((double)map.rows<warpY &&  (double)map.rows<warpX)
//     {
//         map.resize(warpResult.cols,warpResult.rows);

//     }
//     else if((double)map.rows>warpY &&  (double)map.cols<warpX)
//     {
//         warpResult.resize(warpResult.cols,map.rows);
//         map.resize(warpResult.cols,map.rows);

//     }
//     else if((double)map.rows<warpY &&  (double)map.cols>warpX)
//     {
//         warpResult.resize(map.cols,warpResult.rows);
//         map.resize(map.cols,warpResult.rows);


//     }
//     else
//     {

//         warpResult.resize(map.rows,map.cols);


//     }




//     cv::Mat oldpart(newmap,cv::Rect(offsetX,offsetY,map.cols,map.rows));
     qDebug()<<"freaking upset is"<<offsetX<<"   "<<offsetY;

//     //this block of code worked last night:/ today it just make garbeges.

        inRange(map,Scalar(1,0,0),Scalar(255,255,255),oldmask);
        inRange(warpResult,Scalar(1,0,0),Scalar(255,255,255),newmask);

//              Mat warpResultMask;
//              cvtColor(warpResult,warpResultMask,COLOR_BGR2GRAY);
//              threshold(warpResultMask,warpResultMask, 100, 255, THRESH_BINARY);

//              // 4. Splitting & adding Alpha
//              vector<Mat> channels;   // C++ version of ArrayList<Mat>
//              split(warpResult, channels);   // Automatically splits channels and adds them to channels. The size of channels = 3
//              channels.push_back(warpResultMask);   // Adds mask(alpha) channel. The size of channels = 4

//              // 5. Merging
//              Mat warpResultRoi;
//              merge(channels, warpResultRoi);
                Mat newmap(std::max((double)map.rows+offsetY,warpY),std::max((double)map.cols+offsetX,warpX),frame1.type());
               newmap.setTo(0);

             //cv::Mat mapRoi(newmap,cv::Rect(cornersTransform[0].x+offsetX,cornersTransform[0].y+offsetY,warpResult.cols,warpResult.rows));


              cv::Mat newpart(newmap,cv::Rect(0,0,warpResult.cols,warpResult.rows));
//                    namedWindow("newmap1",WINDOW_NORMAL);
//                    namedWindow("newmap1",WINDOW_NORMAL);
//                    resizeWindow("newmap1",700,700);
//                    imshow("newmap1",warpResultRoi);
                      cv::Mat oldpart(newmap,cv::Rect(0,0,map.cols,map.rows));
                 map.copyTo(oldpart);
              warpResult.copyTo(newpart,newmask);






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

//    }


    qDebug()<<"over";
     imwrite("/home/kim/Desktop/r/result1.jpg",newmap);
     Mat newmap1;
     Mat newmap2;
    if(i==1000)
    {
        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(0, 0);
        corners[1] = cv::Point2f(0, newmap.rows);
        corners[2] = cv::Point2f(newmap.cols,0);
        corners[3] = cv::Point2f(newmap.cols, newmap.rows);

        std::vector<cv::Point2f> cornersTransform(4);



        cv::perspectiveTransform(corners, cornersTransform, revhom2);
         double offsetXinv;
         double offsetYinv;


//        offsetX=abs(offsetX);
//        offsetY=abs(offsetY);
        for(size_t i = 0; i < 4; i++) {
            std::cout << "cornersTransform[" << i << "]=" << cornersTransform[i] << std::endl;
            std::cout << "corners[" << i << "]=" << corners[i] << std::endl;
            if(cornersTransform[i].x < offsetXinv) {

               offsetXinv = cornersTransform[i].x;
            }

            if(cornersTransform[i].y < offsetYinv) {
               offsetYinv = cornersTransform[i].y;
            }


        }
        offsetXinv=-offsetXinv;
        offsetYinv=-offsetYinv;
        Mat minv=Mat::eye(Size(3,3),CV_64F);

        minv.at<double>(0,2) = offsetXinv;
        minv.at<double>(1,2) = offsetYinv;
        double warpXinv=-INT64_MAX,warpYinv=-INT64_MAX;
        for(size_t i = 0; i < 4; i++) {

            if(corners[i].x + offsetXinv> warpXinv ) {

                warpXinv = cornersTransform[i].x;
            }

            if(corners[i].y + offsetYinv> warpYinv) {
                warpYinv = cornersTransform[i].y ;
            }


        }

    Size size_warp_inv(warpXinv,warpYinv);



       // warpPerspective(newmap,newmap1,revhom1,Size(newmap.cols+400,newmap.rows+400));
        warpPerspective(newmap,newmap2,/*m1rev**/minv*revhom2,(size_warp_inv));
        hom=minv*revhom2*hom;
  /*      namedWindow("newmap1",WINDOW_NORMAL);
        namedWindow("newmap1",WINDOW_NORMAL);
        resizeWindow("newmap1",700,700);
        imshow("newmap1",newmap1)*/;
        //waitkey(1);
        namedWindow("newmap2",WINDOW_NORMAL);
        namedWindow("newmap2",WINDOW_NORMAL);
        resizeWindow("newmap2",700,700);
        imshow("newmap2",newmap2);
        waitKey(1);

    }
   //  imwrite("/home/kim/Desktop/tests/result2.jpg",newmap1);





    namedWindow("blendR",WINDOW_NORMAL);
    resizeWindow("blendR", 600,600);
//        if(i==3)
//              imwrite("/home/kim/Desktop/tests/result1.jpg",newmap2);

    imshow("blendR",newmap);

    }









  return a.exec();
}

