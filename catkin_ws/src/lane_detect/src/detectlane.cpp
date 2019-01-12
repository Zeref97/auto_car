#include "detectlane.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <math.h>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

//#define VIDEO_NAME "/home/tan/Downloads/Cuộc Đua Số - Trang chủ - Facebook.mp4"     //Link video
#define Hred_max 130            //Ngưỡng H đỏ max
#define Hred_min 10             //Ngưỡng H đỏ min
#define Sred_min 70             //Ngưỡng S đỏ min
#define Vred_min 30             //Ngưỡng V đỏ min
#define Hblue_max 120           //Ngưỡng H xanh max
#define Hblue_min 100           //Ngưỡng H xanh min
#define Sblue_max 94            //Ngưỡng S xanh max
#define Sblue_min 64            //Ngưỡng S xanh min
#define Vblue_max 255           //Ngưỡng V xanh max
#define Vblue_min 64            //Ngưỡng B xanh min
#define Hwhite_min 0            //Ngưỡng H trắng min
#define Lwhite_min 100          //Ngưỡng L trắng min
#define Swhite_min 50           //Ngưỡng S trắng min
#define ratio_max 1.1           //Tỉ lệ cạnh hình chữ nhật bao quanh lớn nhất để lọc hình học
#define ratio_min 0.6           //Tỉ lệ cạnh hình chữ nhật bao quanh nhỏ nhất để lọc hình học
#define MIN_OBJECT_AREA_SIGN 900            //Kích thước đối tượng tối thiểu nghi ngờ là biển báo
#define MAX_OBJECT_AREA_SIGN 153600          //Kích thước đối tượng tối đa nghi ngờ là biển báo
#define MIN_OBJECT_COLOR_SEG 100             //Kích thước đối tượng tối thiểu nghi ngờ là biển báo sau khi phân đoạn màu
#define MAX_OBJECT_COLOR_SEG 153600           //Kích thước đối tượng tối đa nghi ngờ là biển báo sau khi phân đoạn màu
#define max_vertices 7                      //Số đỉnh tối thiểu để là hình tròn hoặc Ellipse
#define ratio_area_max 0.45//0.42         //0.15              //Độ lệch max của tỉ lệ diện tích của hình ellipse
#define ratio_shape_max 0.4//0.38        //0.24              //Độ lệch max của tỉ lệ hình học của hình ellipse
#define E_max 1                     //1                 //Độ lệch tâm max của ellipse
#define E_min 0.65                  //0.968             //Độ lệch tâm min của ellipse

//Sử dụng đặc trưng HOG (Histogram of gradient)
HOGDescriptor hog(
        Size(36,36), //winSize
        Size(6,6), //blocksize
        Size(6,6), //blockStride,
        Size(6,6), //cellSize,
                 9, //nbins,
                  1, //derivAper,
                 -1, //winSigma,
                  0, //histogramNormType,
                0.2, //L2HysThresh,
                  0,//gammal correction,
                  64,//nlevels=64
                  1);

/**********************************Cấu trúc ghi vào file*************************************************/
struct position{
    int index;
    int sign;
    Rect rect;
};

vector<position> pos;
int k=0;

/********************************************************************************************************/

/***********************************Load file svm đã training********************************************/

//cv::Ptr<cv::ml::SVM> mSvm = SVM::load<SVM>("train.xml");
Ptr<SVM> svm = Algorithm::load<SVM>("/home/tan/catkin_ws/src/lane_detect/src/train.xml");             //Load bien xanh
Ptr<SVM> svm1 = Algorithm::load<SVM>("/home/tan/catkin_ws/src/lane_detect/src/train1.xml"); 
// Ptr<SVM> svm2= Algorithm::load<SVM>("train2.xml");            //Load bien trang

/********************************************************************************************************/

/************************************Biến đổi hình thái học**********************************************/
void Erosion(Mat src, Mat &dst, int erosion_elem, int erosion_size){
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( src, dst, element );
}

void Dilation(Mat src, Mat &dst, int dilation_elem, int dilation_size ){
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );

  /// Apply the dilation operation
  dilate(src, dst, element );
}

void BinaryOpening(Mat src, Mat &dst, int erosion_size, int dilation_size){
    Mat elementErode = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );

    Mat elemenDilate = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );

    erode(src,dst,elementErode);
    dilate(dst, dst, elemenDilate );
}

void BinaryClosing(Mat src, Mat &dst, int dilation_size, int erosion_size){
    Mat elementErode = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );

    Mat elemenDilate = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );
    dilate(src, dst, elemenDilate );
    erode(dst,dst,elementErode);
}

/***************************************************************************************************/


/****************************************Lọc diện tích**********************************************/

void removeSmallOject(Mat src, Mat &dst, int min_size, int max_size){
    vector<vector<Point> > contours;
    findContours(src, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

    src.copyTo(dst);

    for(int i=0;i<contours.size();i++){
        double S=contourArea(contours[i]);
        //Kiem tra dieu kien neu nho hon min_size hoac lon hon max_size thi dua cac diem anh trong contour ve 0 (mau den)
        if (fabs(S) <= min_size)
            drawContours(dst, contours, i, Scalar::all(0), CV_FILLED);
        else if (fabs(S) > max_size)
            drawContours(dst, contours, i, Scalar::all(0), CV_FILLED);
        else
            continue;
    }
}

/***************************************************************************************************/

/******************************************Phân đoạn màu********************************************/

void redColorSegmentation(Mat src, Mat &dst){
    Mat hsv_frame;
    cvtColor(src, hsv_frame, CV_BGR2HSV);

    for(int i=0; i<hsv_frame.rows;i++){
        for(int j=0; j<hsv_frame.cols;j++){
            Vec3b& v=hsv_frame.at<Vec3b>(i,j);
            if((v[0]>Hred_min) && (v[0]<Hred_max))
                continue;
            else{
                if((v[1]>Sred_min) && (v[2]>Vred_min))
                    dst.at<uchar>(i,j)=255;
                else
                    continue;
            }
        }
    }
    removeSmallOject(dst, dst, MIN_OBJECT_COLOR_SEG, MAX_OBJECT_COLOR_SEG);
}

void blueColorSegmentation(Mat src, Mat &dst){
    Mat hsv_frame;
    cvtColor(src, hsv_frame, CV_BGR2HSV);

    for(int i=0; i<hsv_frame.rows; i++)
        for(int j=0; j<hsv_frame.cols; j++){
            Vec3b& v=hsv_frame.at<Vec3b>(i,j);
            if((v[0]<Hblue_min) || (v[0]>Hblue_max))
                continue;
            else{
                if((v[2]>Vblue_max) || (v[2]<Vblue_min) || (v[1]<Sblue_min))
                    continue;
                else if(v[1] > Sblue_max)
                    dst.at<uchar>(i,j)=255;
                else
                    continue;
            }
        }
    removeSmallOject(dst, dst, MIN_OBJECT_COLOR_SEG, MAX_OBJECT_COLOR_SEG);
}

void whiteColorSegmentation(Mat src, Mat &dst){
    Mat hsv_frame;
    cvtColor(src, hsv_frame, CV_BGR2HLS);

    //Mat temp=Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
    //inRange(hsv_frame, Scalar(Hwhite_min, Lwhite_min, Swhite_min), cv::Scalar(255, 255, 255), temp);
    //imshow("temp", temp);

    for(int i=0; i<src.rows; i++)
        for(int j=0; j<src.cols; j++){
            Vec3b& v=src.at<Vec3b>(i,j);
            Vec3b& k=hsv_frame.at<Vec3b>(i,j);

            //Tính f theo công thưc f(R,G,B)=(|R-G|+|G-B|+|B-R|)/2*D, với D=20 để tính achromatic
            float f=(fabs(v[2]-v[1])+fabs(v[1]-v[0])+fabs(v[0]-v[2]))/(3*20);
            if(f<1){
                if(k[1]<Lwhite_min)
                    continue;
                else{
                    if(k[2]>Swhite_min)
                        dst.at<uchar>(i,j)=255;
                    else
                        continue;
                }
            }
            else
                continue;
        }
    BinaryOpening(dst, dst, 3, 3);
    BinaryClosing(dst, dst, 7, 3);
    removeSmallOject(dst, dst, MIN_OBJECT_COLOR_SEG, MAX_OBJECT_COLOR_SEG);
}

/*****************************************************************************************************************/

/************************************Lọc diện tích và tỉ lệ cạnh của đối tượng************************************/

void areaAndRatio(Mat src, Mat &dst){
    dilate(src, src, 5);
    erode(src, src, 5);
    vector<vector<Point> > contours;
    findContours(src, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

    dst=Mat::zeros(Size(src.cols, src.rows), src.type());

    for(int i=0;i<contours.size();i++){
        double S=contourArea(contours[i]);
        if ((S > MIN_OBJECT_AREA_SIGN) && (S < MAX_OBJECT_AREA_SIGN)){
            Rect rect=boundingRect(Mat(contours[i]));
            double ratio= (double)rect.width/ (double)rect.height;
            //Trường hợp biển nằm ngoài khung hình
            /*if((rect.x>15)&&(rect.y>15) && (rect.x+rect.width<src.cols-15)
                &&(rect.y+rect.height<src.rows-15) && (ratio>ratio_min) && (ratio<ratio_max))*/
            if((ratio>ratio_min) && (ratio<ratio_max)){
                drawContours(dst, contours, i, Scalar::all(255), CV_FILLED);
            }
            else
                continue;
        }
        else
            continue;
    }
}

/************************************************************************************************************/

/*****************************************Phát hiện hình Ellipse*********************************************/

bool detectShape(Mat src, Rect rect){
    Mat temp;
    src(rect).copyTo(temp);
    //imshow("temp", temp);

    int filled_blob_height, filled_blob_width;
    float f_x_y = 0;
    float micro_20 = 0;	//dùng để tính I
    float micro_02 = 0;	//dùng để tính I
    float micro_11 = 0;	//dùng để tính I
    float micro_00 = 0;	//dùng để tính I
    float I = 0;
    float E = 0;

    filled_blob_height = temp.rows;
    filled_blob_width = temp.cols;

    int x_center = filled_blob_width / 2;
    int y_center = filled_blob_height / 2;

    for (int y = 0; y < filled_blob_height; y++){
        for (int x = 0; x < filled_blob_width; x++){
            f_x_y = temp.at<uchar>(x, y)/255;
            micro_20 += (x - x_center) * (x - x_center) * f_x_y;
            micro_02 += (y - y_center) * (y - y_center) * f_x_y;
            micro_11 += (x - x_center) * (y - y_center) * f_x_y;
            micro_00 += f_x_y;
        }
    }

    //tinh I1
    I = (micro_20 * micro_02 - micro_11 * micro_11) / (micro_00 * micro_00 * micro_00 * micro_00);

    //tinh E ap dung cong thuc
    if (I <= (1 / 16 / 3.14159 / 3.14159))
        E = 16 * 3.14159 * 3.14159 * I;
    else
        E = 1 / (16 * 3.14159 * 3.14159 * I);

    cout<<E<<endl;
    if ((E >= E_min) && (E <= E_max))
        return true; //circle dung
    else
        return false; //nothing
}

void detectEllipse(Mat src, Mat &dst){
    vector<vector<Point> > contours;
    vector<vector<Point> > temp;
    Mat x;

    src.copyTo(x);

    findContours(src, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

    temp.resize(contours.size());

    dst=Mat::zeros(Size(src.cols, src.rows), CV_8UC1);

    for(int i=0;i<contours.size();i++){
        //convexHull(contours[i], temp1[i]);
        approxPolyDP(Mat(contours[i]), temp[i], 0.02*arcLength(contours[i], true), true);
        if(temp[i].size()>max_vertices){
            double S = contourArea(contours[i]);
            Rect r = boundingRect(contours[i]);
            int radius = r.width / 2;
            float ff=abs(1 - (double)(S / (CV_PI * (radius*radius))));
            //cout<<ff<<endl;

            if ((abs(1 - ((double)r.width / r.height)) <= ratio_shape_max) && (abs(1 - (double)(S / (CV_PI * (radius*radius)))) <= ratio_area_max)){
                //imshow("x", x);
                if(detectShape(x, r))
                    drawContours(dst, contours, i, Scalar::all(255), CV_FILLED);
                else
                    continue;
            }
            else
                continue;
        }
    }
}

/***************************************************************************************************************************/

/************************************************Nhận diện loại biển báo theo yêu cầu***************************************/

int trafficBlueSignRecognize(Mat src, Rect rect){

    /*0-> Bien 02
     *1-> Bien 08
     *2-> Bien 03
     *3-> Negative
     *4-> Bien 09  */

    Mat temp;
    src(rect).copyTo(temp);
    vector<float> feature;
    resize(temp, temp, Size(36,36));
    cvtColor(temp, temp, CV_BGR2GRAY);
    hog.compute(temp, feature);
    Mat data = Mat(1 , feature.size(), CV_32FC1);
    for(size_t t=0; t<feature.size(); t++)
        data.at<float>(0, t) = feature.at(t);
    cout<<svm->predict(data)<<endl;
    return svm->predict(data);
}
int countx = 0;
void recognizeBlueSign(Mat &src, Mat& dst, int& sign){
    dst=Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
    vector<vector<Point> > contours;
    blueColorSegmentation(src, dst);
    Dilation(dst, dst, 2, 5);
    Erosion(dst, dst, 2, 5);
    BinaryClosing(dst, dst, 9, 5);
    areaAndRatio(dst,dst);
    //medianBlur(dst, dst, 5);
    //detectEllipse(dst, dst);
    findContours(dst, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    //imshow("dst",dst);
    //countx++;
    for(int i=0;i<contours.size();i++){
        Rect r = boundingRect(Mat(contours[i]));
        //string str1 = "/home/tan/catkin_ws/pic/";
        //str1.append(to_string(countx));
        //str1.append(".jpg");
        //imwrite(str1 ,src(r));
        // if(trafficBlueSignRecognize(src, r)==0){
        //     rectangle(src, r, Scalar(0,0,255),1,8,0);
        //     sign = 0;
        //     putText(src, "unknow", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
        //     k++;
        //     position p;
        //     p.index = k;
        //     p.sign = 2;
        //     p.rect = r;
        //     pos.push_back(p);
        // }
        if(trafficBlueSignRecognize(src, r)==1){
            rectangle(src, r, Scalar(0,0,255),1,8,0);
            sign = 1;
            putText(src, "turn_left", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
            k++;
            position p;
            p.index = k;
            p.sign = 8;
            p.rect = r;
            pos.push_back(p);
        }
        if(trafficBlueSignRecognize(src, r)==2){
            rectangle(src, r, Scalar(0,0,255),1,8,0);
            sign = 2;
            putText(src, "turn_right", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
            k++;
            position p;
            p.index = k;
            p.sign = 3;
            p.rect = r;
            pos.push_back(p);
        }
    }
}

int obstaclePredict(Mat src, Rect rect){

    /*0-> Bien 02
     *1-> Bien 08
     *2-> Bien 03
     *3-> Negative
     *4-> Bien 09  */

    Mat temp;
    src(rect).copyTo(temp);
    vector<float> feature;
    resize(temp, temp, Size(36,36));
    cvtColor(temp, temp, CV_BGR2GRAY);
    hog.compute(temp, feature);
    Mat data = Mat(1 , feature.size(), CV_32FC1);
    for(size_t t=0; t<feature.size(); t++)
        data.at<float>(0, t) = feature.at(t);
    //cout<<svm1->predict(data)<<endl;
    return svm1->predict(data);
}

void obstacleDetection(Mat &src, Mat& dst){
    //Mat dst=Mat::zeros(src.size(), src.type());
    //Mat tmp;
    //Rect crop(src.cols/6, src.rows/3, 4*src.cols/6, src.rows/3);
    //src(crop).copyTo(dst(crop));
    for(int i=src.cols/6; i<=4*src.cols/6; i+=src.cols/6){
        Rect r(i, src.rows/3, src.cols/6, src.rows/3);
        // countx++;
        // string str1 = "/home/tan/catkin_ws/whole/";
        // str1.append(to_string(countx));
        // str1.append(".jpg");
        // imwrite(str1 ,src(r));
        if(obstaclePredict(src, r)==1){
            rectangle(src, r, Scalar(0,0,255),1,8,0);
            putText(src, "obstuction", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
        }
    }
    //imshow("crop", tmp);
}
// int trafficRedSignRecognize(Mat src, Rect rect){
//     /*Bien do
//      *0->01
//      *1->05
//      *2->04
//      *3->Negative
//      *4->07
//      *5->06
//     */

//     Mat temp;
//     src(rect).copyTo(temp);
//     vector<float> feature;
//     resize(temp, temp, Size(36,36));
//     cvtColor(temp, temp, CV_BGR2GRAY);
//     hog.compute(temp, feature);
//     Mat data = Mat(1 , feature.size(), CV_32FC1);
//     for(size_t t=0; t<feature.size(); t++)
//         data.at<float>(0, t) = feature.at(t);
//     cout<<svm1->predict(data)<<endl;
//     return svm1->predict(data);
// }

// void recognizeRedSign(Mat &src, Mat& dst){
//     dst=Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
//     vector<vector<Point> > contours;
//     redColorSegmentation(src, dst);
//     Dilation(dst, dst, 2, 5);
//     Erosion(dst, dst, 2, 5);
//     BinaryClosing(dst, dst, 7, 3);
//     areaAndRatio(dst,dst);
//     medianBlur(dst, dst, 5);
//     //Dilation(dst, dst, 2, 5);
//     //Erosion(dst, dst, 2, 5);
//     //areaAndRatio(dst, dst);
//     //detectEllipse(dst, dst);
//     findContours(dst, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
//     for(int i=0;i<contours.size();i++){
//         Rect r = boundingRect(Mat(contours[i]));
//         if(trafficRedSignRecognize(src, r)==0){
//             rectangle(src, r, Scalar(0,0,255),1,8,0);
//             putText(src, "1", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
//             k++;
//             position p;
//             p.index = k;
//             p.sign = 1;
//             p.rect = r;
//             pos.push_back(p);
//         }
//         if(trafficRedSignRecognize(src, r)==1){
//             rectangle(src, r, Scalar(0,0,255),1,8,0);
//             putText(src, "5", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
//             k++;
//             position p;
//             p.index = k;
//             p.sign = 5;
//             p.rect = r;
//             pos.push_back(p);
//         }
//         if(trafficRedSignRecognize(src, r)==2){
//             rectangle(src, r, Scalar(0,0,255),1,8,0);
//             putText(src, "4", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
//             k++;
//             position p;
//             p.index = k;
//             p.sign = 4;
//             p.rect = r;
//             pos.push_back(p);
//         }
//         if(trafficRedSignRecognize(src, r)==4){
//             rectangle(src, r, Scalar(0,0,255),1,8,0);
//             putText(src, "7", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
//             k++;
//             position p;
//             p.index = k;
//             p.sign = 7;
//             p.rect = r;
//             pos.push_back(p);
//         }
//         if(trafficRedSignRecognize(src, r)==5){
//             rectangle(src, r, Scalar(0,0,255),1,8,0);
//             putText(src, "6", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
//             k++;
//             position p;
//             p.index = k;
//             p.sign = 6;
//             p.rect = r;
//             pos.push_back(p);
//         }
//     }
// }

// int trafficWhiteSignRecognize(Mat src, Rect rect){
//     /*Bien trang
//      * 0->10
//      * 1->Negative
//      * 2->10
//     */

//     Mat temp;
//     src(rect).copyTo(temp);
//     vector<float> feature;
//     resize(temp, temp, Size(36,36));
//     cvtColor(temp, temp, CV_BGR2GRAY);
//     hog.compute(temp, feature);
//     Mat data = Mat(1 , feature.size(), CV_32FC1);
//     for(size_t t=0; t<feature.size(); t++)
//         data.at<float>(0, t) = feature.at(t);
//     cout<<svm2->predict(data)<<endl;
//     return svm2->predict(data);
// }

// void recognizeWhiteSign(Mat &src, Mat& dst){
//     dst=Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
//     vector<vector<Point> > contours;
//     whiteColorSegmentation(src, dst);
//     //Dilation(dst, dst, 2, 5);
//     //Erosion(dst, dst, 2, 5);
//     areaAndRatio(dst,dst);
//     //medianBlur(dst, dst, 5);
//     //detectEllipse(dst, dst);
//     findContours(dst, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
//     for(int i=0;i<contours.size();i++){
//         Rect r = boundingRect(Mat(contours[i]));
//         if(trafficWhiteSignRecognize(src, r)==0 || trafficWhiteSignRecognize(src, r)==2){
//             rectangle(src, r, Scalar(0,0,255),1,8,0);
//             putText(src, "8", Point(r.x, r.y), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,255,0));
//             k++;
//             position p;
//             p.index = k;
//             p.sign = 8;
//             p.rect = r;
//             pos.push_back(p);
//         }
//     }
// }

// /*********************************************************************************************************************/

// /**************************************Hiện tất cả các loại biển phát hiện được***************************************/

// void trafficSignDetection(Mat src, Mat &dst){
//     dst=Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
//     redColorSegmentation(src, dst);
//     blueColorSegmentation(src, dst);
//     Dilation(dst, dst, 2, 5);
//     Erosion(dst, dst, 2, 5);
//     areaAndRatio(dst,dst);
//     medianBlur(dst, dst, 5);
//     Dilation(dst, dst, 2, 5);
//     Erosion(dst, dst, 2, 5);
//     areaAndRatio(dst, dst);
//     detectEllipse(dst, dst);
//     //whiteColorSegmentation(src, dst);
//     //areaAndRatio(dst, dst);
// }

// /*********************************************************************************************************************/

// void WriteText(vector<position> pos, int sum){
//     //string s;
//     ofstream myfile;
//     //cout<<"Rename your text file: ";
//     //cin>>s;
//     myfile.open ("Output.txt");
//     myfile<<sum<<"\n";
//     for(size_t i=0; i<pos.size();i++){
//         myfile<<pos[i].index<<" ";
//         myfile<<pos[i].sign<<" ";
//         myfile<<pos[i].rect.x<<" ";
//         myfile<<pos[i].rect.y<<" ";
//         myfile<<(pos[i].rect.x + pos[i].rect.width)<<" ";
//         myfile<<(pos[i].rect.y + pos[i].rect.height);
//         myfile<<"\n";
//     }
//     myfile.close();
//     return;
// }

int min(int a, int b)
{
    return a < b ? a : b;
}

int DetectLane::slideThickness = 10;
int DetectLane::BIRDVIEW_WIDTH = 240;
int DetectLane::BIRDVIEW_HEIGHT = 320;
int DetectLane::VERTICAL = 0;
int DetectLane::HORIZONTAL = 1;
int x, y, width, height;
Point DetectLane::null = Point();

DetectLane::DetectLane() {
    cvCreateTrackbar("LowH", "Threshold", &minThreshold[0], 179);
    cvCreateTrackbar("HighH", "Threshold", &maxThreshold[0], 179);

    cvCreateTrackbar("LowS", "Threshold", &minThreshold[1], 255);
    cvCreateTrackbar("HighS", "Threshold", &maxThreshold[1], 255);

    cvCreateTrackbar("LowV", "Threshold", &minThreshold[2], 255);
    cvCreateTrackbar("HighV", "Threshold", &maxThreshold[2], 255);

    cvCreateTrackbar("Shadow Param", "Threshold", &shadowParam, 255);
}

DetectLane::~DetectLane(){}

vector<Point> DetectLane::getLeftLane()
{
    return leftLane;
}

vector<Point> DetectLane::getRightLane()
{
    return rightLane;
}

void DetectLane::update(const Mat &src, int& sign)
{
    Mat dst, temp, src_gray;
    src.copyTo(temp);
    // cvtColor( src, src_gray, CV_BGR2GRAY );

    // threshold( src_gray, src_gray, 100, 150, 3 );
    // Canny( src_gray, src_gray, 0, 255, 3 );
    // Mat obstacle=Mat::zeros(Size(src_gray.cols, src_gray.rows), src_gray.type());
    // vector<vector<Point> > contours;
    // findContours(src_gray, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    // for(int i=0;i<contours.size();i++){
    //     drawContours(obstacle, contours, i, Scalar::all(255), CV_FILLED);
    // }
    // imshow("obstacles",obstacle);
    // countx++;
    // string str1 = "/home/tan/catkin_ws/whole/";
    // str1.append(to_string(countx));
    // str1.append(".jpg");
    // imwrite(str1 ,src);
    obstacleDetection(temp, dst);
    recognizeBlueSign(temp, dst, sign);
    
    Mat img = preProcess(src);
    
    vector<Mat> layers1 = splitLayer(img);
    vector<vector<Point> > points1 = centerRoadSide(layers1);
    // vector<Mat> layers2 = splitLayer(img, HORIZONTAL);
    // vector<vector<Point> > points2 = centerRoadSide(layers2, HORIZONTAL);

    Mat birdView, lane;
    birdView = Mat::zeros(img.size(), CV_8UC3);
    // for (int i = 0; i < points1.size(); i++)
    // {
    //    for (int j = 0; j < points1[i].size(); j++)
    //    {
    //        circle(birdView, points1[i][j], 1, Scalar(0,0,255), 2, 8, 0 );
    //    }
    // }

    // imshow("Debug", birdView);

    detectLeftRight(points1);

    
    
    lane = Mat::zeros(img.size(), CV_8UC3);

    /*for (int i = 0; i < points1.size(); i++)
     {
        for (int j = 0; j < points1[i].size(); j++)
        {
            circle(birdView, points1[i][j], 1, Scalar(0,0,255), 2, 8, 0 );
        }
    }*/

    // for (int i = 0; i < points2.size(); i++)
    //  {
    //     for (int j = 0; j < points2[i].size(); j++)
    //     {
    //         circle(birdView, points2[i][j], 1, Scalar(0,255,0), 2, 8, 0 );
    //     }
    // }

    // imshow("Debug", birdView);

    for (int i = 1; i < leftLane.size(); i++)
    {
        if (leftLane[i] != null)
        {
            circle(lane, leftLane[i], 1, Scalar(0,0,255), 2, 8, 0 );
        }
    }

    for (int i = 1; i < rightLane.size(); i++)
    {
        if (rightLane[i] != null) {
            circle(lane, rightLane[i], 1, Scalar(255,0,0), 2, 8, 0 );
        }
    }

    Point avgLeft = Point(0,0);
    Point avgRight = Point(0,0);
    int countLeft=0, countRight=0;

    for(int i=0; i<leftLane.size(); i++){
        if (leftLane[i] != null){
            countLeft++;
            avgLeft.x = avgLeft.x + leftLane[i].x;
            avgLeft.y = avgLeft.y + leftLane[i].y;
        }
    }

    if(countLeft!=0){
        avgLeft.x = avgLeft.x/countLeft;
        avgLeft.y = avgLeft.y/countLeft;
    }

    for(int i=0; i<rightLane.size(); i++){
        if (rightLane[i] != null){
            countRight++;
            avgRight.x = avgRight.x + rightLane[i].x;
            avgRight.y = avgRight.y + rightLane[i].y;
        }
    }

    if(countRight!=0){
        avgRight.x = avgRight.x/countRight;
        avgRight.y = avgRight.y/countRight;
    }

    Point avg = (avgLeft + avgRight) / 2;

    circle(temp, avg, 1, Scalar(0,255,0), 2, 8, 0 );
    circle(lane, avg, 1, Scalar(0,255,0), 2, 8, 0 );
    imshow("sign", temp);
    imshow("Lane Detect", lane);
}

Mat DetectLane::preProcess(const Mat &src)
{
    Mat imgThresholded, imgHSV, dst;

    cvtColor(src, imgHSV, COLOR_BGR2HSV);

    inRange(imgHSV, Scalar(minThreshold[0], minThreshold[1], minThreshold[2]), 
        Scalar(maxThreshold[0], maxThreshold[1], maxThreshold[2]), 
        imgThresholded);

    dst = birdViewTranform(imgThresholded);

    //imshow("Bird View", dst);

    fillLane(dst);

    Mat tmp=Mat::zeros(dst.size(), dst.type());
    Rect crop(dst.cols/6, dst.rows/3, 4*dst.cols/6, 2*dst.rows/3);
    //tmp = dst(crop);
    dst(crop).copyTo(tmp(Rect(dst.cols/6, dst.rows/3, 4*dst.cols/6, 2*dst.rows/3)));
    //imshow("crop", tmp);

    //imshow("Binary", imgThresholded);

    return tmp;
}

Mat DetectLane::laneInShadow(const Mat &src)
{
    Mat shadowMask, shadow, imgHSV, shadowHSV, laneShadow;
    cvtColor(src, imgHSV, COLOR_BGR2HSV);

    inRange(imgHSV, Scalar(minShadowTh[0], minShadowTh[1], minShadowTh[2]),
    Scalar(maxShadowTh[0], maxShadowTh[1], maxShadowTh[2]),  
    shadowMask);

    src.copyTo(shadow, shadowMask);

    cvtColor(shadow, shadowHSV, COLOR_BGR2HSV);

    inRange(shadowHSV, Scalar(minLaneInShadow[0], minLaneInShadow[1], minLaneInShadow[2]), 
        Scalar(maxLaneInShadow[0], maxLaneInShadow[1], maxLaneInShadow[2]), 
        laneShadow);

    return laneShadow;
}

void DetectLane::fillLane(Mat &src)
{
    vector<Vec4i> lines;
    HoughLinesP(src, lines, 1, CV_PI/180, 1);
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 3, CV_AA);
    }
}

vector<Mat> DetectLane::splitLayer(const Mat &src, int dir)
{
    int rowN = src.rows;
    int colN = src.cols;
    std::vector<Mat> res;

    if (dir == VERTICAL)
    {
        for (int i = 0; i < rowN - slideThickness; i += slideThickness) {
            Mat tmp;
            Rect crop(0, i, colN, slideThickness);
            tmp = src(crop);
            res.push_back(tmp);
        }
    }
    else 
    {
        for (int i = 0; i < colN - slideThickness; i += slideThickness) {
            Mat tmp;
            Rect crop(i, 0, slideThickness, rowN);
            tmp = src(crop);
            res.push_back(tmp);
        }
    }
    
    return res;
}

vector<vector<Point> > DetectLane::centerRoadSide(const vector<Mat> &src, int dir)
{
    vector<std::vector<Point> > res;
    int inputN = src.size();
    for (int i = 0; i < inputN; i++) {
        std::vector<std::vector<Point> > cnts;
        std::vector<Point> tmp;
        findContours(src[i], cnts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        int cntsN = cnts.size();
        if (cntsN == 0) {
            res.push_back(tmp);
            continue;
        }

        for (int j = 0; j < cntsN; j++) {
            int area = contourArea(cnts[j], false);
            if (area > 3) {
                Moments M1 = moments(cnts[j], false);
                Point2f center1 = Point2f(static_cast<float> (M1.m10 / M1.m00), static_cast<float> (M1.m01 / M1.m00));
                if (dir == VERTICAL) {
                    center1.y = center1.y + slideThickness*i;
                } 
                else
                {
                    center1.x = center1.x + slideThickness*i;
                }
                if (center1.x > 0 && center1.y > 0) {
                    tmp.push_back(center1);
                }
            }
        }
        res.push_back(tmp);
    }

    return res;
}

void DetectLane::detectLeftRight(const vector<vector<Point> > &points)
{
    static vector<Point> lane1, lane2;
    lane1.clear();
    lane2.clear();
    
    leftLane.clear();
    rightLane.clear();
    for (int i = 0; i < BIRDVIEW_HEIGHT / slideThickness; i ++)
    {
        leftLane.push_back(null);
        rightLane.push_back(null);
    }

    int pointMap[points.size()][20];
    int prePoint[points.size()][20];
    int postPoint[points.size()][20];
    int dis = 10;
    int max = -1, max2 = -1;
    Point2i posMax, posMax2;

    memset(pointMap, 0, sizeof pointMap);

    for (int i = 0; i < points.size(); i++)
    {
        for (int j = 0; j < points[i].size(); j++)
        {
            pointMap[i][j] = 1;
            prePoint[i][j] = -1;
            postPoint[i][j] = -1;
        }
    }

    for (int i = points.size() - 2; i >= 0; i--)
    {
        for (int j = 0; j < points[i].size(); j++)
        {
            int err = 320;
            for (int m = 1; m < min(points.size() - 1 - i, 5); m++)
            {
                bool check = false;
                for (int k = 0; k < points[i + 1].size(); k ++)
                {
                    if (abs(points[i + m][k].x - points[i][j].x) < dis && 
                    abs(points[i + m][k].x - points[i][j].x) < err) {
                        err = abs(points[i + m][k].x - points[i][j].x);
                        pointMap[i][j] = pointMap[i + m][k] + 1;
                        prePoint[i][j] = k;
                        postPoint[i + m][k] = j;
                        check = true;
                    }
                }   
                break; 
            }
            
            if (pointMap[i][j] > max) 
            {
                max = pointMap[i][j];
                posMax = Point2i(i, j);
            }
        }
    }

    for (int i = 0; i < points.size(); i++)
    {
        for (int j = 0; j < points[i].size(); j++)
        {
            if (pointMap[i][j] > max2 && (i != posMax.x || j != posMax.y) && postPoint[i][j] == -1)
            {
                max2 = pointMap[i][j];
                posMax2 = Point2i(i, j);
            }
        }
    }

    if (max == -1) return;

    while (max >= 1)
    {
        lane1.push_back(points[posMax.x][posMax.y]);
        if (max == 1) break;

        posMax.y = prePoint[posMax.x][posMax.y];
        posMax.x += 1;        
        
        max--;
    }

    while (max2 >= 1)
    {
        lane2.push_back(points[posMax2.x][posMax2.y]);
        if (max2 == 1) break;

        posMax2.y = prePoint[posMax2.x][posMax2.y];
        posMax2.x += 1;        
        
        max2--;
    }
    
    vector<Point> subLane1(lane1.begin(), lane1.begin() + 5);
    vector<Point> subLane2(lane2.begin(), lane2.begin() + 5);

    Vec4f line1, line2;

    fitLine(subLane1, line1, 2, 0, 0.01, 0.01);
    fitLine(subLane2, line2, 2, 0, 0.01, 0.01);

    int lane1X = (BIRDVIEW_WIDTH - line1[3]) * line1[0] / line1[1] + line1[2];
    int lane2X = (BIRDVIEW_WIDTH - line2[3]) * line2[0] / line2[1] + line2[2];

    if (lane1X < lane2X)
    {
        for (int i = 0; i < lane1.size(); i++)
        {
            leftLane[floor(lane1[i].y / slideThickness)] = lane1[i];
        }
        for (int i = 0; i < lane2.size(); i++)
        {
            rightLane[floor(lane2[i].y / slideThickness)] = lane2[i];
        }
    }
    else
    {
        for (int i = 0; i < lane2.size(); i++)
        {
            leftLane[floor(lane2[i].y / slideThickness)] = lane2[i];
        }
        for (int i = 0; i < lane1.size(); i++)
        {
            rightLane[floor(lane1[i].y / slideThickness)] = lane1[i];
        }
    }
}


Mat DetectLane::morphological(const Mat &img)
{
    Mat dst;

    // erode(img, dst, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)) );
    // dilate( dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)) );

    dilate(img, dst, getStructuringElement(MORPH_ELLIPSE, Size(10, 20)) );
    erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(10, 20)) );

    // blur(dst, dst, Size(3, 3));

    return dst;
}

void transform(Point2f* src_vertices, Point2f* dst_vertices, Mat& src, Mat &dst){
    Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
    warpPerspective(src, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
}

Mat DetectLane::birdViewTranform(const Mat &src)
{
    Point2f src_vertices[4];

    int width = src.size().width;
    int height = src.size().height;

    src_vertices[0] = Point(0, skyLine);
    src_vertices[1] = Point(width, skyLine);
    src_vertices[2] = Point(width, height);
    src_vertices[3] = Point(0, height);

    Point2f dst_vertices[4];
    dst_vertices[0] = Point(0, 0);
    dst_vertices[1] = Point(BIRDVIEW_WIDTH, 0);
    dst_vertices[2] = Point(BIRDVIEW_WIDTH - 105, BIRDVIEW_HEIGHT);
    dst_vertices[3] = Point(105, BIRDVIEW_HEIGHT);

    Mat M = getPerspectiveTransform(src_vertices, dst_vertices);

    Mat dst(BIRDVIEW_HEIGHT, BIRDVIEW_WIDTH, CV_8UC3);
    warpPerspective(src, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);

    return dst;
}

