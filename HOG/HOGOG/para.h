#ifndef PARA
#define PARA
#include <command.h>

#define PATH "/home/tan/traffic_sign/database/training"
#define SAVE "/home/tan/traffic_sign/database/training/train.xml"
#define LOAD "/home/tan/traffic_sign/database/training/train.xml"
#define TEST "/home/tan/catkin_ws/pic/511.jpg"

#define PATH1 "//home/tan/catkin_ws/database/training"
#define SAVE1 "/home/tan/catkin_ws/database/training/train1.xml"
#define LOAD1 "/home/tan/catkin_ws/database/training/train1.xml"
#define TEST1 "/home/tan/catkin_ws/database/training/1/1.jpg"

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

#endif // PARA

