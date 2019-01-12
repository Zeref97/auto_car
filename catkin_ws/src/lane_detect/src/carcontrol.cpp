#include "carcontrol.h"
#include <unistd.h>

CarControl::CarControl()
{
    carPos.x = 120;
    carPos.y = 300;
    steer_publisher = node_obj1.advertise<std_msgs::Float32>("Team1_steerAngle",10);
    speed_publisher = node_obj2.advertise<std_msgs::Float32>("Team1_speed",10);
}

CarControl::~CarControl() {}

// void sleep(unsigned int mseconds){
//     clock_t goal = mseconds + clock();
//     while(goal > clock());
// }

float CarControl::errorAngle(const Point &dst)
{

    if (dst.x == carPos.x) return 0;
    if (dst.y == carPos.y) return (dst.x < carPos.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - carPos.x;
    double dy = carPos.y - dst.y; 
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}

float pre=0;
void CarControl::driverCar(const vector<Point> &left, const vector<Point> &right, float velocity, int sign)
{
    float error = pre;
    //cout << "----------------------------->" << sign <<endl;
    if (sign == 1 ){
        error = 45;
        pre = error;
    }
    else if (sign == 2 ){
        error = -45;
        pre = error;
    }
    else {
        Point avgLeft = Point(0,0);
        Point avgRight = Point(0,0);
        int countLeft=0, countRight=0;

        for(int i=0; i<left.size(); i++){
            if (left[i] != DetectLane::null){
                countLeft++;
                avgLeft.x = avgLeft.x + left[i].x;
                avgLeft.y = avgLeft.y + left[i].y;
            }
        }

        if(countLeft!=0){
            avgLeft.x = avgLeft.x/countLeft;
            avgLeft.y = avgLeft.y/countLeft;
        }

        for(int i=0; i<right.size(); i++){
            if (right[i] != DetectLane::null){
                countRight++;
                avgRight.x = avgRight.x + right[i].x;
                avgRight.y = avgRight.y + right[i].y;
            }
        }

        if(countRight!=0){
            avgRight.x = avgRight.x/countRight;
            avgRight.y = avgRight.y/countRight;
        }

        if(countLeft==0 && countRight!=0){
            error = errorAngle(avgRight);
            pre = error;
        }
        else if(countRight==0 && countLeft!=0){
            error = errorAngle(avgLeft);
            pre = error;
        }
        else if (countLeft == 0 && countRight==0)
            error = pre;
        else{
            error = errorAngle((avgLeft + avgRight) / 2);
            pre = error;
        }
    }

    // int i = left.size() - 11;
    // float error = preError;
    // while (left[i] == DetectLane::null && right[i] == DetectLane::null) {
    //     i--;
    //     if (i < 0) return;
    // }
    // if (left[i] != DetectLane::null && right[i] !=  DetectLane::null)
    // {
    //     error = errorAngle((left[i] + right[i]) / 2);
    // } 
    // else if (left[i] != DetectLane::null)
    // {
    //     error = errorAngle(left[i] + Point(laneWidth / 2, 0));
    // }
    // else
    // {
    //     error = errorAngle(right[i] - Point(laneWidth / 2, 0));
    // }


    if(sign == 1|| sign == 2){
        usleep(300000);
    }

    std_msgs::Float32 angle;
    std_msgs::Float32 speed;

    angle.data = error;
    speed.data = velocity;

    steer_publisher.publish(angle);
    speed_publisher.publish(speed);  

    if(sign == 1 || sign == 2){
        usleep(1000000);
    }
} 
