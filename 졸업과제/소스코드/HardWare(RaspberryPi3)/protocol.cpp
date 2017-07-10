#include <iostream>
#include <cstdio>
#include <vector>
#include <list>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>

#define NAME "/Users/HwangJinYoung/Documents/Opencv/real.mp4"

#define MAXLINE 4096
#define MAXSUB  200
#define LISTENQ         1024
#define AREA 2700
#define STANDARD1_X 200
#define STANDARD1_Y 0
#define STANDARD2_X 200
#define STANDARD2_Y 240

typedef unsigned char BYTE;
#define CV_GET_B(img,x,y) CV_IMAGE_ELEM((img), BYTE, (y), (x) * 3 + 0)
#define CV_GET_G(img,x,y) CV_IMAGE_ELEM((img), BYTE, (y), (x) * 3 + 1)
#define CV_GET_R(img,x,y) CV_IMAGE_ELEM((img), BYTE, (y), (x) * 3 + 2)

typedef struct roi{
    int id;
    int leftTop_x;
    int leftTop_y;
    int rightBot_x;
    int rightBot_y;
    CvPoint past;
    int inDirection;
    int outDirection;
    int lifetime;
}ROI;


using namespace cv;
using namespace std;

extern int h_errno;
double MASKSIZE = 10000;
list<ROI> ROIs;
list<ROI>::iterator itor;
int ROI_num = 0;

bool drawRect = false;
int rectLeftTop_x = 0;
int rectLeftTop_y = 0;
int rectRightBot_x = 0;
int rectRightBot_y = 0;

bool isCounterClock(int a, int b, int c, int d, int e, int f){
    float i;
    i = a*d + c*f + e*b - (c*b + a*f + e*d);
    if (i >= 0)
        return 1;
    return 0;
}

bool RighttoLeft(bool a,bool b,bool c, bool d){
    return (!a&&b&&c&&!d);
}

bool LefttoRight(bool a,bool b,bool c, bool d){
    return (a&&!b&&!c&&d);
}

bool UptoDown(bool a,bool b,bool c, bool d){
    return (!a&&b&&c&&!d);
}

bool DowntoUp(bool a,bool b,bool c, bool d){
    return (a&&!b&&!c&&d);
}

ssize_t process_http(int sockfd, const char *host, const char *page, char *poststr){
    char sendline[MAXLINE + 1], recvline[MAXLINE + 1];
    ssize_t n;
    printf("connect ok\n\n\n");
    snprintf(sendline, MAXSUB,
             "POST %s HTTP/1.0\r\n"
             "Host: %s\r\n"
             "Content-type: text/json charset=utf-8\r\n"
             "Content-length: %d\r\n\r\n"
             "%s", page, host, strlen(poststr), poststr);
    
    write(sockfd,sendline,strlen(sendline));
    while((n = read(sockfd, recvline, MAXLINE)) > 0){
        recvline[n]='\0';
        printf("%s",recvline);
    }
    return n;
}
IplConvKernel *element = cvCreateStructuringElementEx(11, 11, 6, 6, CV_SHAPE_RECT, NULL);
Mat fgMaskMOG2;
Ptr<BackgroundSubtractor> pMOG2;

int main(int argc, char** argv) {
    
    int sockfd;
    struct sockaddr_in servaddr;
    char str[50];
    char poststr[MAXLINE+1];
    char **pptr;
    const char *hname= "protocol.mybluemix.net";
    const char *page = "/receive";
    struct hostent *hptr;
    while((hptr = gethostbyname(hname))==NULL){
        printf("Cannot Find Wireless AP\n");
        sleep(10);
    }
    
    double count = 0;
    CvCapture *capture = cvCaptureFromFile(NAME);
    //CvCapture *capture = 0;
    //capture = cvCaptureFromCAM(0);
    if (!capture)    {
        cout << "The video file was not found."<<endl;
        return 0;
    }
    srand(15);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT,240);
    
    int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    
    CvSize frameSize = cvSize(320, 240);
    
    IplImage *origin = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
    IplImage *frame = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
    IplImage *Mask = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
    
    cvZero(Mask);
    
    CvSeq* contours = 0;
    CvSeq* result = 0;
    CvMemStorage* storage;
    CvFont font;

    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 2, CV_AA);
    
    //pMOG2 = createBackgroundSubtractorMOG2(500, 8.0, false);
    pMOG2 = createBackgroundSubtractorKNN(500, 400.0, true);
    char buffer[512];
    CvPoint tmp_polygon[1][4];
    int detectR, detectG, detectB;
    bool tmp1, tmp2, tmp3, tmp4;
    int inCount = 0;
    int outCount = 0;
    int state;
    int fd[2];
    
    time_t timer;
    struct tm *t;
    timer = time(NULL);
    t = localtime(&timer);
    int changeTime = t->tm_mday;
    state = pipe(fd);
    if (state == -1) {
        return -1;
    }
    int buffer2[2];
    
    int countFlag = -1;
    pid_t pid;
    
    pid = fork();
    vector<CvPoint*> pt;
    ////////////////////////////////////////////////////////////////////////////////////////
    printf("hostname: %s\n",hptr->h_name);
    sleep(5);
    if(hptr->h_addrtype == AF_INET
       &&(pptr = hptr->h_addr_list) !=NULL){
        printf("address: %s\n",inet_ntop(hptr->h_addrtype, *pptr, str, sizeof(str)));
    }
    else{
        fprintf(stderr,"Error call inet_ntop \n");
    }
    int sendCount=0;
    
    switch(pid) {
        case -1 : {
            return -1;
        }
        case 0 : {
            while(1) {
                read(fd[0], buffer2, sizeof(buffer2));
                inCount = buffer2[0];
                outCount = buffer2[1];
                snprintf(poststr,MAXSUB,"{\"name\":\"socket\",\"inCount\":%d,\"outCount\":%d}", inCount, outCount);
                sockfd = socket(AF_INET,SOCK_STREAM,0);
                bzero(&servaddr,sizeof(servaddr));
                servaddr.sin_family= AF_INET;
                servaddr.sin_port= htons(80);
                inet_pton(AF_INET, str, &servaddr.sin_addr);
                if(-1==connect(sockfd,(struct sockaddr*)&servaddr,sizeof(servaddr))){
                    printf( "connect fail ~ \n");
                    exit( 1);
                }
                process_http(sockfd,hname,page,poststr);
                close(sockfd);
            }
        }
        default : {
            while (1)    {
                timer = time(NULL);
                t = localtime(&timer);
                count++;
                
                if(changeTime != t->tm_mday){
                    inCount=0;
                    outCount=0;
                    changeTime = t->tm_mday;
                }
                
                for (itor = ROIs.begin(); itor != ROIs.end(); itor++){
                    if ((*itor).lifetime < 0){
                        cvRectangle(Mask, cvPoint((*itor).leftTop_x, (*itor).leftTop_y), cvPoint((*itor).rightBot_x, (*itor).rightBot_y), CV_RGB(0, 0, 0), CV_FILLED, 8);
                        ROIs.erase(itor);
                        break;
                    }
                    (*itor).lifetime--;
                }
                
                countFlag++;
                countFlag %= 7;
                origin =cvQueryFrame(capture);
                cvSaveImage("/home/pi/stream/stream.jpg", origin);
                cvResize(origin, frame, CV_INTER_CUBIC);
                Mat mframe;
                mframe = cvarrToMat(frame);
                pMOG2->apply(mframe, fgMaskMOG2);
                mframe.release();
                IplImage *bkgImage = new IplImage(fgMaskMOG2);
                //--------------------Filter------------------------//
                //cvErode(bkgImage, bkgImage);
                cvSmooth(bkgImage, bkgImage, CV_MEDIAN, 7, 7);
                //cvSmooth(bkgImage , bkgImage , CV_GAUSSIAN, 5, 5);
                //cvCanny(bkgImage, bkgImage, 20, 220, 3);
                //cvErode(bkgImage, bkgImage);
                //cvDilate(bkgImage, bkgImage, element, 1);
                //cvDilate(bkgImage, bkgImage);
                //cvErode(bkgImage, bkgImage, element, 1);
                //cvErode(bkgImage, bkgImage);
                //cvDilate(bkgImage, bkgImage, element, 1);
                //cvMorphologyEx(bkgImage, bkgImage, NULL, element, CV_MOP_OPEN, 1);
                //cvMorphologyEx(bkgImage, bkgImage, NULL, NULL, CV_MOP_CLOSE, 1);
                //---------------------------------------------------//
                cvThreshold(bkgImage, bkgImage, 200, 255, CV_THRESH_BINARY);
                threshold(fgMaskMOG2,fgMaskMOG2,200,255,CV_THRESH_BINARY);
                cvLine(frame, cvPoint(STANDARD1_X, STANDARD1_Y), cvPoint(STANDARD2_X, STANDARD2_Y), CV_RGB(0, 255, 0), 2, 0);
                
                vector<vector<Point> > contourss;
                vector<Vec4i> hierarchy;
                
                findContours(fgMaskMOG2, contourss, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
                
                medianBlur(fgMaskMOG2, fgMaskMOG2, 9);
                
                /// Get the moments
                vector<Moments> mu(contourss.size());
                for (int i = 0; i < contourss.size(); i++)
                {
                    mu[i] = moments(contourss[i], false);
                }
                ///  Get the mass centers:
                vector<Point2f> mc(contourss.size());
                for (int i = 0; i < contourss.size(); i++)
                {
                    mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
                }
                /// Draw contours
                Mat drawing = Mat::zeros(fgMaskMOG2.size(), CV_8UC3);
                for (int i = 0; i< contourss.size(); i++)
                {
                    if (contourArea(contourss[i]) > AREA){
                        Scalar color = Scalar(255, 255,255);
                        drawContours(drawing, contourss, i, color, 2, 8, hierarchy, 0, Point());
                        circle(drawing, mc[i], 4, Scalar(0,255,0), -1, 8, 0);
                    }
                }
                namedWindow("Contours", CV_WINDOW_AUTOSIZE);
                imshow("Contours", drawing);
                drawing.release();
                tmp_polygon[0][0] = cvPoint(160, 240);tmp_polygon[0][1] = cvPoint(160, 0);
                sprintf(buffer, "IN : %d, OUT : %d", outCount, inCount);
                cvPutText(frame, buffer, cvPoint(50, 220), &font, cvScalar(0, 255, 0));
                
                if (drawRect){
                    cvRectangle(bkgImage, cvPoint(rectLeftTop_x, rectLeftTop_y), cvPoint(rectRightBot_x, rectRightBot_y), CV_RGB(255, 255, 255), CV_FILLED, 0);
                }
                storage = cvCreateMemStorage(0);
                cvNamedWindow("Original Video");
                delete []bkgImage;
                for(int i=0;i< contourss.size();i++){
                    if (contourArea(contourss[i])>AREA){
                        int posX = mc[i].x;
                        int posY = mc[i].y;
                        detectR = CV_GET_R(Mask, posX, posY);
                        detectG = CV_GET_G(Mask, posX, posY);
                        detectB = CV_GET_B(Mask, posX, posY);
                        
                        MASKSIZE = sqrt(contourArea(contourss[i])) / 8 * 5;
                        MASKSIZE = MASKSIZE > 40 ? 40 : MASKSIZE;
                        
                        cvCircle(frame, cvPoint(posX, posY), 5, CV_RGB(rand() % 255, rand() % 255, rand() % 255), -1);
                        cvRectangle(frame, cvPoint(posX - MASKSIZE, posY - MASKSIZE), cvPoint(posX + MASKSIZE, posY + MASKSIZE), CV_RGB(rand() % 255, rand() % 255, rand() % 255), 1, 8);
                        if (detectR == 255 && detectG == 0 && detectB == 255){
                            
                            for (itor = ROIs.begin(); itor != ROIs.end(); itor++){
                                if ((*itor).rightBot_x > posX && posX > (*itor).leftTop_x && (*itor).rightBot_y > posY && posY > (*itor).leftTop_y){
                                    cvRectangle(Mask, cvPoint((*itor).leftTop_x, (*itor).leftTop_y), cvPoint((*itor).rightBot_x, (*itor).rightBot_y), CV_RGB(0, 0, 0), CV_FILLED, 8);
                                    cvRectangle(Mask, cvPoint(posX - MASKSIZE, posY - MASKSIZE), cvPoint(posX + MASKSIZE, posY + MASKSIZE), CV_RGB(255, 0, 255), CV_FILLED, 8);
                                    (*itor).leftTop_x = posX - MASKSIZE;
                                    (*itor).leftTop_y = posY - MASKSIZE;
                                    (*itor).rightBot_x = posX + MASKSIZE;
                                    (*itor).rightBot_y = posY + MASKSIZE;
                                    
                                    if (countFlag == 0){
                                        tmp1 = isCounterClock((*itor).past.x, (*itor).past.y, posX, posY, STANDARD2_X,STANDARD2_Y);
                                        tmp2 = isCounterClock((*itor).past.x, (*itor).past.y, posX, posY, STANDARD1_X,STANDARD1_Y);
                                        tmp3 = isCounterClock(STANDARD2_X,STANDARD2_Y,STANDARD1_X,STANDARD1_Y, (*itor).past.x, (*itor).past.y);
                                        tmp4 = isCounterClock(STANDARD2_X, STANDARD2_Y,STANDARD1_X,STANDARD1_Y, posX, posY);
                                        
                                        if (RighttoLeft(tmp1, tmp2, tmp3, tmp4)){
                                            inCount++;
                                            sendCount++;
                                            cout << "incount = " << inCount << endl;
                                        }
                                        if (LefttoRight(tmp1, tmp2, tmp3, tmp4)){
                                            outCount++;
                                            sendCount++;
                                            cout << "outcount = " << outCount << endl;
                                        }
                                        /*
                                        if (UptoDown(tmp1, tmp2, tmp3, tmp4)){
                                            inCount++;
                                            sendCount++;
                                            cout << "incount = " << inCount << endl;
                                        }
                                        if (DowntoUp(tmp1, tmp2, tmp3, tmp4)){
                                            outCount++;
                                            sendCount++;
                                            cout << "outcount = " << outCount << endl;
                                        }
                                        */
                                        (*itor).past = cvPoint(posX, posY);
                                    }
                                    
                                    sprintf(buffer, "%d", (int)contourArea(contourss[i]));
                                    cvPutText(frame, buffer, cvPoint(posX, posY), &font, CV_RGB(0, 255, 255));
                                    
                                    (*itor).lifetime++;
                                    break;
                                }
                            }
                        }
                        else{
                            cvRectangle(Mask, cvPoint(posX - MASKSIZE, posY - MASKSIZE), cvPoint(posX + MASKSIZE, posY + MASKSIZE), CV_RGB(255, 0, 255), CV_FILLED, 8);
                            ROI tmp;
                            tmp.id = ROI_num++;
                            tmp.leftTop_x = posX - MASKSIZE;
                            tmp.leftTop_y = posY - MASKSIZE;
                            tmp.rightBot_x = posX + MASKSIZE;
                            tmp.rightBot_y = posY + MASKSIZE;
                            tmp.past = cvPoint(posX, posY);
                            tmp.lifetime = 10;
                            ROIs.push_back(tmp);
                        }
                    }

                    if (sendCount ==10) {
                        if (outCount > inCount) outCount = inCount;
                        int inoutCount[2] = {0, 0};
                        inoutCount[0] = inCount;
                        inoutCount[1] = outCount;
                        write(fd[1], inoutCount, sizeof(inoutCount));
                        sendCount = 0;
                    }
                }
                cvReleaseMemStorage(&storage);
                cvShowImage("Original Video", frame);
                if (cvWaitKey(1) == 'p')
                    break;
            }
            cvReleaseCapture(&capture);
            cvReleaseImage(&Mask);
            cvReleaseImage(&frame);
            cvClearSeq(contours);
            cvClearSeq(result);
            cvDestroyAllWindows();
        }
    }
}