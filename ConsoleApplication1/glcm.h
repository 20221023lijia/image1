#pragma once
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<vector>
#include<algorithm>
#include<iterator>
#include<cmath>
#include<fstream>
using namespace std;
using namespace cv;

#include"hsv.h"
typedef vector<vector<uchar>> VecGLCM;

void VecGLCMCount0(VecGLCM& GM_VecGLCM, cv::Mat PriImage, int nCols, int nRows);  //0度共生矩阵

double ComputeEntropy(VecGLCM& GM_VecGLCM, int size);       //计算熵值
double ComputeEnergy(VecGLCM& GM_VecGLCM, int size);        //计算能量
double ComputeContrast(VecGLCM& GM_VecGLCM, int size);      //计算对比度 
double ComputeUniformity(VecGLCM& GM_VecGLCM, int size);    //计算均匀度
double ComputeCorrelation(VecGLCM& GM_VecGLCM, int size);   //计算相关性

vector<vector<double>> binaryImagesFileRead(string PatternJPG, VecGLCM& GM_VecGLCM, int size);     //批量读取图片操作:in、out、大小
vector<vector<double>> ValueWrite(vector<vector<double>> Matrix, int n);
vector<vector<double>>  ComputeImageFeatures(const Mat& image, VecGLCM& GM_VecGLCM, int size);
void InitVecGLCM(VecGLCM& GM_VecGLCM, int size);     //初始化VecGLCM：读入图片、大小：
