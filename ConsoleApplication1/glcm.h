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

void VecGLCMCount0(VecGLCM& GM_VecGLCM, cv::Mat PriImage, int nCols, int nRows);  //0�ȹ�������

double ComputeEntropy(VecGLCM& GM_VecGLCM, int size);       //������ֵ
double ComputeEnergy(VecGLCM& GM_VecGLCM, int size);        //��������
double ComputeContrast(VecGLCM& GM_VecGLCM, int size);      //����Աȶ� 
double ComputeUniformity(VecGLCM& GM_VecGLCM, int size);    //������ȶ�
double ComputeCorrelation(VecGLCM& GM_VecGLCM, int size);   //���������

vector<vector<double>> binaryImagesFileRead(string PatternJPG, VecGLCM& GM_VecGLCM, int size);     //������ȡͼƬ����:in��out����С
vector<vector<double>> ValueWrite(vector<vector<double>> Matrix, int n);
vector<vector<double>>  ComputeImageFeatures(const Mat& image, VecGLCM& GM_VecGLCM, int size);
void InitVecGLCM(VecGLCM& GM_VecGLCM, int size);     //��ʼ��VecGLCM������ͼƬ����С��
