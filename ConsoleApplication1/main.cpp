#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <windows.h>
#include <iterator>
#include <ctime>

#include"hsv.h"
#include"glcm.h"
#include"svm.h"
using namespace std;
using namespace cv;

bool SVMModel::DetectFire(const Mat& image)
{
	
	bool detected = false;
	VecGLCM VecGlcm(256);
	for (int i = 0; i < 256; i++)
	{
		VecGlcm[i].resize(256);
	}
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			VecGlcm[i][j] = 0;
		}
	}
	Mat Data, Data0;
	vector<vector<double>> modifiedMatrix3 = ComputeImageFeatures(image, VecGlcm, 256);
	InitVecGLCM(VecGlcm, 256);
	OpenImages(modifiedMatrix3, Data0);
	bool svmPrediction = testSVM(Data0);
	if (svmPrediction)
	{
		detected = true; // 检测到火灾
	}
	return detected;
}

int main()
{   
	int maxGray = 255;
	int  size = maxGray + 1;
	VecGLCM VecGlcm(size, vector<uchar >(size));
	vector<vector<double>> modifiedMatrix1 = ValueWrite(binaryImagesFileRead("K:\\VS2015code\\new\\SVM_master\\testimage\\resize-fire500\\*.jpg", VecGlcm, 256), 1);     //写操作
	InitVecGLCM(VecGlcm, 256);
	vector<vector<double>> modifiedMatrix2 = ValueWrite(binaryImagesFileRead("K:\\VS2015code\\new\\SVM_master\\testimage\\resize-non500\\*.jpg", VecGlcm, 256), 2);     //写操作
	InitVecGLCM(VecGlcm, 256);																																								   // 合并两个矩阵
	vector<vector<double>> combinedMatrix;
	combinedMatrix.insert(combinedMatrix.end(), modifiedMatrix1.begin(), modifiedMatrix1.end());
	combinedMatrix.insert(combinedMatrix.end(), modifiedMatrix2.begin(), modifiedMatrix2.end());
	Mat Data, Data0;
	Mat Labels;
	SVMModel obj; // 创建类的对象
	obj.OpenImagesFromTxt(combinedMatrix, Data, Labels); // 调用成员函数
	obj.SVM_Train_Custom(Data, Labels);

	// 创建并打开输出文件流
	ofstream outputFile("output.txt");
	if (!outputFile.is_open())
	{
		cout << "Error opening output file." << endl;
		return 1;
	}
	// 将标准输出（cout）定向到输出文件流
	streambuf* originalCoutBuffer = cout.rdbuf(); // 保存原始的cout缓冲区
	cout.rdbuf(outputFile.rdbuf()); // 将cout的输出重定向到文件流
	vector<Mat> BinaryImages = ImageFileRead("K:\\VS2015code\\new\\SVM_master\\testimage\\hsv_glcm_sv\\*.jpg"); 
	for (size_t i = 0; i < BinaryImages.size(); ++i)
	{
		if (countNonZero(BinaryImages[i]) > 0)
		{
			bool fireDetected = obj.DetectFire(BinaryImages[i]);
			if (fireDetected)
			{
				cout << "image " << i << ": fire detected" << endl;
				// 保存或处理检测到火灾的图像
			}
			else
			{
				cout << "image " << i << ": no fire detected" << endl;
			}
		}

	}
	// 恢复标准输出
	cout.rdbuf(originalCoutBuffer);

	// 关闭输出文件流
	outputFile.close();
	return 0;
}




