#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>
#include <windows.h>
#include <iterator>
#include <ctime>
#include <iomanip>
#include <vector>
#include <limits>


using namespace cv;
using namespace std;
using namespace cv::ml;

class SVMModel {
public:
	SVMModel();//<类名>(参数表);---构造函数
	//vector<vector<Mat>> Data;
	//Mat Labels;
	//vector<vector<double>> rawData;
	void OpenImagesFromTxt(vector<vector<double>>& rawData, Mat& Data, Mat& Labels);
	void SVM_Train_Custom(Mat& Data, Mat& Labels);
	void OpenImages(vector<vector<double>>& rawData, Mat& Data);
	bool testSVM(Mat& Data);
	bool DetectFire(const Mat& image);
private://作为成员传递
	cv::Ptr<cv::ml::SVM> svm;
};
