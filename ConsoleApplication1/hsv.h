#pragma once
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;
vector<Mat> ImageFileRead(string PatternJPG);	//利用HSV划分阈值进行火焰检测