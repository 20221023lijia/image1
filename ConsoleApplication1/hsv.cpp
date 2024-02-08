#include"hsv.h"
/*
*    第二种方法，利用火焰的HSV分量来区分火焰
*    经过多次实验：采用H:0-60  S:160-255  V: 160-255 效果还可以
*    想要更加精确的话可以用多张火焰图片得出其更加精确的范围。
*/
vector<Mat> ImageFileRead(string PatternJPG)
{	//选择返回类型 vector<vector<double>> 是因为它最能代表函数返回的数据结构。它允许函数返回图像的集合，每个图像都有一组纹理特征，同时保持处理多个图像和每个图像多个特征的灵活性。
	vector<String>ImageFile;
	glob(PatternJPG, ImageFile);
	vector<Mat> BinaryImages;
	if (ImageFile.size() == 0) {
		cout << "NO image file[jpg]" << endl;
	}
	for (unsigned int frame = 0; frame < ImageFile.size(); ++frame)
	{
		Mat image = cv::imread(ImageFile[frame]);
		Mat imageHSV;
		cvtColor(image, imageHSV, CV_BGR2HSV);
		Mat gray;
		// 定义统一的图像大小,就是图像的分辨率，就是64像素X64像素
		//int targetWidth = 64;
		//int targetHeight = 64;
		// 将图像缩小0.6倍，指定单个比例因子。
		double scale_down = 300.0/64;
		Mat scaled_f_down;
		//resize 
		resize(image, scaled_f_down, Size(), scale_down, scale_down, INTER_LINEAR);
		for (int i = 0; i < imageHSV.rows; i++)
		{
			for (int j = 0; j < imageHSV.cols; j++)
			{
				int value_h = imageHSV.at<cv::Vec3b>(i, j)[0];
				int value_s = imageHSV.at<cv::Vec3b>(i, j)[1];
				int value_v = imageHSV.at<cv::Vec3b>(i, j)[2];
				if ((value_h >= 0 && value_h <= 60) && (value_s <= 255 && value_s >= 160) && (value_v <= 255 && value_v >= 160))
				{
					// 高斯滤波去噪
					cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
					// 直方图均衡化
					cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
					cv::equalizeHist(gray, gray);
					//cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
				}
			}
		}
		BinaryImages.push_back(gray);//二进制图像
	}
	return BinaryImages;
}
