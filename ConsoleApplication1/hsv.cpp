#include"hsv.h"
/*
*    �ڶ��ַ��������û����HSV���������ֻ���
*    �������ʵ�飺����H:0-60  S:160-255  V: 160-255 Ч��������
*    ��Ҫ���Ӿ�ȷ�Ļ������ö��Ż���ͼƬ�ó�����Ӿ�ȷ�ķ�Χ��
*/
vector<Mat> ImageFileRead(string PatternJPG)
{	//ѡ�񷵻����� vector<vector<double>> ����Ϊ�����ܴ��������ص����ݽṹ��������������ͼ��ļ��ϣ�ÿ��ͼ����һ������������ͬʱ���ִ�����ͼ���ÿ��ͼ��������������ԡ�
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
		// ����ͳһ��ͼ���С,����ͼ��ķֱ��ʣ�����64����X64����
		//int targetWidth = 64;
		//int targetHeight = 64;
		// ��ͼ����С0.6����ָ�������������ӡ�
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
					// ��˹�˲�ȥ��
					cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
					// ֱ��ͼ���⻯
					cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
					cv::equalizeHist(gray, gray);
					//cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
				}
			}
		}
		BinaryImages.push_back(gray);//������ͼ��
	}
	return BinaryImages;
}
