
#include"svm.h"
SVMModel::SVMModel() { svm = cv::ml::SVM::create(); }
/*ֱ�Ӷ�ȡtxt����*/
void SVMModel::OpenImagesFromTxt(vector<vector<double>>& rawData, Mat& Data, Mat& Labels) {
	int Nb_Files = rawData.size();
	cout << "hello" << endl;
	int TotalSz_Zone = rawData[0].size() - 1;
	//vector<vector<float>> Data;// (Nb_Files, vector<Mat>(TotalSz_Zone, Mat(1, 1, CV_32FC1)));
	//Mat Labels = Mat::zeros(Nb_Files, 1, CV_32S);
	//vector<float> maxValues(TotalSz_Zone); // ��C++��vector�����У�Ԫ�ص����������Եģ����û���ϸ���к��еĸ���
	Data = Mat::zeros(Nb_Files, TotalSz_Zone, CV_32FC1);
	Labels = Mat::zeros(Nb_Files, 1, CV_32SC1);
	vector<double> minValues = rawData[0];
	vector<double> maxValues = rawData[0];
	for (int i = 0; i < Nb_Files; ++i) {
		for (int ii = 0; ii < TotalSz_Zone; ++ii) {
			//float value = rawData[i][ii];
			////����ÿ����������Сֵ�����ֵ
			//if (value < minValues[ii]) {
			//minValues[ii] = value;
			//}
			//else {
			//maxValues[ii] = value;
			//}
			////������ֵ��һ���� [0, 1] ��Χ
			//float normalizedvalue = (value - minValues[ii]) / (maxValues[ii] - minValues[ii]);
			//Data.at<float>(i, ii) = normalizedvalue;
			Data.at<float>(i, ii) = static_cast<float>(rawData[i][ii]);
			/*at<float>(0, 0)���������ڷ���Mat������Ԫ�صķ��������У�<float>��ʾ��ϣ�����ʵ�Ԫ��������float���͡�(0, 0)������Ԫ�ص�����������һ����ͨ����Mat����������������������(0, 0)��ʾ��һ�е�һ�е�Ԫ��*/
		}
		Labels.at<int>(i, 0) = static_cast<int>(rawData[i].back()); // �������ǩֵת��Ϊ����
	}
	Labels.convertTo(Labels, CV_32SC1);
	Data.convertTo(Data, CV_32FC1);
	cout << "Data Matrix:\n" << Data << endl;
}
//�� trainingData �� labels ������Ϊ���ô��ݣ��Ա��ں����ڲ��޸�����
void SVMModel::SVM_Train_Custom(Mat& Data, Mat& Labels)
{
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);//C֧����������
	svm->setKernel(cv::ml::SVM::POLY);//����ʽ�˺�
	svm->setDegree(2);//���ζ���ʽ
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 1e-9));
	// Train the SVM
	cout << "training SVM..." << endl;
	svm->train(Data, cv::ml::ROW_SAMPLE, Labels);//����ָ����rowһ��һ��������ÿ��һ������
												 //svm->save("K:\\VS2015code\\hsv_glcm_svm\\svm\\ConsoleApplication1\\modell.xml");	
	cv::FileStorage fs("svm_model.xml", cv::FileStorage::WRITE);
	if (fs.isOpened()) {
		svm->write(fs);
		fs.release();
		cout << "Model saved successfully!" << endl;
	}
	else {
		cout << "Failed to save the model!" << endl;
	}
}
void SVMModel::OpenImages(vector<vector<double>>& rawData, Mat& Data) {
	int Nb_Files = rawData.size();
	int TotalSz_Zone = rawData[0].size();
	Data = Mat::zeros(Nb_Files, TotalSz_Zone, CV_32FC1);
	for (int i = 0; i < Nb_Files; ++i) {
		for (int ii = 0; ii < TotalSz_Zone; ++ii) {
			Data.at<float>(i, ii) = static_cast<float>(rawData[i][ii]);
		}
	}
	Data.convertTo(Data, CV_32FC1);
	cout << "Data Matrix:\n" << Data << endl;
}
bool SVMModel::testSVM(Mat& Data) {
	cv::Mat predictionResults;
	svm->predict(Data, predictionResults);
	//int predictedLabel = cvRound(predictionResults.at<float>(0, 0)); 
	int predictedLabel = static_cast<int>(predictionResults.at<float>(0, 0)); // ��ȡԤ��������ĵ�һ�е�һ��Ԫ�ص�ֵ,��ȡԤ������������ʽ=label
	cout << "Predicted Label: " << predictedLabel << endl; // ���Ԥ���ǩ
	bool results = false;
	for (int i = 0; i < predictionResults.rows; ++i) {
		if (predictedLabel ==1) {
			results = true;
		}
	}
	return results;
}//�������ֻ���ж�һ��
