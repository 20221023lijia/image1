
#include"svm.h"
SVMModel::SVMModel() { svm = cv::ml::SVM::create(); }
/*直接读取txt数据*/
void SVMModel::OpenImagesFromTxt(vector<vector<double>>& rawData, Mat& Data, Mat& Labels) {
	int Nb_Files = rawData.size();
	cout << "hello" << endl;
	int TotalSz_Zone = rawData[0].size() - 1;
	//vector<vector<float>> Data;// (Nb_Files, vector<Mat>(TotalSz_Zone, Mat(1, 1, CV_32FC1)));
	//Mat Labels = Mat::zeros(Nb_Files, 1, CV_32S);
	//vector<float> maxValues(TotalSz_Zone); // 在C++的vector容器中，元素的排列是线性的，因此没有严格的行和列的概念
	Data = Mat::zeros(Nb_Files, TotalSz_Zone, CV_32FC1);
	Labels = Mat::zeros(Nb_Files, 1, CV_32SC1);
	vector<double> minValues = rawData[0];
	vector<double> maxValues = rawData[0];
	for (int i = 0; i < Nb_Files; ++i) {
		for (int ii = 0; ii < TotalSz_Zone; ++ii) {
			//float value = rawData[i][ii];
			////更新每个特征的最小值和最大值
			//if (value < minValues[ii]) {
			//minValues[ii] = value;
			//}
			//else {
			//maxValues[ii] = value;
			//}
			////将特征值归一化到 [0, 1] 范围
			//float normalizedvalue = (value - minValues[ii]) / (maxValues[ii] - minValues[ii]);
			//Data.at<float>(i, ii) = normalizedvalue;
			Data.at<float>(i, ii) = static_cast<float>(rawData[i][ii]);
			/*at<float>(0, 0)：这是用于访问Mat对象中元素的方法。其中，<float>表示你希望访问的元素类型是float类型。(0, 0)：这是元素的索引。对于一个单通道的Mat（例如在这里你的情况），(0, 0)表示第一行第一列的元素*/
		}
		Labels.at<int>(i, 0) = static_cast<int>(rawData[i].back()); // 将浮点标签值转换为整数
	}
	Labels.convertTo(Labels, CV_32SC1);
	Data.convertTo(Data, CV_32FC1);
	cout << "Data Matrix:\n" << Data << endl;
}
//将 trainingData 和 labels 参数作为引用传递，以便在函数内部修改它们
void SVMModel::SVM_Train_Custom(Mat& Data, Mat& Labels)
{
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);//C支持向量分类
	svm->setKernel(cv::ml::SVM::POLY);//多项式核函
	svm->setDegree(2);//二次多项式
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 1e-9));
	// Train the SVM
	cout << "training SVM..." << endl;
	svm->train(Data, cv::ml::ROW_SAMPLE, Labels);//进行指定：row一行一个样本，每列一个特征
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
	int predictedLabel = static_cast<int>(predictionResults.at<float>(0, 0)); // 获取预测结果矩阵的第一行第一列元素的值,获取预测结果的整数形式=label
	cout << "Predicted Label: " << predictedLabel << endl; // 输出预测标签
	bool results = false;
	for (int i = 0; i < predictionResults.rows; ++i) {
		if (predictedLabel ==1) {
			results = true;
		}
	}
	return results;
}//这边暂且只能判定一张
