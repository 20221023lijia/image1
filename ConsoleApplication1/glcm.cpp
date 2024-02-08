#include"glcm.h"
//==============================================================================
// ��������: VecGLCMCount0
// ����˵��: PriImageΪ��ʼ��ͼƬ��nColsΪ�У�nRowsΪ����
// ��������: ����0�ȷ���Ĺ���������� 
//==============================================================================
void VecGLCMCount0(VecGLCM& GM_VecGLCM, cv::Mat PriImage, int nCols, int nRows)
{
	int VecGLCM_Col;
	int VecGLCM_Row;
	double maxGrayLevel = 0;
	uchar* p;//uchar ����+ָ�� p������ָ��ͼ������ÿһ�е��׵�ַ
	for (int i = 0; i < nRows; i++)
	{
		p = PriImage.ptr<uchar>(i);//��ȡÿ���׵�ַ
		for (int j = 0; j < nCols - 1; ++j)
		{
			VecGLCM_Col = p[j];
			VecGLCM_Row = p[j + 1];
			GM_VecGLCM[VecGLCM_Col][VecGLCM_Row]++;
		}
	}
}

//==============================================================================
// ��������: ComputeEntropy
// ����˵��: GM_VecGLCMΪ��������sizeΪ����Ĵ�С��size X size��
// ��������: �����������
//==============================================================================
double ComputeEntropy(VecGLCM& GM_VecGLCM, int size)
{
	double sum = 0;
	vector<vector<uchar>>::iterator IE;
	vector<uchar>::iterator it;
	for (IE = GM_VecGLCM.begin(); IE < GM_VecGLCM.end(); IE++)
		//����һ���ⲿѭ�������� GM_VecGLCM ��ÿһ�С�
	{
		for (it = (*IE).begin(); it < (*IE).end(); it++)
			//����һ��Ƕ�׵��ڲ�ѭ����������ǰ���е�ÿ��Ԫ�أ�Ҳ���ǻҶȹ��������е�ֵ����
		{
			if ((*it) != 0)  sum += -(*it) * log(*it);
			//cout << *it << " ";
		}
	}
	return sum;
}
//==============================================================================
// ��������: ComputeEnergy
// ����˵��: GM_VecGLCMΪ��������sizeΪ����Ĵ�С��size X size��
// ��������: �������������
//==============================================================================
double ComputeEnergy(VecGLCM& GM_VecGLCM, int size)
{
	double sum = 0;
	vector<vector<uchar>>::iterator IE;
	vector<uchar>::iterator it;
	for (IE = GM_VecGLCM.begin(); IE < GM_VecGLCM.end(); IE++)
	{
		for (it = (*IE).begin(); it < (*IE).end(); it++)
		{
			sum += (*it) ^ 2;
		}
	}
	return sum;
}

//==============================================================================
// ��������: ComputeContrast
// ����˵��: GM_VecGLCMΪ��������sizeΪ����Ĵ�С��size X size��
// ��������: ��������ĶԱȶ�
//==============================================================================
double ComputeContrast(VecGLCM& GM_VecGLCM, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			sum += (i - j) ^ 2 * GM_VecGLCM[i][j];
		}
	}
	return sum;
}

//==============================================================================
// ��������: ComputeUniformity
// ����˵��: GM_VecGLCMΪ��������sizeΪ����Ĵ�С��size X size��
// ��������: ��������ľ��ȶ�
//==============================================================================
double ComputeUniformity(VecGLCM& GM_VecGLCM, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			sum += GM_VecGLCM[i][j] / (1 + abs(i - j));
		}
	}
	return sum;
}

//==============================================================================
// ��������: ComputeCorrelation
// ����˵��: GM_VecGLCMΪ��������sizeΪ����Ĵ�С��size X size��
// ��������: ��������������
//==============================================================================
double ComputeCorrelation(VecGLCM& GM_VecGLCM, int size)
{
	double Ui = 0; double Uj = 0;
	double Si = 0; double Sj = 0;
	double Si_Square = 0;    //Si��ƽ��ΪSi_Square
	double Sj_Square = 0;    //Sj��ƽ��ΪSj_Square
	double COR = 0;  //�����
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			Ui += (i + 1) * GM_VecGLCM[i][j];
			Uj += (j + 1) * GM_VecGLCM[i][j];
		}
	}
	//������ÿ���Ҷȼ��ں��������ļ�Ȩƽ��ֵ
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			Si_Square += GM_VecGLCM[i][j] * (i + 1 - Ui) * (i + 1 - Ui);
			Sj_Square += GM_VecGLCM[i][j] * (j + 1 - Uj) * (j + 1 - Uj);
		}
	}
	//������ÿ���Ҷȼ��ں��������������ƽ��ֵ֮���ƽ���ļ�Ȩ��s
	Si = sqrt(Si_Square);
	Sj = sqrt(Sj_Square);
	//�����˺��������ı�׼��
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			COR += (((i + 1) * (j + 1)) * GM_VecGLCM[i][j] - Ui * Uj) / (Si*Sj);
		}
	}
	//������ÿ�� (i, j) ���ĻҶȼ�ֵ�Լ���Ӧ�� GLCM ֵ��Ȼ������˼�Ȩֵ֮��Ĳ��죬�����Ա�׼��ĳ˻���
	return COR;
}

//==============================================================================
// ��������: ImageFileRead
// ����˵��: PatternJPGΪ�ļ���������"E:\\ͼƬ\\*.jpg",GM_VecGLCMΪ��������sizeΪ����Ĵ�С
// ��������: ��������ͼƬ�ļ��������㹲��������ء��������ԱȶȺ;��ȶȷ���һ��num X 4�ľ���numΪͼƬ��������
//==============================================================================
//void ImageFileRead(string PatternJPG, VecGLCM& GM_VecGLCM, int size)
vector<vector<double>> binaryImagesFileRead(string PatternJPG, VecGLCM& GM_VecGLCM, int size)
{// size�ͼ�����ֵ�й�
	vector<String>ImageFile;
	glob(PatternJPG, ImageFile);    //��PatternJPG·���µ������ļ������ImageFile��
	if (ImageFile.size() == 0) {
		cout << "NO image file[jpg]" << endl;
	}
	//��ά���󴴽���ValueVec�б������ͼƬ�Ĺ���������ĸ�������Ϣ��ͼƬ�����롢�Ƕȡ��Ҷȵȼ�
	vector<vector<double>> ValueVec;
	int num = ImageFile.size();//
	ValueVec.resize(num);
	for (int i = 0; i < num; i++)
	{
		ValueVec[i].resize(5);
	}
	int targetWidth = 64;
	int targetHeight = 64;
	for (unsigned int frame = 0; frame < ImageFile.size(); ++frame)
	{
		Mat image = cv::imread(ImageFile[frame], IMREAD_GRAYSCALE);
		cout << ImageFile[frame] << "��ȡ�ɹ���" << endl;
		//cv::resize(image, image, cv::Size(targetWidth, targetHeight));
		// ��˹�˲�ȥ��
		cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
		cv::equalizeHist(image, image);
		//imshow("Image", image);
		//waitKey(0);
		int nRows = image.rows;
		int nCols = image.cols;

		VecGLCMCount0(GM_VecGLCM, image, nCols, nRows);
		ValueVec[frame][0] = ComputeEntropy(GM_VecGLCM, size);       //������ֵ
		ValueVec[frame][1] = ComputeEnergy(GM_VecGLCM, size);       //��������
		ValueVec[frame][2] = ComputeContrast(GM_VecGLCM, size);       //����Աȶ�
		ValueVec[frame][3] = ComputeUniformity(GM_VecGLCM, size);     //������ȶ�
		ValueVec[frame][4] = ComputeCorrelation(GM_VecGLCM, size);
		InitVecGLCM(GM_VecGLCM, size);  //���½�GM_VecGLCM��Ϊȫ0�ľ���
	}
	return ValueVec;
}

vector<vector<double>> ComputeImageFeatures(const Mat& image, VecGLCM& GM_VecGLCM, int size)
	{
		vector<vector<double>> ValueVec;

		int nRows = image.rows;
		int nCols = image.cols;
		VecGLCMCount0(GM_VecGLCM, image, nCols, nRows);

		// ��������������������ӵ�ValueVec
		vector<double> features;
		features.push_back(ComputeEntropy(GM_VecGLCM, size));
		features.push_back(ComputeEnergy(GM_VecGLCM, size));
		features.push_back(ComputeContrast(GM_VecGLCM, size));
		features.push_back(ComputeUniformity(GM_VecGLCM, size));
		features.push_back(ComputeCorrelation(GM_VecGLCM, size));
		ValueVec.push_back(features);
		InitVecGLCM(GM_VecGLCM, size); // ���½�GM_VecGLCM��Ϊȫ0�ľ���

		return ValueVec;
	}

//==============================================================================
// ��������: ValueWrite
// ����˵��: MatrixΪ��������FileNameΪ��Ҫд����ļ�������"B2F.txt"
// ��������: �ļ�д��������Matrix�е�����д���ļ�FileName��
//==============================================================================
vector<vector<double>> ValueWrite(vector<vector<double>> Matrix,int n)
{
	vector<vector<double>> modifiedMatrix;
	for (int i = 0; i < Matrix.size(); i++)
	{
		vector<double> newRow;
		for (int j = 0; j < Matrix[i].size(); j++)
		{
			newRow.push_back(Matrix[i][j] ); //������Ʊ�� \t ��Ϊ�ָ���

		}
		newRow.push_back(n);
		modifiedMatrix.push_back(newRow);
	}
	return modifiedMatrix;
}

void InitVecGLCM(VecGLCM& GM_VecGLCM, int size)
{	for (int i = 0; i < size; i++) 
	{
		for (int j = 0; j < size; j++)
		{
			GM_VecGLCM[i][j] = 0;
		}
	}
}
vector<vector<double>> ValueWrite1(vector<vector<double>> Matrix)
{
	vector<vector<double>> modifiedMatrix;
	for (int i = 0; i < Matrix.size(); i++)
	{
		vector<double> newRow;
		for (int j = 0; j < Matrix[i].size(); j++)
		{
			newRow.push_back(Matrix[i][j]); //������Ʊ�� \t ��Ϊ�ָ���

		}
		modifiedMatrix.push_back(newRow);
	}
	return modifiedMatrix;
}