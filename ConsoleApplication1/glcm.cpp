#include"glcm.h"
//==============================================================================
// 函数名称: VecGLCMCount0
// 参数说明: PriImage为初始的图片，nCols为列，nRows为行数
// 函数功能: 进行0度方向的共生矩阵求解 
//==============================================================================
void VecGLCMCount0(VecGLCM& GM_VecGLCM, cv::Mat PriImage, int nCols, int nRows)
{
	int VecGLCM_Col;
	int VecGLCM_Row;
	double maxGrayLevel = 0;
	uchar* p;//uchar 类型+指针 p，用于指向图像矩阵的每一行的首地址
	for (int i = 0; i < nRows; i++)
	{
		p = PriImage.ptr<uchar>(i);//获取每行首地址
		for (int j = 0; j < nCols - 1; ++j)
		{
			VecGLCM_Col = p[j];
			VecGLCM_Row = p[j + 1];
			GM_VecGLCM[VecGLCM_Col][VecGLCM_Row]++;
		}
	}
}

//==============================================================================
// 函数名称: ComputeEntropy
// 参数说明: GM_VecGLCM为共生矩阵，size为矩阵的大小（size X size）
// 函数功能: 求共生矩阵的熵
//==============================================================================
double ComputeEntropy(VecGLCM& GM_VecGLCM, int size)
{
	double sum = 0;
	vector<vector<uchar>>::iterator IE;
	vector<uchar>::iterator it;
	for (IE = GM_VecGLCM.begin(); IE < GM_VecGLCM.end(); IE++)
		//这是一个外部循环，遍历 GM_VecGLCM 的每一行。
	{
		for (it = (*IE).begin(); it < (*IE).end(); it++)
			//这是一个嵌套的内部循环，遍历当前行中的每个元素（也就是灰度共生矩阵中的值）。
		{
			if ((*it) != 0)  sum += -(*it) * log(*it);
			//cout << *it << " ";
		}
	}
	return sum;
}
//==============================================================================
// 函数名称: ComputeEnergy
// 参数说明: GM_VecGLCM为共生矩阵，size为矩阵的大小（size X size）
// 函数功能: 求共生矩阵的能量
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
// 函数名称: ComputeContrast
// 参数说明: GM_VecGLCM为共生矩阵，size为矩阵的大小（size X size）
// 函数功能: 求共生矩阵的对比度
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
// 函数名称: ComputeUniformity
// 参数说明: GM_VecGLCM为共生矩阵，size为矩阵的大小（size X size）
// 函数功能: 求共生矩阵的均匀度
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
// 函数名称: ComputeCorrelation
// 参数说明: GM_VecGLCM为共生矩阵，size为矩阵的大小（size X size）
// 函数功能: 求共生矩阵的相关性
//==============================================================================
double ComputeCorrelation(VecGLCM& GM_VecGLCM, int size)
{
	double Ui = 0; double Uj = 0;
	double Si = 0; double Sj = 0;
	double Si_Square = 0;    //Si的平方为Si_Square
	double Sj_Square = 0;    //Sj的平方为Sj_Square
	double COR = 0;  //相关性
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			Ui += (i + 1) * GM_VecGLCM[i][j];
			Uj += (j + 1) * GM_VecGLCM[i][j];
		}
	}
	//计算了每个灰度级在横向和纵向的加权平均值
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			Si_Square += GM_VecGLCM[i][j] * (i + 1 - Ui) * (i + 1 - Ui);
			Sj_Square += GM_VecGLCM[i][j] * (j + 1 - Uj) * (j + 1 - Uj);
		}
	}
	//计算了每个灰度级在横向和纵向上与其平均值之差的平方的加权和s
	Si = sqrt(Si_Square);
	Sj = sqrt(Sj_Square);
	//计算了横向和纵向的标准差
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			COR += (((i + 1) * (j + 1)) * GM_VecGLCM[i][j] - Ui * Uj) / (Si*Sj);
		}
	}
	//考虑了每个 (i, j) 处的灰度级值以及对应的 GLCM 值，然后计算了加权值之间的差异，并除以标准差的乘积。
	return COR;
}

//==============================================================================
// 函数名称: ImageFileRead
// 参数说明: PatternJPG为文件名，形如"E:\\图片\\*.jpg",GM_VecGLCM为共生矩阵，size为矩阵的大小
// 函数功能: 批量处理图片文件，并计算共生矩阵的熵、能量、对比度和均匀度返回一个num X 4的矩阵（num为图片的总数）
//==============================================================================
//void ImageFileRead(string PatternJPG, VecGLCM& GM_VecGLCM, int size)
vector<vector<double>> binaryImagesFileRead(string PatternJPG, VecGLCM& GM_VecGLCM, int size)
{// size和计算熵值有关
	vector<String>ImageFile;
	glob(PatternJPG, ImageFile);    //将PatternJPG路径下的所用文件名存进ImageFile中
	if (ImageFile.size() == 0) {
		cout << "NO image file[jpg]" << endl;
	}
	//二维矩阵创建，ValueVec中保存各个图片的共生矩阵的四个参数信息：图片、距离、角度、灰度等级
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
		cout << ImageFile[frame] << "读取成功！" << endl;
		//cv::resize(image, image, cv::Size(targetWidth, targetHeight));
		// 高斯滤波去噪
		cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
		cv::equalizeHist(image, image);
		//imshow("Image", image);
		//waitKey(0);
		int nRows = image.rows;
		int nCols = image.cols;

		VecGLCMCount0(GM_VecGLCM, image, nCols, nRows);
		ValueVec[frame][0] = ComputeEntropy(GM_VecGLCM, size);       //计算熵值
		ValueVec[frame][1] = ComputeEnergy(GM_VecGLCM, size);       //计算能量
		ValueVec[frame][2] = ComputeContrast(GM_VecGLCM, size);       //计算对比度
		ValueVec[frame][3] = ComputeUniformity(GM_VecGLCM, size);     //计算均匀度
		ValueVec[frame][4] = ComputeCorrelation(GM_VecGLCM, size);
		InitVecGLCM(GM_VecGLCM, size);  //重新将GM_VecGLCM变为全0的矩阵
	}
	return ValueVec;
}

vector<vector<double>> ComputeImageFeatures(const Mat& image, VecGLCM& GM_VecGLCM, int size)
	{
		vector<vector<double>> ValueVec;

		int nRows = image.rows;
		int nCols = image.cols;
		VecGLCMCount0(GM_VecGLCM, image, nCols, nRows);

		// 计算各种特征并将结果添加到ValueVec
		vector<double> features;
		features.push_back(ComputeEntropy(GM_VecGLCM, size));
		features.push_back(ComputeEnergy(GM_VecGLCM, size));
		features.push_back(ComputeContrast(GM_VecGLCM, size));
		features.push_back(ComputeUniformity(GM_VecGLCM, size));
		features.push_back(ComputeCorrelation(GM_VecGLCM, size));
		ValueVec.push_back(features);
		InitVecGLCM(GM_VecGLCM, size); // 重新将GM_VecGLCM变为全0的矩阵

		return ValueVec;
	}

//==============================================================================
// 函数名称: ValueWrite
// 参数说明: Matrix为矩阵名，FileName为需要写如的文件名，如"B2F.txt"
// 函数功能: 文件写操作，将Matrix中的内容写入文件FileName中
//==============================================================================
vector<vector<double>> ValueWrite(vector<vector<double>> Matrix,int n)
{
	vector<vector<double>> modifiedMatrix;
	for (int i = 0; i < Matrix.size(); i++)
	{
		vector<double> newRow;
		for (int j = 0; j < Matrix[i].size(); j++)
		{
			newRow.push_back(Matrix[i][j] ); //并添加制表符 \t 作为分隔符

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
			newRow.push_back(Matrix[i][j]); //并添加制表符 \t 作为分隔符

		}
		modifiedMatrix.push_back(newRow);
	}
	return modifiedMatrix;
}