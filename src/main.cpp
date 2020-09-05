#include <stdio.h>
#include <fstream>
//#include "cv.h"
//#include "highgui.h"
#include <opencv2/opencv.hpp>
#include "CannyLine.h"
#include "Sobel.hpp"

using namespace cv;
using namespace std;

void MyGammaCorrection(Mat& src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);

	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));

	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			//(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;
			//(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;
			//(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}



int main()
{	
	//string fileCur = "G://edge_contour.jpg";
	string fileCur = "G:/CZCV/LineSegment/project/CannyLines-v3/img/test.bmp";
	std::cout <<"img:"<<fileCur<<std::endl;
	//string fileCur = "G:/CZCV/LineSegment/data/YorkUrbanDB/P1020171/P1020171.jpg";

	cv::Mat img = imread(fileCur);
	std::cout<<img.empty()<<std::endl;



	//resize(img, img, Size(1280, 720));
	//resize(img, img, Size(800, 600));
	cv::Mat grayImg;
	cv::Mat imgShow(img.rows, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

	cvtColor(img, grayImg, COLOR_BGR2GRAY);

	/*float aaa = 0;
	for (int i = 0; i < 100; i++) {
		auto a1 = cv::getTickCount();
		cv::Mat dx(grayImg.rows, grayImg.cols, CV_16S, Scalar(0));
		cv::Mat dy(grayImg.rows, grayImg.cols, CV_16S, Scalar(0));

		cv::Sobel(grayImg, dx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
		cv::Sobel(grayImg, dy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
		auto a2 = cv::getTickCount();
		aaa += (a2 - a1) / cv::getTickFrequency() * 1000;
	}
	std::cout << "aaa:" << aaa/100 << std::endl;
	system("pause");*/

	float bbb = 0;
	cv::Mat res = cv::Mat::zeros(img.size(),img.type()); 
	for (int i = 0; i < 100; i++) {
		auto b1 = cv::getTickCount();
		int succ = IM_SobelSSE(img.data, res.data, img.cols, img.rows, img.cols * 3);
		if (succ) {
			std::cout << "failed:" << succ << std::endl;
		}
		/*namedWindow("result", 0);
		imshow("result", res);
		cv::waitKey(1);*/
		auto b2 = cv::getTickCount();
		bbb += (b2 - b1) / cv::getTickFrequency() * 1000;
	}
	std::cout << "bbb:" << bbb / 100 << std::endl;
	system("pause");


	float duration = 0;
	int64 start = 0, end = 0;
	for(int i=0;i<20;i++){
		start = cv::getTickCount();
		CannyLine detector;
		std::vector<std::vector<float> > lines;  // N x 4 array
		detector.cannyLine(grayImg, lines);
		end = cv::getTickCount();

		// show
		for (int m = 0; m < lines.size(); ++m)
		{
			cv::line(imgShow, cv::Point(lines[m][0], lines[m][1]),
				cv::Point(lines[m][2], lines[m][3]),
				cv::Scalar(0, 0, 255),
				1,
				cv::LINE_AA);
		}
		//namedWindow("result0", 0);
		//imshow("result0", imgShow);
		//cv::waitKey(0);
		duration += (end - start) / cv::getTickFrequency() * 1000;

	}

	std::cout << "total duration in main:" << duration/20 << std::endl;
	imwrite("result.png",imgShow);

	// std::cout << "****************" << std::endl;


	// vector<Mat> channels;
	// split(img, channels);//锟街革拷image1锟斤拷通锟斤拷


	// for (int i = 0; i < 3; i++) {
	// 	grayImg = channels[i];
	// 	//MyGammaCorrection(grayImg, grayImg, 1.1);

	// 	//Point center(grayImg.cols / 2, grayImg.rows / 2); //锟斤拷转锟斤拷锟斤拷
	// 	//double angle = -45;  //锟角讹拷
	// 	//double scale = 1.0;  //锟斤拷锟斤拷系锟斤拷
	// 	//Mat rotMat = getRotationMatrix2D(center, angle, scale);
	// 	//warpAffine(grayImg, grayImg, rotMat, grayImg.size());
	// 	//warpAffine(img, img, rotMat, img.size());

	// 	//Canny(grayImg, grayImg, 20, 40, 3, false);
	// 	//namedWindow("grayImg", 0);
	// 	//imshow("grayImg", grayImg);
	// 	//cv::waitKey(0);

	// 	CannyLine detector;
	// 	std::vector<std::vector<float> > lines;  // N x 4 array
	// 	detector.cannyLine(grayImg, lines);

	// 	// show
	// 	for (int m = 0; m < lines.size(); ++m)
	// 	{
	// 		cv::line(imgShow, cv::Point(lines[m][0], lines[m][1]),
	// 			cv::Point(lines[m][2], lines[m][3]),
	// 			cv::Scalar(0, 0, 255),
	// 			1,
	// 			cv::LINE_AA);
	// 	}
		
	// }
	// //namedWindow("result", 0);
	// //imshow("result", imgShow);
	// //cv::waitKey(0);
	// imwrite("result.png",imgShow);
	return 0;
}

