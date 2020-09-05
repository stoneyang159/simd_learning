#ifndef _CANNY_LINE_H_
#define _CANNY_LINE_H_
#pragma once

//#include "cv.h"
//#include "highgui.h"
//#include "cxcore.h"
#include <opencv2/opencv.hpp>

class CannyLine
{
public:
	CannyLine(void);
	~CannyLine(void);

	static void cannyLine(cv::Mat &image,std::vector<std::vector<float> > &lines);
};

#endif // _CANNY_LINE_H_

