#ifndef _META_LINE_H_
#define _META_LINE_H_
#pragma once

#include <opencv2/opencv.hpp>
#include "types.h"
#include <vector>
#include "MyTime.h"
#include "math.h"

using namespace cv;
using namespace std;

#define DEBUG 0

enum DIR{LEFT,RIGHT,UP,DOWN};

class MetaLine
{
public:
	MetaLine() :visualMeaningfulGradient(70),p(0.125), sigma(1.0), thAngle(0.0){};
	~MetaLine() {};

	void MetaLineDetection(cv::Mat originalImage,float gausSigma, int gausHalfSize,std::vector<std::vector<float> > &lines);

private:
	void getInformations(cv::Mat &originalImage,float gausSigma, int gausHalfSize,float p);
	bool smartRouting(clusters_list_t &segments,float minDeviation,int minSize);
	void getMetaLine(clusters_list_t &segments,lines_list_t &metaLines,float thMSE);
	void getMetaLine(strings_list_t &strings,lines_list_t &metaLines,float thMSE);
	void metaLineExtending(lines_list_t &metaLines,int *removal);
	void metaLineMerging(lines_list_t &metaLines,int *removal);
	void lineValidityCheck(lines_list_t &metaLines,int *removal);

	//
	bool next(int &x_seed,int &y_seed);
	void subDivision(clusters_list_t &clusters, const string_t &string, const size_t first_index, const size_t last_index, const float min_deviation, const size_t min_size);

	void extendHirozontal(line_t &metaLineCur,lines_list_t  &metaLines,int *removal);
	void extendVertical(line_t &metaLineCur,lines_list_t  &metaLines,int *removal);
	int lineMerging(int IDCur,line_t &metaLineCur,size_list_t &IDHyps,lines_list_t &metaLines,float thAngle);
	int lineMerging2(int IDCur,line_t &metaLineCur,size_list_t &IDHyps,lines_list_t &metaLines,size_list_t &IDMerged);
	bool crossingCheck(Point2f pts,Point2f pte,int ID);
	int crossSearch(Point2i pts,float &angle,int ID);

	bool leastSquareFitting(string_t &string, float *parameters,float thMSE);
	bool leastSquareFitting(cluster_t &cluster, float *parameters,float thMSE);
	bool leastSquareFitting(std::vector<Point2f> &points, float *parameters,float thMSE);
	bool gradientWeightedLeastSquareFitting(string_t &string,float *parameters,float thMSE);

	float lineValidityCheckGradientLevel(line_t &metaLines);
	float lineValidityCheckGradientOrientation(line_t &metaLines);
	float probability(int N,int k,float p);

	//
public:
	float p;       //  const value for calc lmin, according to lsd
	float sigma;
	float thAngle;

	int thMeaningfulLength;     // l_min
	float visualMeaningfulGradient;   //  lambda_v, same as paper
	float thGradientLow;      // g_min
	float thGradientHigh;     // g_max
	cv::Mat cannyEdge;

private:
	int rows,cols;     // img h and w 
	int rows_1,cols_1 ; // h-1 and w-1
	int thSearchSteps;

	float N4;
	float N2;

	cv::Mat filteredImage;
	cv::Mat gradientMap; //the gradient value
	cv::Mat orientationMap; //index for the gradient orientation
	cv::Mat orientationMapInt;//
	cv::Mat searchingMap; //index for searching direction
	cv::Mat maskImage; //index for the position of gradient points

	std::vector<Point> gradientPoints;   // gradient loc
	std::vector<float> gradientValue;    // gradient val
	std::vector<float> greaterThan;
	std::vector<float> smallerThan;
};
#endif  // _META_LINE_H_
