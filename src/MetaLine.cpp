#include "MetaLine.h"
#include "QuickSort.h"
#include "math.h"
#define INF 10000000000

#ifndef M_PI_4
#define M_PI_4  0.785398163397448309f
#endif

static const float atan2_p1 = 0.9997878412794807f;
static const float atan2_p3 = -0.3258083974640975f;
static const float atan2_p5 = 0.1555786518463281f;
static const float atan2_p7 = -0.04432655554792128f;

#define CZ_ATAN2_WITH_LUT  1
#define  ATAN_LOOK_UP_TABLE_N  256
#define  ATAN_LOOK_UP_TABLE_N_1 255
const static float ATAN_LOOK_UP_TABLE_SCALE = 100.0f;
float  ATAN_LOOK_UP_TABLE[ATAN_LOOK_UP_TABLE_N];

/*
opencv 改进版本
*/
float cvfastAtan2(float y, float x)
{
	float ax = (float)fabs(x);
	float ay = (float)fabs(y);
	float a, c, c2;
	if (ax >= ay)
	{
		c = ay / (ax + (float)DBL_EPSILON);
		c2 = c * c;
		a = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c;
	}
	else
	{
		c = ax / (ay + (float)DBL_EPSILON);
		c2 = c * c;
		a = CV_PI - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3) * c2 + atan2_p1) * c;
	}
	if (x < 0)
		a = CV_2PI - a;
	if (y < 0)
		a = -a;
	return a;
}


inline float czfastAtan2(float y, float x, float &mag)
{
	float ax = (float)fabs(x);
	float ay = (float)fabs(y);

	mag = ax + ay;

	float a, c, c2;
	if (ax >= ay)
	{
		c = ay / (ax + (float)DBL_EPSILON);
#if CZ_ATAN2_WITH_LUT
		int idx = int(c *ATAN_LOOK_UP_TABLE_SCALE);
		idx = idx > ATAN_LOOK_UP_TABLE_N_1 ? ATAN_LOOK_UP_TABLE_N_1 : idx;
		a = ATAN_LOOK_UP_TABLE[idx];
#else
		a = M_PI_4*c - c*(fabs(c) - 1.f)*(0.2447f + 0.0663f*fabs(c));
#endif		
	}
	else
	{
		c = ax /(ay + (float)DBL_EPSILON);
#if CZ_ATAN2_WITH_LUT
		int idx = int(c *ATAN_LOOK_UP_TABLE_SCALE);
		idx = idx > ATAN_LOOK_UP_TABLE_N_1 ? ATAN_LOOK_UP_TABLE_N_1 : idx;
		a = CV_PI - ATAN_LOOK_UP_TABLE[idx];
#else
		a = CV_PI - (M_PI_4*c - c*(fabs(c) - 1.f)*(0.2447f + 0.0663f*fabs(c)));
#endif		
	}
	if (x < 0)
		a = CV_2PI - a;
	if (y < 0)
		a = -a;
	return a;
}

/*
精度有缺失
*/
double FastArcTan(double y, double x)
{
	double y_x = y / x;
	double retval = M_PI_4*y_x - y_x*(fabs(y_x) - 1.f)*(0.2447f + 0.0663f*fabs(y_x));
	//float retval = atan(y_x);
	if (x >= 0 && y>= 0) {
		return retval;
	}
	else if(x >= 0 && y<0){
		return retval;
	}
	else if (x < 0 && y >= 0) {
		return CV_PI + retval;
	}
	else {
		return retval - CV_PI;
	}
}

float FastArcTan(float x)
{
	return M_PI_4*x - x*(fabs(x) - 1.f)*(0.2447f + 0.0663f*fabs(x));
}

bool MetaLine::gradientWeightedLeastSquareFitting(string_t &string,float *parameters,float sigma)
{
	int i,j;
	int N=string.size();

	float kCoarse=0.0;
	if (string[0].x==string[string.size()-1].x)
		kCoarse=INF;
	else
		kCoarse=float(string[0].y-string[string.size()-1].y)/(string[0].x-string[string.size()-1].x);

	float k=0.0,b=0.0,dev=0.0,temp=0.0;
	float totalGradient=0;
	std::vector<float> weight(N,0);
	float *ptr=(float*) gradientMap.data;	
	for (i=0;i<N;++i)
	{
		weight[i]=*(ptr+int(string[i].y)*cols+int(string[i].x));
		totalGradient+=weight[i];
	}
	for (i=0;i<N;++i)
		weight[i]/=totalGradient;
	
	if (abs(kCoarse)<1)
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumX2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=weight[i]*string[i].x;
			sumY+=weight[i]*string[i].y;
			sumX2+=weight[i]*string[i].x*string[i].x;
			sumXY+=weight[i]*string[i].x*string[i].y;
		}

		b=(sumX2*sumY-sumX*sumXY)/(sumX2-sumX*sumX);
		k=(sumXY-sumX*sumY)/(sumX2-sumX*sumX);

		for (i=0;i<N;++i)
		{
			temp=string[i].y-k*string[i].x-b;
			dev+=temp*temp;
		}
		dev=sqrt(dev/float(N-2));

		parameters[0]=0; //label
	}
	else
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumY2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=weight[i]*string[i].x;
			sumY+=weight[i]*string[i].y;
			sumY2+=weight[i]*string[i].y*string[i].y;
			sumXY+=weight[i]*string[i].x*string[i].y;
		}

		b=(sumY2*sumX-sumY*sumXY)/(sumY2-sumY*sumY);
		k=(sumXY-sumX*sumY)/(sumY2-sumY*sumY);

		for (i=0;i<N;++i)
		{
			temp=string[i].x-k*string[i].y-b;
			dev+=temp*temp;
		}
		dev=sqrt(dev/float(N-2));

		parameters[0]=1; //label
	}
	parameters[1]=k;
	parameters[2]=b;
	parameters[3]=dev;

	if (dev<sigma)
		return true;
	else
		return false;
}

bool MetaLine::leastSquareFitting(string_t &string, float *parameters,float sigma)
{
	int i,j;
	int N=string.size();

	float kCoarse=0.0;
	if (string[0].x==string[N-1].x)
		kCoarse=INF;
	else
		kCoarse=float(string[0].y-string[N-1].y)/(string[0].x-string[N-1].x);

	float k=0.0,b=0.0,dev=0.0,temp=0.0,devMax=0.0;
	if (abs(kCoarse)<1)
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumX2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=string[i].x;
			sumY+=string[i].y;
			sumX2+=string[i].x*string[i].x;
			sumXY+=string[i].x*string[i].y;
		}

		b=(sumX2*sumY-sumX*sumXY)/(N*sumX2-sumX*sumX);
		k=(N*sumXY-sumX*sumY)/(N*sumX2-sumX*sumX);

		for (i=0;i<N;++i)
		{
			temp=abs(string[i].y-k*string[i].x-b);
			if (temp>devMax) devMax=temp;
			dev+=temp*temp;
		}
		dev=sqrt(dev/float(N-2));
		parameters[0]=0; //label
	}
	else
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumY2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=string[i].x;
			sumY+=string[i].y;
			sumY2+=string[i].y*string[i].y;
			sumXY+=string[i].x*string[i].y;
		}

		b=(sumY2*sumX-sumY*sumXY)/(N*sumY2-sumY*sumY);
		k=(N*sumXY-sumX*sumY)/(N*sumY2-sumY*sumY);

		for (i=0;i<N;++i)
		{
			temp=abs(string[i].x-k*string[i].y-b);
			if (temp>devMax) devMax=temp;
			dev+=temp*temp;
		}
		dev=sqrt(dev/float(N-2));
		parameters[0]=1; //label
	}
	parameters[1]=k;
	parameters[2]=b;
	parameters[3]=dev;

	if (dev<sigma&&devMax<3*sigma)
		return true;
	else
		return false;
}

bool MetaLine::leastSquareFitting(cluster_t &cluster, float *parameters,float sigma)
{
	int i,j;
	int N=cluster.size;

	float kCoarse=0.0;
	if (cluster.pixels[0].x==cluster.pixels[N-1].x)
		kCoarse=INF;
	else
		kCoarse=float(cluster.pixels[0].y-cluster.pixels[N-1].y)/(cluster.pixels[0].x-cluster.pixels[N-1].x);

	float k=0.0,b=0.0,dev=0.0,temp=0.0;
	if (abs(kCoarse)<1)
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumX2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=cluster.pixels[i].x;
			sumY+=cluster.pixels[i].y;
			sumX2+=cluster.pixels[i].x*cluster.pixels[i].x;
			sumXY+=cluster.pixels[i].x*cluster.pixels[i].y;
		}

		b=(sumX2*sumY-sumX*sumXY)/(N*sumX2-sumX*sumX);
		k=(N*sumXY-sumX*sumY)/(N*sumX2-sumX*sumX);

		std::vector<float> offsets(N,0);
		for (i=0;i<N;++i)
		{
			temp=cluster.pixels[i].y-k*cluster.pixels[i].x-b;
			offsets[i]=temp;
			dev+=temp*temp;
		}
		float thegma=sqrt(dev/float(N-2));

		int start=0,end=N-1;
		int index=0;
		float devOutliers=0.0;
		for (i=0;i<N;++i)
		{
			if (offsets[i]<1.0)
				index++;
			if (index==2)
			{
				start=i;
				break;
			}
			else
				devOutliers+=offsets[i]*offsets[i];
		}

		index=0;
		for (i=N-1;i>=0;--i)
		{
			if (offsets[i]<1.0)
				index++;
			if (index==2)
			{
				end=i;
				break;
			}
			else
				devOutliers+=offsets[i]*offsets[i];
		}
		if (end<=start) return false;

		dev=sqrt((dev-devOutliers)/float(N-2));
		cluster.pixels=&(cluster.pixels[start]);
		cluster.size=end-start+1;
		parameters[0]=0; //label

	}
	else
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumY2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=cluster.pixels[i].x;
			sumY+=cluster.pixels[i].y;
			sumY2+=cluster.pixels[i].y*cluster.pixels[i].y;
			sumXY+=cluster.pixels[i].x*cluster.pixels[i].y;
		}

		b=(sumY2*sumX-sumY*sumXY)/(N*sumY2-sumY*sumY);
		k=(N*sumXY-sumX*sumY)/(N*sumY2-sumY*sumY);

		std::vector<float> offsets(N,0);
		for (i=0;i<N;++i)
		{
			temp=cluster.pixels[i].x-k*cluster.pixels[i].y-b;
			offsets[i]=temp;
			dev+=temp*temp;
		}
		float thegma=sqrt(dev/float(N-2));

		int start=0,end=N-1;
		int index=0;
		float devOutliers=0.0;
		for (i=0;i<N;++i)
		{
			if (offsets[i]<1.0)
				index++;
			if (index==2)
			{
				start=i;
				break;
			}
			else
				devOutliers+=offsets[i]*offsets[i];
		}

		index=0;
		for (i=N-1;i>=0;--i)
		{
			if (offsets[i]<1.0)
				index++;
			if (index==2)
			{
				end=i;
				break;
			}
			else
				devOutliers+=offsets[i]*offsets[i];
		}
		if (end<=start) return false;

		dev=sqrt((dev-devOutliers)/float(N-2));

		cluster.pixels=&(cluster.pixels[start]);
		cluster.size=end-start+1;
		parameters[0]=1; //label
	}
	parameters[1]=k;
	parameters[2]=b;
	parameters[3]=dev;

	return true;
}


bool MetaLine::leastSquareFitting(std::vector<Point2f> &points, float *parameters,float sigma)
{
	int i,j;
	int N=points.size();

	float kCoarse=0.0;
	if (points[0].x==points[N-1].x)
		kCoarse=INF;
	else
		kCoarse=float(points[0].y-points[N-1].y)/(points[0].x-points[N-1].x);

	float k=0.0,b=0.0,dev=0.0,temp=0.0,devMax=0.0;
	if (abs(kCoarse)<1)   // y = kx + b
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumX2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=points[i].x;   // B
			sumY+=points[i].y;   // D
			sumX2+=points[i].x*points[i].x;  // A
			sumXY+=points[i].x*points[i].y;  // C
		}

		b=(sumX2*sumY-sumX*sumXY)/(N*sumX2-sumX*sumX);   // (AD - BC)/(An - BB)
		k=(N*sumXY-sumX*sumY)/(N*sumX2-sumX*sumX);      // (Cn - BD)/(An - BB)

		for (i=0;i<N;++i)
		{
			temp=abs(points[i].y-k*points[i].x-b);
			if (temp>devMax) devMax=temp;
			dev+=temp*temp;   // 锟斤拷锟斤拷
		}
		dev=sqrt(dev/float(N-2));  // 锟斤拷准锟斤拷
		parameters[0]=0; // 锟斤拷锟斤拷k锟斤拷b 锟斤拷 y=kx+b 锟斤拷
	}
	else  // 锟斤拷锟斤拷x = ky + b
	{
		float sumX=0.0;
		float sumY=0.0;
		float sumY2=0.0;
		float sumXY=0.0;

		for (i=0;i<N;++i)
		{
			sumX+=points[i].x;  // B
			sumY+=points[i].y;  // D
			sumY2+=points[i].y*points[i].y;  // E
			sumXY+=points[i].x*points[i].y;  // C
		}

		b=(sumY2*sumX-sumY*sumXY)/(N*sumY2-sumY*sumY);  //(EB - DC)/(En - DD)
		k=(N*sumXY-sumX*sumY)/(N*sumY2-sumY*sumY);

		for (i=0;i<N;++i)
		{
			temp=abs(points[i].x-k*points[i].y-b);
			if (temp>devMax) devMax=temp;
			dev+=temp*temp;
		}
		dev=sqrt(dev/float(N-2));
		parameters[0]=1;  // 锟斤拷锟斤拷k锟斤拷b 锟斤拷 x=ky+b 
	}
	parameters[1]=k;
	parameters[2]=b;
	parameters[3]=dev;

	if (dev<sigma&&devMax<3*sigma)  //锟斤拷准锟斤拷小锟斤拷一锟斤拷锟斤拷sigma锟斤拷 devMax小锟斤拷锟斤拷锟斤拷锟斤拷sigma锟斤拷锟缴癸拷
		return true;
	else
		return false;
}

int MetaLine::crossSearch(Point2i pts,float &angle,int ID)
{
	int i,j;

	float *ptrMI=(float*) maskImage.data;

	//search along edges
	static const int X_OFFSET[8] = {  0,  1,  0, -1,  1, -1, -1,  1 };
	static const int Y_OFFSET[8] = {  1,  0, -1,  0,  1,  1, -1, -1 };

	std::vector<Point2i> edgeChain;
	int xSeed=pts.x;
	int ySeed=pts.y;
	int x,y;
	
	int count=0;
	bool isEnd=false;
	while(count!=10&&!isEnd)
	{
		for (size_t i=0; i!=8; ++i)
		{
			x = xSeed + X_OFFSET[i];
			if ((0 <= x) && (x < cols))
			{
				y = ySeed + Y_OFFSET[i];
				if ((0 <= y) && (y < rows))
				{
					cout<<ptrMI[y*cols+x]<<endl;
					if (ptrMI[y*cols+x]<0&&ptrMI[y*cols+x]!=ID) return -1; //-1 for stop merging

					if (ptrMI[y*cols+x]==1)
					{
						xSeed = x;
						ySeed = y;
						edgeChain.push_back(Point2i(xSeed,ySeed));
						ptrMI[y*cols+x]=0;
						break;
					}
				}
				else
					isEnd=true;
			}
			else
				isEnd=true;
		}
		count++;
	}

	if (edgeChain.size()>=3)
	{
		float deltaX=edgeChain[0].x-edgeChain[edgeChain.size()-1].x;
		float deltaY=edgeChain[0].y-edgeChain[edgeChain.size()-1].y;
		angle=atan(deltaY/deltaX+0.00000101);

		return 1;  //1 for finding a edge chain
	}
	else
		return 0;  //0 for no edge chain
	
}

bool MetaLine::crossingCheck(Point2f pts,Point2f pte,int ID)
{
	int i,j;

	float *ptrMI=(float*) maskImage.data;

	float deltaX=pts.x-pte.x;
	float deltaY=pts.y-pte.y;
	if (abs(deltaX)>abs(deltaY))
	{
		float k=deltaY/deltaX;
		float b=pts.y-k*pts.x;
		float angle=atan(k);

		int xMin=int(min(pts.x,pte.x)+0.5);
		int xMax=int(max(pts.x,pte.x)+0.5);
		if (xMin<0) xMin=0;
		if (xMin>=cols) xMin=cols-2;
		if (xMax<0) xMax=0;
		if (xMax>=cols) xMax=cols-1;

		float y=k*xMin+b;
		for (i=xMin+1;i<xMax;++i)
		{
			int xInt=i;
			y+=k;
			int yInt=int(y+0.5);
			if (0<yInt&&yInt<rows_1)
			{
				int loc=yInt*cols+xInt;
				int e0=(int)ptrMI[loc-cols];
				int e1=(int)ptrMI[loc];
				int e2=(int)ptrMI[loc+cols];

				if ((e0<0&&e0!=ID)||(e1<0&&e1!=ID)||(e2<0&&e2!=ID)) return false; //find another line
				if (e0==1||e1==1||e2==1)
				{
					float angleEdge=0;
					int Rnt=crossSearch(Point2i(xInt,yInt), angleEdge,ID); // 1for finding a edge chain, 0 for non, -1 for another line segment
					if (Rnt)
					{
						if ((abs(angle-angleEdge)>CV_PI/6.0)||(CV_PI-abs(angle-angleEdge)>CV_PI/6.0))
							return false;
					}
					else if (Rnt<0)
						return false;
				}
			}
			else 
				break;
		}
	}
	else
	{
		float k=deltaX/deltaY;
		float b=pts.x-k*pts.y;
		float angle=CV_PI/2.0-atan(k);

		int yMin=int(min(pts.y,pte.y)+0.5);
		int yMax=int(max(pts.y,pte.y)+0.5);
		if (yMin<0) yMin=0;
		if (yMin>=rows) yMin=rows-2;
		if (yMax<0) yMax=0;
		if (yMax>=rows) yMax=rows-1;

		float x=k*yMin+b;
		for (i=yMin+1;i<yMax;++i)
		{
			int yInt=i;
			x+=k;
			int xInt=int(x+0.5);

			if (0<xInt&&xInt<cols_1)
			{
				int loc=yInt*cols+xInt;
				int e0=(int)ptrMI[loc-1];
				int e1=(int)ptrMI[loc];
				int e2=(int)ptrMI[loc+1];

				if ((e0<0&&e0!=ID)||(e1<0&&e1!=ID)||(e2<0&&e2!=ID)) return false; //find another line 
				if (e0==1||e1==1||e2==1)
				{
					float angleEdge=0;
					int Rnt=crossSearch(Point2i(xInt,yInt), angleEdge,ID); // 1for finding a edge chain, 0 for non, -1 for another line segment
					if (Rnt)
					{
						if ((abs(angle-angleEdge)>CV_PI/6.0)||(CV_PI-abs(angle-angleEdge)>CV_PI/6.0))
							return false;
					}
					else if (Rnt<0)
						return false;
				}
			}
			else
				break;
		}
	}

	return true;
}

void MetaLine::extendHirozontal(line_t &metaLineCur,lines_list_t  &metaLines,int *removal)
{
	int i,j;

	float *ptrMI=(float*) maskImage.data;
	float *ptrGM=(float*) gradientMap.data;

	int ID=metaLineCur.ID;
	float k=metaLineCur.k;
	float b=metaLineCur.b;
	float xs=metaLineCur.points[0].x;
	float xe=metaLineCur.points[metaLineCur.points.size()-1].x;
	float ye=k*xe+b;
	float xCur=xe;
	float yCur=ye;
	int xInt=int(xCur+0.5);
	int yInt=int(yCur+0.5);
	int index=(xe-xs)/abs(xe-xs);

	cluster_t pointsFormer;
	pointsFormer.pixels=&metaLineCur.points[0];
	pointsFormer.size=metaLineCur.points.size();

	int loc=0;
	int gap=0;
	int edge=0;
	int edgeTotal=0;

	float m0,m1,m2;
	float g0,g1,g2;
	bool extend=false;
	while (1)
	{
		xInt+=index;
		yCur+=index*k;
		yInt=int(yCur+0.5);
		bool chooseUp=false;
		if (yInt+0.5>yCur)chooseUp=true;

		if (0<xInt&&xInt<cols_1&&0<yInt&&yInt<rows_1)
		{
			loc=yInt*cols+xInt;
			m0=ptrMI[loc];  
			m1=ptrMI[loc-cols];
			m2=ptrMI[loc+cols];

			size_list_t lineHyps;
			if (m0<0&&m0!=-ID) { size_t &n=lineHyps.push_back(); n=-m0-1; }
			if (m1<0&&m1!=-ID) { size_t &n=lineHyps.push_back(); n=-m1-1; }
			if (m2<0&&m2!=-ID) { size_t &n=lineHyps.push_back(); n=-m2-1; }
			if (lineHyps.size())
			{
				int Rnt=lineMerging(ID,metaLineCur,lineHyps,metaLines,thAngle);
				if (Rnt!=-1)
				{
					float newPara[4];
					if (leastSquareFitting(metaLineCur.points,newPara,sigma))
					{
						if (!newPara[0])
						{
							k=newPara[1];
							b=newPara[2];
							xInt=metaLineCur.points[metaLineCur.points.size()-1].x;
							yCur=k*xInt+b;
							removal[Rnt]=1;
							extend=true;
						}
						else
						{
							metaLineCur.dir=newPara[0];  //0 for k<1, 1 for k >=1
							metaLineCur.k=newPara[1];
							metaLineCur.b=newPara[2];
							metaLineCur.dev=newPara[3];

							metaLineCur.ys=metaLineCur.points[0].y;
							metaLineCur.xs=metaLineCur.k*metaLineCur.ys+metaLineCur.b;
							metaLineCur.ye=metaLineCur.points[metaLineCur.points.size()-1].y;
							metaLineCur.xe=metaLineCur.k*metaLineCur.ye+metaLineCur.b;

							extendVertical(metaLineCur,metaLines,removal);

							extend=true;
						}

						//
						pointsFormer.size=metaLineCur.points.size();
						for (j=0;j<metaLineCur.points.size();++j)
						{
							int loc=metaLineCur.points[j].y*cols+metaLineCur.points[j].x;
							*(ptrMI+loc)=-float(ID);
						}
					}
					else
					{
						metaLineCur.points.resize(pointsFormer.size);
						break;
					}
				}
				else
					break;
			}
			else
			{
				if ((m0==1||m1==1||m2==1)&&(m0+m1+m2)==1)
				{
					if (m0==1&&m0>=m1&&m0>=m2) 
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=xInt;  p.y=yInt;
						ptrMI[loc]=-ID;
						
					}
					else if (m1==1&&m1>=m0&&m1>=m2) 
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=xInt;  p.y=yInt-1;
						ptrMI[loc-cols]=-ID;
					}
					else if (m2==1&&m2>=m0&&m2>=m1) 
					{
						if (chooseUp) {gap++;continue;};
						pixel_t &p=metaLineCur.points.push_back();
						p.x=xInt;  p.y=yInt+1;
						ptrMI[loc+cols]=-ID;
					}
					edge++;
					edgeTotal++;
				}
				else
					gap++;

				if ((double)gap/edge>=0.25||edge==0) break;
				if (gap==2)
				{
					edge=0;
					gap=0;
				}

				if (edgeTotal>=thMeaningfulLength)
				{
					float metaLineParas[4];
					leastSquareFitting(metaLineCur.points,metaLineParas,sigma);
					k=metaLineParas[1];
					b=metaLineParas[2];
					yCur=xInt*k+b;
					edgeTotal=0;
					gap=0;
					extend=true;

					pointsFormer.size=metaLineCur.points.size();
				}
			}
		}
		else
			break;
	}

	if (extend)
	{
		float metaLineParas[4];
		leastSquareFitting(metaLineCur.points,metaLineParas,sigma);

		metaLineCur.dir=metaLineParas[0];  //0 for k<1, 1 for k >=1
		metaLineCur.k=metaLineParas[1];
		metaLineCur.b=metaLineParas[2];
		metaLineCur.dev=metaLineParas[3];

		if (metaLineCur.dir==0)
		{
			metaLineCur.xs=metaLineCur.points[0].x;
			metaLineCur.ys=metaLineCur.k*metaLineCur.xs+metaLineCur.b;
			metaLineCur.xe=metaLineCur.points[metaLineCur.points.size()-1].x;
			metaLineCur.ye=metaLineCur.k*metaLineCur.xe+metaLineCur.b;
		}
		else
		{
			metaLineCur.ys=metaLineCur.points[0].y;
			metaLineCur.xs=metaLineCur.k*metaLineCur.ys+metaLineCur.b;
			metaLineCur.ye=metaLineCur.points[metaLineCur.points.size()-1].y;
			metaLineCur.xe=metaLineCur.k*metaLineCur.ye+metaLineCur.b;
		}

	}
	else
		metaLineCur.points.resize(pointsFormer.size);
}

void MetaLine::extendVertical(line_t &metaLineCur,lines_list_t  &metaLines,int *removal)
{
	int i,j;

	float *ptrMI=(float*) maskImage.data;
	float *ptrGM=(float*) gradientMap.data;

	int ID=metaLineCur.ID;
	float k=metaLineCur.k;
	float b=metaLineCur.b;
	float ys=metaLineCur.points[0].y;
	float ye=metaLineCur.points[metaLineCur.points.size()-1].y;
	float xe=k*ye+b;
	float xCur=xe;
	float yCur=ye;
	int xInt=int(xCur+0.5);
	int yInt=int(yCur+0.5);
	int index=(ye-ys)/abs(ye-ys);

	cluster_t pointsFormer;
	pointsFormer.pixels=&metaLineCur.points[0];
	pointsFormer.size=metaLineCur.points.size();

	int loc=0;
	int gap=0;
	int edge=0;
	int edgeTotal=0;
	
	float m0,m1,m2;
	float g0,g1,g2;
	bool extend=false;
	while (1)
	{
		yInt+=index;
		xCur+=index*k;
		xInt=int(xCur+0.5);
		bool chooseLeft=false;
		if (xInt+0.5>xCur)chooseLeft=true;

		if (0<xInt&&xInt<cols_1&&0<yInt&&yInt<rows_1)
		{
			loc=yInt*cols+xInt;
			m0=ptrMI[loc];
			m1=ptrMI[loc-1];
			m2=ptrMI[loc+1];

			size_list_t lineHyps;
			if (m0<0&&m0!=-ID) { size_t &n=lineHyps.push_back(); n=-m0-1; }
			if (m1<0&&m1!=-ID) { size_t &n=lineHyps.push_back(); n=-m1-1; }
			if (m2<0&&m2!=-ID) { size_t &n=lineHyps.push_back(); n=-m2-1; }
			if (lineHyps.size())
			{
				int Rnt=lineMerging(ID,metaLineCur,lineHyps,metaLines,thAngle);

				if (Rnt!=-1)
				{
					float newPara[4];
					if (leastSquareFitting(metaLineCur.points,newPara,sigma))
					{
						if (newPara[0])
						{
							k=newPara[1];
							b=newPara[2];
							yInt=metaLineCur.points[metaLineCur.points.size()-1].y;
							xCur=k*yInt+b;
							removal[Rnt]=1;
							extend=true;
						}
						else
						{
							metaLineCur.dir=newPara[0];  //0 for k<1, 1 for k >=1
							metaLineCur.k=newPara[1];
							metaLineCur.b=newPara[2];
							metaLineCur.dev=newPara[3];

							metaLineCur.xs=metaLineCur.points[0].x;
							metaLineCur.ys=metaLineCur.k*metaLineCur.xs+metaLineCur.b;
							metaLineCur.xe=metaLineCur.points[metaLineCur.points.size()-1].x;
							metaLineCur.ye=metaLineCur.k*metaLineCur.xe+metaLineCur.b;
							extendHirozontal(metaLineCur,metaLines,removal);
						}

						//
						pointsFormer.size=metaLineCur.points.size();
						for (j=0;j<metaLineCur.points.size();++j)
						{
							int loc=metaLineCur.points[j].y*cols+metaLineCur.points[j].x;
							*(ptrMI+loc)=-float(ID);
						}
					}
					else
					{
						metaLineCur.points.resize(pointsFormer.size);
						break;
					}
				}
				else
					break;
			}
			else
			{
				if ((m0==1||m1==1||m2==1)&&(m0+m1+m2)==1)
				{
					if (m0==1&&m0>=m1&&m0>=m2) 
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=xInt;  p.y=yInt;
						ptrMI[loc]=-ID;
					}
					else if (m1==1&&m1>=m0&&m1>=m2) 
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=xInt-1;  p.y=yInt;
						ptrMI[loc-1]=-ID;
					}
					else if (m2==1&&m2>=m0&&m2>=m1) 
					{
						if (chooseLeft) {gap++;continue;};
						pixel_t &p=metaLineCur.points.push_back();
						p.x=xInt+1;  p.y=yInt;
						ptrMI[loc+1]=-ID;
					}
					edge++;
					edgeTotal++;
				}
				else
					gap++;

				if ((double)gap/edge>=0.25||edge==0) break;
				if (gap==2)
				{
					edge=0;
					gap=0;
				}

				if (edgeTotal>=thMeaningfulLength)
				{
					float metaLineParas[4];
					leastSquareFitting(metaLineCur.points,metaLineParas,sigma);
					k=metaLineParas[1];
					b=metaLineParas[2];
					xCur=yInt*k+b;
					edgeTotal=0;
					gap=0;
					extend=true;

					pointsFormer.size=metaLineCur.points.size();
				}
			}
		}
		else
			break;
	}

	//
	if (extend)
	{
		float metaLineParas[4];
		leastSquareFitting(metaLineCur.points,metaLineParas,sigma);

		metaLineCur.dir=metaLineParas[0];  //0 for k<1, 1 for k >=1
		metaLineCur.k=metaLineParas[1];
		metaLineCur.b=metaLineParas[2];
		metaLineCur.dev=metaLineParas[3];

		if (metaLineCur.dir==0)
		{
			metaLineCur.xs=metaLineCur.points[0].x;
			metaLineCur.ys=metaLineCur.k*metaLineCur.xs+metaLineCur.b;
			metaLineCur.xe=metaLineCur.points[metaLineCur.points.size()-1].x;
			metaLineCur.ye=metaLineCur.k*metaLineCur.xe+metaLineCur.b;
		}
		else
		{
			metaLineCur.ys=metaLineCur.points[0].y;
			metaLineCur.xs=metaLineCur.k*metaLineCur.ys+metaLineCur.b;
			metaLineCur.ye=metaLineCur.points[metaLineCur.points.size()-1].y;
			metaLineCur.xe=metaLineCur.k*metaLineCur.ye+metaLineCur.b;
		}
	}
	else
		metaLineCur.points.resize(pointsFormer.size);
}

float MetaLine::probability(int N,int k,float p)
{
	int i=0;
	float v=pow(p,N);
	float prob=v;
	for (i=0;i<N-k;++i)
	{
		v=v*float(N-i)/float(1+i)*(1-p)/p;
		prob+=v;
	}
	return prob;
}

void MetaLine::getInformations(cv::Mat &originalImage,float gausSigma, int gausHalfSize,float p)
{
	//const int64 starta = cv::getTickCount();
	int i,j,m,n;
	int grayLevels=255;  // bin number of hist
	int apertureSize=3;
	float anglePer=CV_PI/8.0;
	float gNoise=1.3333;//1.3333 
	thGradientLow=gNoise;   

	cols = originalImage.cols;  //
	rows = originalImage.rows;
	cols_1=cols-1;  //
	rows_1=rows-1;
	N4=pow(double(rows)*cols,2.0);  // l_max 锟斤拷锟侥次凤拷

	// pixel num 
	int imageDataLength = rows*originalImage.cols; 

	// l_max = sqrt(rows * cols) , which is max(rows, cols) in paper
	// l_min in paper,  l_min = -4 * log(l_max) / log(p)  ,  p = 1/8
	// l_min = -4 * log(sqrt(rows * cols)) / log(p) 
	//       = 2 * log(rows * cols) / log(1/p)              (锟斤拷锟斤拷)
	//       = int(2 * log(rows * cols) / log(1/p) + 0.5)   (锟斤拷锟斤拷锟斤拷锟斤拷)
	thMeaningfulLength=int(2.0*log((float)rows*cols)/log(8.0)+0.5);
	// theta_m in paper,   theta_m = 2 * atan(2/lmin)
	thAngle=2*atan(2.0/float(thMeaningfulLength));         

	//get gray image
	cv::Mat grayImage;
	int aa=originalImage.channels();

	if ( originalImage.channels() == 1 )
		grayImage = originalImage;
	else
		cv::cvtColor(originalImage, grayImage, COLOR_RGB2GRAY);

	//gaussian filter
	if ( gausSigma > 0.0 && gausHalfSize > 0 )
	{
		int gausSize = gausHalfSize*2 + 1;
		cv::GaussianBlur(grayImage, filteredImage, cv::Size(gausSize,gausSize), gausSigma);
	}

	//get gradient map and orientation map
	gradientMap = Mat::zeros(filteredImage.rows,filteredImage.cols,CV_32FC1);
	orientationMap = Mat::zeros(filteredImage.rows,filteredImage.cols,CV_32FC1);  // 锟斤拷锟斤拷锟斤拷 -pi~pi
	orientationMapInt=Mat::zeros(filteredImage.rows,filteredImage.cols,CV_8U);    // 0~16 锟斤拷应 0~2*pi
	maskImage = Mat::zeros(filteredImage.rows,filteredImage.cols,CV_32FC1);
	cv::Mat orientationIndex = Mat::zeros(filteredImage.rows,filteredImage.cols,CV_8U);

	cv::Mat dx(filteredImage.rows, filteredImage.cols, CV_16S,Scalar(0));
	cv::Mat dy(filteredImage.rows, filteredImage.cols, CV_16S,Scalar(0));

	cv::Sobel(filteredImage, dx, CV_16S, 1, 0, apertureSize, 1, 0, cv::BORDER_REPLICATE);
	cv::Sobel(filteredImage, dy, CV_16S, 0, 1, apertureSize, 1, 0, cv::BORDER_REPLICATE);
	

	//const int64 startbb = cv::getTickCount();

	//temp
	cv::Mat temp(filteredImage.rows, filteredImage.cols, CV_8U,Scalar(0));
	char *ptr_temp=(char *)temp.data;
	for (unsigned int i = 0; i < ATAN_LOOK_UP_TABLE_N; i++)
	{
		ATAN_LOOK_UP_TABLE[i] = std::atan(i / ATAN_LOOK_UP_TABLE_SCALE);
	}

	//calculate gradient and orientation
	int totalNum = 0;
	int t = 8;
	std::vector<int> histogram(t*grayLevels, 0);

#pragma omp parallel for 
	for (i = 0; i<rows; ++i)
	{
		float *ptrG  = gradientMap.ptr<float>(i);
		float *ptrO  = orientationMap.ptr<float>(i);
		uchar *ptrOI = orientationMapInt.ptr<uchar>(i);
		short *ptrX = dx.ptr<short>(i);
		short *ptrY = dy.ptr<short>(i);
		for (j = 0; j < cols; j++)
		{
			float gx = ptrX[j];
			float gy = ptrY[j];
			ptrO[j] = czfastAtan2(gx, -gy, ptrG[j]);
			ptrOI[j] = int((ptrO[j] + CV_PI) / anglePer);  // -pi ~ pi  ->  0 ~ 16
			ptrOI[j] = ptrOI[j] % 16;
		}
	}

	for (i = 0; i<rows; ++i)
	{
		float *ptrG = gradientMap.ptr<float>(i);
		for (j = 0; j < cols; ++j)
		{
			float &g = ptrG[j];
			if (g > thGradientLow)
			{
				histogram[int(g + 0.5)]++;  //TODO  maybe use subblock for boost
				totalNum++; // M in paper, for normalize
			}
			else
				g = 0.0;  // regarded as noise and set to zero
		}
	}



	//const int64 startcc = cv::getTickCount();
	//gradient statistic
	N2=0;
	for (i=0;i<histogram.size();++i)
	{
		if (histogram[i])
			N2+=(float)histogram[i]*(histogram[i]-1);    // l_k * (l_k - 1)
	}
	float pMax=1.0/exp((log(N2)/thMeaningfulLength));  //
	float pMin=1.0/exp((log(N2)/sqrt((float)cols*rows)));

	greaterThan=std::vector<float>(t*grayLevels,0);  // H(u), in paper
	smallerThan=std::vector<float>(t*grayLevels,0);  // no use
	float count=0.f;
	float totalNum_add_1 = totalNum + 1;
	for (i=t*grayLevels-1;i>=0;--i)
	{
		count+=histogram[i];
		float probabilityGreater=count/ totalNum_add_1;
		greaterThan[i]=probabilityGreater;
	}
	count=0;

	// NFA(S) = N_p * H(u)^l
	
	// NFA(S) >= 1 
	// N_p * H(u)^l >= 1
	// H(u)^l >= 1/ N_p
	// l * log(H(u)) >= -log(N_p)
	// H(u) >= exp(-log(N_p)/l)
	
	// pMax = exp(-log(N_p)/l_min)
	// pMin = exp(-log(N_p)/l_max)


	for (i=t*grayLevels-1;i>=0;--i)
	{
		if (greaterThan[i]>pMax)   // all S has NFA(S)>=1
		{
			thGradientHigh=i;
			break;
		}
	}
	for (i=t*grayLevels-1;i>=0;--i)
	{
		if (greaterThan[i]>pMin)   // all S has NFA(S)<=1
		{
			thGradientLow=i;
			break;
		}
	}
	if (thGradientLow<gNoise) thGradientLow=gNoise;

	//convert probabilistic meaningful to visual meaningful
	// g_max = sqrt(lambda_v / g_max) * g_max = sqrt(lambda_v * g_max)
	thGradientHigh=sqrt(thGradientHigh*visualMeaningfulGradient);
	//const int64 startdd = cv::getTickCount();
	//canny
	cv::Canny(filteredImage,cannyEdge,thGradientLow,thGradientHigh,apertureSize);
	//const int64 startee = cv::getTickCount();
	//
	int num=0;
	uchar *ptrCanny=cannyEdge.data;       // canny retval
	float *ptrM=(float*)maskImage.data;   // edge map
	float *ptrG=(float*)gradientMap.data; // gradientMap

	std::vector<std::vector<Point>> gradientPointsVec4(4);
	std::vector<std::vector<float>> gradientValueVec4(4);

	int numVec4[4] = { 0 };

	for (i=0;i<rows;++i)
	{
		int baseIdx = i*cols;
		for (j=0; j<cols; j+=4)
		{
			if (*(ptrCanny + 0))
			{
				* (ptrM+0) = 1;
				int idx = j + 0;
				gradientPointsVec4[0].emplace_back(Point(idx, i));
				gradientValueVec4[0].emplace_back(ptrG[baseIdx + idx]);
				numVec4[0] += 1;
			}
			if (*(ptrCanny + 1))
			{
				*(ptrM + 1) = 1;
				int idx = j + 1;
				gradientPointsVec4[1].emplace_back(std::move(Point(idx, i)));
				gradientValueVec4[1].emplace_back(ptrG[baseIdx + idx]);
				numVec4[1] += 1;
			}
			if (*(ptrCanny + 2))
			{
				*(ptrM + 2) = 1;
				int idx = j + 2;
				gradientPointsVec4[2].emplace_back(std::move(Point(idx, i)));
				gradientValueVec4[2].emplace_back(ptrG[baseIdx + idx]);
				numVec4[2] += 1;
			}
			if (*(ptrCanny + 3))
			{
				*(ptrM + 3) = 1;
				int idx = j + 3;
				gradientPointsVec4[3].emplace_back(std::move(Point(idx, i)));
				gradientValueVec4[3].emplace_back(ptrG[baseIdx + idx]);
				numVec4[3] += 1;
			}
			ptrM += 4;
			ptrCanny += 4;
		}
	}
	for (unsigned int i = 0; i < 4; i++)
	{
		gradientPoints.insert(gradientPoints.end(), gradientPointsVec4[i].begin(), gradientPointsVec4[i].end());
		gradientValue.insert(gradientValue.end(), gradientValueVec4[i].begin(), gradientValueVec4[i].end());
	}
	num = numVec4[0] + numVec4[1] + numVec4[2] + numVec4[3];

	/*const int64 startff = cv::getTickCount();
	double duration = (startbb - starta) / cv::getTickFrequency() * 1000;
	std::cout << "stepa duration:" << duration << std::endl;
	duration = (startcc - startbb) / cv::getTickFrequency() * 1000;
	std::cout << "stepb duration:" << duration << std::endl;
	duration = (startdd - startcc) / cv::getTickFrequency() * 1000;
	std::cout << "stepc duration:" << duration << std::endl;
	duration = (startee - startdd) / cv::getTickFrequency() * 1000;
	std::cout << "stepd duration:" << duration << std::endl;
	duration = (startff - startee) / cv::getTickFrequency() * 1000;
	std::cout << "stepe duration:" << duration << std::endl;*/

}

bool MetaLine::next(int &xSeed,int &ySeed)
{
	int x, y;
	float *ptrM=(float *)maskImage.data;
	uchar *ptrO=orientationMapInt.data;

	int direction=ptrO[ySeed*cols+xSeed];  // 锟斤拷锟斤拷ori
	int direction0=direction-1;    // 锟铰斤拷   torlerance  pi/8 -> 1 pixel 
	if (direction0<0) direction0=15;
	int direction1=direction;      // 锟斤拷锟街?
	int direction2=direction+1;    // 锟较斤拷
	if (direction2==16) direction2=0;

	// 8锟斤拷锟斤拷x,y锟斤拷锟斤拷偏锟斤拷锟斤拷
	static const int X_OFFSET[8] = {  0,  1,  0, -1,  1, -1, -1,  1 };
	static const int Y_OFFSET[8] = {  1,  0, -1,  0,  1,  1, -1, -1 };

	for (size_t i=0; i!=8; ++i)
	{
		x = xSeed + X_OFFSET[i];
		if ((0 <= x) && (x < cols))
		{
			y = ySeed + Y_OFFSET[i];
			if ((0 <= y) && (y < rows))
			{
				if (ptrM[y*cols+x])  // 锟斤拷锟斤拷锟斤拷point锟斤拷没锟叫憋拷锟斤拷锟斤拷
				{
					int directionTemp=ptrO[y*cols+x];  // 取ori
					if (directionTemp==direction0||directionTemp==direction1||directionTemp==direction2) 
					{
						xSeed = x;      
						ySeed = y;
						return true;  // 锟斤拷锟斤拷锟教凤拷围锟斤拷锟斤拷锟斤拷
					}
				}
			}
		}
	}
	return false;  
}

bool MetaLine::smartRouting(clusters_list_t &segments,float minDeviation,int minSize)
{
	const int64 starta = cv::getTickCount();
	int i,j,m,n;
	if (minSize<3) minSize=3;

	cv::Mat maskImageOri=maskImage.clone();
	float* ptrM=(float*)maskImage.data;
	uchar *ptrO=orientationMapInt.data;  // ori map 0~16

	//get the sorted gradient points
	// descent sort gradientValue and gradientPoints by gradientValue
	int numGradientPoints=gradientPoints.size();
	QuickSort<float,cv::Point>::SortDescent(&gradientValue[0], 0, numGradientPoints-1, &gradientPoints[0]);
	
	const int64 startb = cv::getTickCount();

	//find strings
	strings_list_t strings;
	for (i=0;i<numGradientPoints;++i)
	{
		string_t &str = strings.push_back();
		str.clear();

		int count=0;
		int x = gradientPoints[i].x;
		int y = gradientPoints[i].y;
		do  // 没锟斤拷if锟斤拷直锟斤拷do锟斤拷锟斤拷证每锟斤拷锟姐都锟结处锟斤拷锟斤拷
		{
			pixel_t &p = str.push_back();
			p.x=x;
			p.y=y;
			ptrM[y*cols+x]=0;  // 锟斤拷证锟斤拷锟斤拷锟截革拷锟斤拷锟斤拷
		}
		while (next( x, y));

		pixel_t temp;
		for (m=0, n=str.size()-1; m<n; ++m, --n)  // reverse str
		{
			temp = str[m];
			str[m] = str[n];
			str[n] = temp;
		}

		// Find and add feature pixels to the begin of the string.
		x = gradientPoints[i].x;
		y = gradientPoints[i].y;
		if (next( x, y))
		{
			do
			{
				pixel_t &p = str.push_back();
				p.x=x;
				p.y=y;
				ptrM[y*cols+x]=0;   // 锟斤拷止锟斤拷锟斤拷
			}
			while (next( x, y));
		}
		if (str.size()<thMeaningfulLength)  // 去锟斤拷锟斤拷小锟斤拷link edge
			strings.pop_back();
		
	}
	maskImage=maskImageOri;  // 锟街革拷edge map
	const int64 startc = cv::getTickCount();

	//show

#if DEBUG
	cv::Mat stringsImage(rows,cols,CV_8UC3,cv::Scalar(0,0,0));

	/*CvFont font;
	double hScale=1.0;
	double vScale=1.0;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, hScale,vScale,0,1);*/

	std::vector<cv::Scalar> colors(7);
	for (i=0;i<7;++i)
	{
		int R=int(double(rand())/RAND_MAX*255);
		int G=int(double(rand())/RAND_MAX*255);
		int B=int(double(rand())/RAND_MAX*255);

		colors[i]=Scalar(R,G,B);
	}
	for (i=0;i<strings.size();++i)
	{
		int R=colors[(i%7)].val[0];
		int G=colors[(i%7)].val[1];
		int B=colors[(i%7)].val[2];
		for (j=0;j<strings[i].size();++j)
		{
			int x=strings[i][j].x;
			int y=strings[i][j].y;
			*(stringsImage.data+3*(y*cols+x)+0)=R;
			*(stringsImage.data+3*(y*cols+x)+1)=G;
			*(stringsImage.data+3*(y*cols+x)+2)=B;
		}
		char text[100];
		sprintf(text,"%d", i);
		Point mid=Point(strings[i][strings[i].size()/2].x,strings[i][strings[i].size()/2].y);
		cv::putText(stringsImage,text,mid,1,1,colors[i%7],1);
	}
	//imwrite("C:\\12345stringsImage.bmp",stringsImage);

	namedWindow("stringsImage", 0);
 	imshow("stringsImage",stringsImage);
 	cv::waitKey(0);
#endif
	//find segments
	segments.clear();
	for (i=0;i<strings.size();++i)
	{
		const string_t &str = strings[i];
		subDivision(segments, str, 0, strings[i].size()-1, minDeviation, minSize);
	}
#if DEBUG
	int times=1;
	cv::Mat clustersImage(times*rows,times*cols,CV_8UC3,cv::Scalar(0,0,0));
	for (i=0;i<segments.size();++i)
	{
		int R=colors[(i%7)].val[0];
		int G=colors[(i%7)].val[1];
		int B=colors[(i%7)].val[2];
		for (j=0;j<segments[i].size;++j)
		{
			int x=times*segments[i].pixels[j].x;
			int y=times*segments[i].pixels[j].y;
			*(clustersImage.data+3*(y*times*cols+x)+0)=R;
			*(clustersImage.data+3*(y*times*cols+x)+1)=G;
			*(clustersImage.data+3*(y*times*cols+x)+2)=B;
		}

		char text[100];
		sprintf(text,"%d", i);
		Point mid=times*cv::Point(int(segments[i].pixels[segments[i].size/2].x),int(segments[i].pixels[segments[i].size/2].y));
		//cv::putText(clustersImage,text,mid,1,1,colors[i%7],1);
	}
	//imwrite("C:\\12345clustersImage.bmp",clustersImage);

	namedWindow("clustersImage", 0);
	imshow("clustersImage",clustersImage);
 	cv::waitKey(0);
#endif
	return 1;
}

void MetaLine::getMetaLine(clusters_list_t &segments,lines_list_t &metaLines,float sigma)
{
	int i,j;
	float* ptrMaskImage=(float*)maskImage.data;

	//get meta lines
	int numSegments=segments.size();
	for (i=0;i<numSegments;i++)  //  锟斤拷锟斤拷linked edge
	{
		float parameters[4];
		leastSquareFitting(segments[i],parameters,sigma);

		line_t &metaLineTemp=metaLines.push_back();

		float numLines=-float(metaLines.size());  
		for (j=0;j<segments[i].size;++j)
		{
			pixel_t &p=metaLineTemp.points.push_back();
			p.x=segments[i].pixels[j].x;
			p.y=segments[i].pixels[j].y;

			int loc = segments[i].pixels[j].y*cols +segments[i].pixels[j].x;
			ptrMaskImage[loc]=numLines;
		}

		metaLineTemp.ID=-numLines;
		metaLineTemp.dir=parameters[0];  //0 for k<1, 1 for k >=1
		metaLineTemp.k=parameters[1];
		metaLineTemp.b=parameters[2];
		if (!parameters[0])
		{
			// xs锟斤拷ys锟斤拷xe锟斤拷ye锟街憋拷锟斤拷锟斤拷锟斤拷锟街憋拷锟斤拷锟酵队帮拷锟斤拷锟斤拷锟斤拷盏锟斤拷锟斤拷锟绞碉拷锟斤拷锟较碉拷锟斤拷锟斤拷锟?
			metaLineTemp.xs=metaLineTemp.points[0].x;
			metaLineTemp.ys=metaLineTemp.k*metaLineTemp.xs+metaLineTemp.b;
			metaLineTemp.xe=metaLineTemp.points[metaLineTemp.points.size()-1].x;
			metaLineTemp.ye=metaLineTemp.k*metaLineTemp.xe+metaLineTemp.b; 
		}
		else
		{
			metaLineTemp.ys=metaLineTemp.points[0].y;
			metaLineTemp.xs=metaLineTemp.k*metaLineTemp.ys+metaLineTemp.b;
			metaLineTemp.ye=metaLineTemp.points[metaLineTemp.points.size()-1].y;
			metaLineTemp.xe=metaLineTemp.k*metaLineTemp.ye+metaLineTemp.b;
		}
	}
}

void MetaLine::MetaLineDetection(cv::Mat originalImage, float gausSigma, 
	int gausHalfSize, std::vector<std::vector<float> > &lines)
{
	int i,j;
	
	/*step 1*/
	//myTime.PrintDuration("begin");
	const int64 start = cv::getTickCount();
	
	getInformations(originalImage, gausSigma,  gausHalfSize, p);
	//myTime.PrintDuration("step 1 finish");

	/*step 2*/
	//smart routing
	const int64 start2 = cv::getTickCount();
	float minDeviation=2.0;
	int minSize=thMeaningfulLength/2;   // theta_s
	clusters_list_t segments;  //result of edge linking and splitting
	smartRouting(segments,minDeviation,minSize);
	//myTime.PrintDuration("step 2 finish");

	/*step 3*/
	//get initial meta lines
	const int64 start3 = cv::getTickCount();
	lines_list_t metaLines;
	getMetaLine(segments,metaLines,sigma);

	//meta line extending
	int *removal=(int *)malloc(metaLines.size()*sizeof(int));
	memset(removal,0,metaLines.size()*sizeof(int));
    metaLineExtending(metaLines,removal);

	//meta line merging
	//metaLineMerging(metaLines,removal);
	//myTime.PrintDuration("step 3 finish");

	/*step 4*/
	//line validity check
	const int64 start4 = cv::getTickCount();
	lineValidityCheck(metaLines,removal);
	//myTime.PrintDuration("step 4 finish");
	const int64 start5 = cv::getTickCount();
	//get lines
	std::vector<int> ID;
	for (i=0;i<metaLines.size();++i)
	{
		if (!removal[i])
		{
			std::vector<float> lineTemp(5);
			lineTemp[0]=metaLines[i].xs;
			lineTemp[1]=metaLines[i].ys;
			lineTemp[2]=metaLines[i].xe;
			lineTemp[3]=metaLines[i].ye;
			lineTemp[4]=metaLines[i].ID;
			lines.push_back(lineTemp);
		}
	}
    free(removal);

	double duration = (start2 - start) / cv::getTickFrequency() * 1000;
	std::cout << "step1 duration:" << duration << std::endl;
	duration = (start3 - start2) / cv::getTickFrequency() * 1000;
	std::cout << "step2 duration:" << duration << std::endl;
	duration = (start4 - start3) / cv::getTickFrequency() * 1000;
	std::cout << "step3 duration:" << duration << std::endl;
	duration = (start5 - start4) / cv::getTickFrequency() * 1000;
	std::cout << "step4 duration:" << duration << std::endl;
	duration = (start5 - start) / cv::getTickFrequency() * 1000;
	std::cout << "total duration:" << duration << std::endl;
}

int MetaLine::lineMerging(int IDCur,line_t &metaLineCur,size_list_t &lineHyps,lines_list_t &metaLines,float thAngle)
{
	int i,j;
	int numCur=metaLineCur.points.size();

	float angleCur=0.0;
	if (metaLineCur.points[0].x==metaLineCur.points[numCur-1].x)
	{
		angleCur=CV_PI/2.0;
	}
	else
		angleCur=atan(float(metaLineCur.points[0].y-metaLineCur.points[numCur-1].y)/float(metaLineCur.points[0].x-metaLineCur.points[numCur-1].x));

	//meta line merge judge
	std::vector<int> nums;
	std::vector<float> angles;
	for (i=0;i<lineHyps.size();++i)
	{
		int numTemp=metaLines[lineHyps[i]].points.size();
		float angleTemp=0.0;
		if (metaLines[lineHyps[i]].points[0].x==metaLines[lineHyps[i]].points[numTemp-1].x)
		{
			angleTemp=CV_PI/2.0;
		}
		else
			angleTemp=atan(float(metaLines[lineHyps[i]].points[0].y-metaLines[lineHyps[i]].points[numTemp-1].y)/float(metaLines[lineHyps[i]].points[0].x-metaLines[lineHyps[i]].points[numTemp-1].x));
		angles.push_back(angleTemp);
	}

	float angleMin=100;
	int metaIDHyp=0;
	for (i=0;i<lineHyps.size();++i)
	{
		float angleOffset=min((double)abs(angles[i]-angleCur),CV_PI-abs(angles[i]-angleCur));
		if (angleOffset<angleMin)
		{
			angleMin=angleOffset;
			metaIDHyp=lineHyps[i];
		}
	}
	if (angleMin>thAngle)
		return -1;
	else
	{
		//merged the line
		float thDis=4;
		int numHyp=metaLines[metaIDHyp].points.size();
		float k=abs(tan(angleCur));
		if (k>1)
		{
			float dis_s=abs(metaLines[metaIDHyp].points[0].y-metaLineCur.points[numCur-1].y);
			float dis_e=abs(metaLines[metaIDHyp].points[numHyp-1].y-metaLineCur.points[numCur-1].y);
			if (dis_s<dis_e&&dis_s<thDis)
			{
				if ((metaLines[metaIDHyp].points[numHyp-1].y-metaLineCur.points[numCur-1].y)*(metaLineCur.points[0].y-metaLineCur.points[numCur-1].y)<0)
				{
					//merge
					for (i=0;i<numHyp;++i)
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=metaLines[metaIDHyp].points[i].x;
						p.y=metaLines[metaIDHyp].points[i].y;
					}

					return metaIDHyp;
				}
				
			}
			if (dis_e<dis_s&&dis_e<thDis)
			{
				if ((metaLines[metaIDHyp].points[0].y-metaLineCur.points[numCur-1].y)*(metaLineCur.points[0].y-metaLineCur.points[numCur-1].y)<0)
				{
					//merge
					for (i=0;i<numHyp;++i)
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=metaLines[metaIDHyp].points[numHyp-1-i].x;
						p.y=metaLines[metaIDHyp].points[numHyp-1-i].y;
					}

					return metaIDHyp;
				}
				
			}
			return -1;
		}
		else
		{
			float dis_s=abs(metaLines[metaIDHyp].points[0].x-metaLineCur.points[numCur-1].x);
			float dis_e=abs(metaLines[metaIDHyp].points[numHyp-1].x-metaLineCur.points[numCur-1].x);
			if (dis_s<dis_e&&dis_s<thDis)
			{
				if ((metaLines[metaIDHyp].points[numHyp-1].x-metaLineCur.points[numCur-1].x)*(metaLineCur.points[0].x-metaLineCur.points[numCur-1].x)<0)
				{
					//merge
					for (i=0;i<numHyp;++i)
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=metaLines[metaIDHyp].points[i].x;
						p.y=metaLines[metaIDHyp].points[i].y;
					}

					return metaIDHyp;
				}
				
			}
			if (dis_e<dis_s&&dis_e<thDis)
			{
				if ((metaLines[metaIDHyp].points[0].x-metaLineCur.points[numCur-1].x)*(metaLineCur.points[0].x-metaLineCur.points[numCur-1].x)<0)
				{
					//merge
					for (i=0;i<numHyp;++i)
					{
						pixel_t &p=metaLineCur.points.push_back();
						p.x=metaLines[metaIDHyp].points[numHyp-1-i].x;
						p.y=metaLines[metaIDHyp].points[numHyp-1-i].y;
					}

					return metaIDHyp;
				}
				
			}
			return -1;
		}
	}
}

int MetaLine::lineMerging2(int IDCur,line_t &metaLineCur,size_list_t &lineHyps,lines_list_t &metaLines,size_list_t &IDMerged)
{
	int i,j;
	float thDev=2.0;
	float *ptrMI=(float*) maskImage.data;

	float xs=metaLineCur.xs;
	float ys=metaLineCur.ys;
	float xe=metaLineCur.xe;
	float ye=metaLineCur.ye;
	float length=metaLineCur.points.size();

	float xsCur,ysCur,xeCur,yeCur;
	std::vector<int> removal(lineHyps.size(),0);
	for (i=0;i<lineHyps.size();++i)
	{
		xsCur=metaLines[lineHyps[i]].xs;
		ysCur=metaLines[lineHyps[i]].ys;
		xeCur=metaLines[lineHyps[i]].xe;
		yeCur=metaLines[lineHyps[i]].ye;

		float deviation1 = float( abs( ((xsCur - xs) * (ys - ye)) + ((ysCur - ys) * (xe - xs)) ) );
		deviation1/=length;
		float deviation2 = float( abs( ((xeCur - xs) * (ys - ye)) + ((yeCur - ys) * (xe - xs)) ) );
		deviation2/=length;
		if (deviation1+deviation2>2*thDev) removal[i]=1;
	}

	std::vector<int> IDHypsNew;
	std::vector<float> dis;
	for (i=0;i<lineHyps.size();++i)
	{
		if (!removal[i])
		{
			xsCur=metaLines[lineHyps[i]].xs;
			ysCur=metaLines[lineHyps[i]].ys;
			xeCur=metaLines[lineHyps[i]].xe;
			yeCur=metaLines[lineHyps[i]].ye;

			float dis1=sqrt((xs-xsCur)*(xs-xsCur)+(ys-ysCur)*(ys-ysCur));
			float dis2=sqrt((xs-xeCur)*(xs-xeCur)+(ys-yeCur)*(ys-yeCur));
			float dis3=sqrt((xe-xsCur)*(xe-xsCur)+(ye-ysCur)*(ye-ysCur));
			float dis4=sqrt((xe-xeCur)*(xe-xeCur)+(ye-yeCur)*(ye-yeCur));

			float disMin=min(min(dis1,dis2),min(dis3,dis4));

			IDHypsNew.push_back(lineHyps[i]);
			dis.push_back(disMin);
		}
	}

	//merging judge
	if (IDHypsNew.size())
	{
		//sort according to dis
		QuickSort<float,int>::SortAscent(&dis[0], 0, dis.size()-1, &IDHypsNew[0]);

		for (i=0;i<IDHypsNew.size();++i)
		{
			//dis
			xs=metaLineCur.xs;
			ys=metaLineCur.ys;
			xe=metaLineCur.xe;
			ye=metaLineCur.ye;
			length=metaLineCur.points.size();

			xsCur=metaLines[IDHypsNew[i]].xs;
			ysCur=metaLines[IDHypsNew[i]].ys;
			xeCur=metaLines[IDHypsNew[i]].xe;
			yeCur=metaLines[IDHypsNew[i]].ye;
			float lengthCur=metaLines[IDHypsNew[i]].points.size();

			float dis1=sqrt((xs-xsCur)*(xs-xsCur)+(ys-ysCur)*(ys-ysCur));
			float dis2=sqrt((xs-xeCur)*(xs-xeCur)+(ys-yeCur)*(ys-yeCur));
			float dis3=sqrt((xe-xsCur)*(xe-xsCur)+(ye-ysCur)*(ye-ysCur));
			float dis4=sqrt((xe-xeCur)*(xe-xeCur)+(ye-yeCur)*(ye-yeCur));
			float disMin=min(min(dis1,dis2),min(dis3,dis4));

			Point2f pts,pte;
			float lengthMerged=length+lengthCur+disMin;
			float pMerge=(length+lengthCur)/(lengthMerged);
			float NFA=length*lengthCur*pow(pMerge,int(disMin+0.5));
			if (NFA>1)
			{
				std::vector<Point2f> mergedPoints;
				if (disMin==dis1)
				{
					pts=Point2f(xs,ys);
					pte=Point2f(xsCur,ysCur);
					for (j=length-1;j>=0;--j) { mergedPoints.push_back(Point2f(metaLineCur.points[j].x,metaLineCur.points[j].y)) ; }
					for (j=0;j<lengthCur;++j) { mergedPoints.push_back(Point2f(metaLines[IDHypsNew[i]].points[j].x,metaLines[IDHypsNew[i]].points[j].y)); }
				}
				else if (disMin==dis2)
				{
					pts=Point2f(xs,ys);
					pte=Point2f(xeCur,yeCur);
					for (j=length-1;j>=0;--j)     { mergedPoints.push_back(Point2f(metaLineCur.points[j].x,metaLineCur.points[j].y)) ; }
					for (j=lengthCur-1;j>=0;--j)  { mergedPoints.push_back(Point2f(metaLines[IDHypsNew[i]].points[j].x,metaLines[IDHypsNew[i]].points[j].y)); }
				}
				else if (disMin==dis3)
				{
					pts=Point2f(xe,ye);
					pte=Point2f(xsCur,ysCur);
					for (j=0;j<length;++j)     { mergedPoints.push_back(Point2f(metaLineCur.points[j].x,metaLineCur.points[j].y)) ; }
					for (j=0;j<lengthCur;++j)  { mergedPoints.push_back(Point2f(metaLines[IDHypsNew[i]].points[j].x,metaLines[IDHypsNew[i]].points[j].y)); }
				}
				else
				{
					pts=Point2f(xe,ye);
					pte=Point2f(xeCur,yeCur);
					for (j=0;j<length;++j)       { mergedPoints.push_back(Point2f(metaLineCur.points[j].x,metaLineCur.points[j].y)) ; }
					for (j=lengthCur-1;j>=0;--j) { mergedPoints.push_back(Point2f(metaLines[IDHypsNew[i]].points[j].x,metaLines[IDHypsNew[i]].points[j].y)); }
				}

				//line fitting
				float newPara[4];
				float metaLineParas[4];
				if (leastSquareFitting(mergedPoints,metaLineParas,sigma))
				{
					if (crossingCheck( pts, pte, -IDCur-1))
					{
						metaLineCur.dir=metaLineParas[0];  //0 for k<1, 1 for k >=1
						metaLineCur.k=metaLineParas[1];
						metaLineCur.b=metaLineParas[2];
						metaLineCur.dev=metaLineParas[3];

						if (metaLineCur.dir==0)
						{
							metaLineCur.xs=mergedPoints[0].x;
							metaLineCur.ys=metaLineCur.k*metaLineCur.xs+metaLineCur.b;
							metaLineCur.xe=mergedPoints[mergedPoints.size()-1].x;
							metaLineCur.ye=metaLineCur.k*metaLineCur.xe+metaLineCur.b;
						}
						else
						{
							metaLineCur.ys=mergedPoints[0].y;
							metaLineCur.xs=metaLineCur.k*metaLineCur.ys+metaLineCur.b;
							metaLineCur.ye=mergedPoints[mergedPoints.size()-1].y;
							metaLineCur.xe=metaLineCur.k*metaLineCur.ye+metaLineCur.b;
						}
						size_t &temp=IDMerged.push_back();
						temp=IDHypsNew[i];

						metaLineCur.points.resize(mergedPoints.size());
						for (j=0;j<mergedPoints.size();++j)
						{
							metaLineCur.points[j].x=mergedPoints[j].x;
							metaLineCur.points[j].y=mergedPoints[j].y;

							int loc=metaLineCur.points[j].y*cols+metaLineCur.points[j].x;
							ptrMI[loc]=-IDCur-1;
						}
					}
				}
			}
		}
	}

	if (IDHypsNew.size()) return 1;
	else return -1;
}

void MetaLine::metaLineExtending(lines_list_t &metaLines,int *removal)
{
	int i,j,m,n;
	int num=metaLines.size();

	//sort lines
	std::vector<int> length;
	std::vector<int> index;
	for (i=0;i<metaLines.size();++i)
	{
		if (metaLines[i].points.size()>2*thMeaningfulLength)
		{
			index.push_back(i);
			length.push_back(metaLines[i].points.size());
		}
	}
	// 锟斤拷锟斤拷锟饺斤拷锟斤拷锟斤拷锟斤拷
	QuickSort<int,int>::SortDescent(&length[0], 0, length.size()-1, &index[0]);

	for (i=0;i<length.size();++i)
	{
		int t=index[i];
		if (!removal[t])
		{
			int dir=metaLines[t].dir;
			if (!dir)  //horizontal line
			{
				extendHirozontal(metaLines[t],metaLines,removal);

				//the other direction extend
				// reverse container
				for (m=0,n=metaLines[t].points.size()-1;m<n;++m,--n)
				{
					pixel_t temp=metaLines[t].points[m];
					metaLines[t].points[m]=metaLines[t].points[n];
					metaLines[t].points[n]=temp;
				}
				if (!metaLines[t].dir)
					extendHirozontal(metaLines[t],metaLines,removal);
				else
					extendVertical(metaLines[t],metaLines,removal);

			}
			else  //vertical line
			{
				extendVertical(metaLines[t],metaLines,removal);

				//the other direction extend
				for (m=0,n=metaLines[t].points.size()-1;m<n;++m,--n)
				{
					pixel_t temp=metaLines[t].points[m];
					metaLines[t].points[m]=metaLines[t].points[n];
					metaLines[t].points[n]=temp;
				}
				if (!metaLines[t].dir)
					extendHirozontal(metaLines[t],metaLines,removal);
				else
					extendVertical(metaLines[t],metaLines,removal);
			}

			float metaLineParas[4];
			gradientWeightedLeastSquareFitting(metaLines[t].points,metaLineParas,0.5);
			metaLines[t].dir=metaLineParas[0];  //0 for k<1, 1 for k >=1
			metaLines[t].k=metaLineParas[1];
			metaLines[t].b=metaLineParas[2];
			metaLines[t].dev=metaLineParas[3];
			if (metaLines[t].dir)
			{
				metaLines[t].ys=metaLines[t].points[0].y;
				metaLines[t].xs=metaLines[t].k*metaLines[t].ys+metaLines[t].b;
				metaLines[t].ye=metaLines[t].points[metaLines[t].points.size()-1].y;
				metaLines[t].xe=metaLines[t].k*metaLines[t].ye+metaLines[t].b;
			}
			else
			{
				metaLines[t].xs=metaLines[t].points[0].x;
				metaLines[t].ys=metaLines[t].k*metaLines[t].xs+metaLines[t].b;
				metaLines[t].xe=metaLines[t].points[metaLines[t].points.size()-1].x;
				metaLines[t].ye=metaLines[t].k*metaLines[t].xe+metaLines[t].b;
			}
		}
	}
}

void MetaLine::metaLineMerging(lines_list_t &metaLines,int *removal)
{
	int i,j,m,n;

	//sort lines
	std::vector<int> length;
	std::vector<int> index;
	for (i=0;i<metaLines.size();++i)
	{
		if (!removal[i]&&metaLines[i].points.size()>2*thMeaningfulLength)
		{
			index.push_back(i);
			length.push_back(metaLines[i].points.size());
		}
	}
	QuickSort<int,int>::SortDescent(&length[0], 0, length.size()-1, &index[0]);

	//creat line bins
	int numBins=16;
	float stepBins=CV_PI/numBins;
	std::vector<std::vector<int> > bins(numBins);
	std::vector<int> lineBinIndex(length.size(),0);
	std::vector<float> lineAngle(length.size(),0);
	for (i=0;i<length.size();++i)
	{
		int lineID=index[i];
		float deltaX=metaLines[lineID].xe-metaLines[lineID].xs;
		float deltaY=metaLines[lineID].ye-metaLines[lineID].ys;

		float angle=atan(deltaY/deltaX);
		angle+=CV_PI/2.0;
		int binID=angle/stepBins;
		if (binID==numBins) binID=numBins-1;

		bins[binID].push_back(i);
		lineBinIndex[i]=binID;
		lineAngle[i]=angle;
	}

	//line merging
	for (i=0;i<length.size();++i)
	{
		int binID=lineBinIndex[i];

		int lineID1=index[i];
		float angle1=lineAngle[i];
		if (!removal[lineID1])
		{
			if (metaLines[lineID1].ID==265)
			{
				int aa=0;
			}
			if (bins[binID].size())
			{
				size_list_t lineIDHyps;
				for (j=0;j<bins[binID].size();++j)
				{
					int t=bins[binID][j];
					int lineID2=index[t];
					if (lineID2!=lineID1)
					{
						float angle2=lineAngle[t];

						if (abs(angle1-angle2)<thAngle)
						{
							size_t &temp=lineIDHyps.push_back();
							temp=lineID2;
						}
					}
				}

				//line merging
				size_list_t lineIDMerged;
				int Rnt=lineMerging2(lineID1,metaLines[lineID1],lineIDHyps,metaLines,lineIDMerged);
				if (Rnt)
				{
					for (j=0;j<lineIDMerged.size();++j)
						removal[lineIDMerged[j]]=1;	
				}
			}
		}
	}

}

void MetaLine::subDivision(clusters_list_t &clusters, const string_t &string, const size_t first_index, const size_t last_index, const float min_deviation, const size_t min_size)
{
	size_t clusters_count = clusters.size();

	const pixel_t &first = string[first_index];
	const pixel_t &last = string[last_index];
	
	// Compute the length of the straight line segment defined by the endpoints of the cluster.
	int x = first.x - last.x;
	int y = first.y - last.y;
	float length = sqrt( static_cast<float>( (x * x) + (y * y) ) );
	
	// Find the pixels with maximum deviation from the line segment in order to subdivide the cluster.
	size_t max_pixel_index = 0;
	float deviation, max_deviation = -1.0;

	for (size_t i=first_index, count=string.size(); i!=last_index; i=(i+1)%count)
	{
		const pixel_t &current = string[i];
		
		deviation = static_cast<float>( abs( ((current.x - first.x) * (first.y - last.y)) + ((current.y - first.y) * (last.x - first.x)) ) );

		if (deviation > max_deviation)
		{
			max_pixel_index = i;
			max_deviation = deviation;
		}
	}
 	max_deviation /= length;
// 
// 	// Compute the ratio between the length of the segment and the maximum deviation.
// 	float ratio = length / std::max( max_deviation, min_deviation );

	// Test the number of pixels of the sub-clusters.
	int half_min_size=min_size/2;
	if ((max_deviation>=min_deviation) && ((max_pixel_index - first_index + 1) >= half_min_size) && ((last_index - max_pixel_index + 1) >= half_min_size))
	{
		subDivision( clusters, string, first_index, max_pixel_index, min_deviation, min_size );
		subDivision( clusters, string, max_pixel_index, last_index, min_deviation, min_size );

// 		// Test the quality of the sub-clusters against the quality of the current cluster.
// 		if ((ratio1 > ratio) || (ratio2 > ratio))
// 		{
// 			return std::max( ratio1, ratio2 );
// 		}
	}
	else
	{
		// Remove the sub-clusters from the list of clusters.
		clusters.resize( clusters_count );

		// Keep current cluster
		cluster_t &cluster = clusters.push_back();

		cluster.pixels = &first;
		cluster.size = (last_index - first_index) + 1;
	}

}

void MetaLine::lineValidityCheck(lines_list_t &metaLines,int *removal)
{
	int i,j;

	// line Validity Check on Gradient orientation
	int num=0;
	for (i=0;i<metaLines.size();++i)
	{
		if (removal[i]||metaLines[i].points.size()<thMeaningfulLength)
		{
			removal[i]=1;
			num++;
		}
		else
		{
			float orientProbability=lineValidityCheckGradientOrientation( metaLines[i]);
			float gradientProbability=lineValidityCheckGradientLevel( metaLines[i]);
			if (orientProbability*N4*gradientProbability*N2>1)
			{
				removal[i]=1;
				num++;
			}
		}
	}
}

float MetaLine::lineValidityCheckGradientLevel(line_t &metaLines)
{
	int i,j;

	float* ptrGM=(float*) gradientMap.data;
	int thGptNum=thMeaningfulLength;

	std::vector<float> gradient;
	int numPoints=metaLines.points.size();
	int step=numPoints/thGptNum;
	if (step==0) step=1;
	step=1;

	for (j=0;j<numPoints;j+=step)
	{
		int loc=metaLines.points[j].y*cols+metaLines.points[j].x;
		gradient.push_back(ptrGM[loc]);
	}

	//sort
	QuickSort<float,int>::SortDescent(&gradient[0], 0, gradient.size()-1);

	int index=int(gradient[gradient.size()-1]+0.5);
	float probability=pow(greaterThan[index],numPoints);

	return probability;
}

float MetaLine::lineValidityCheckGradientOrientation(line_t &metaLines)
{
	int i,j;
	float* ptr=(float*) orientationMap.data;
	float angleOffset=CV_PI/8.0;

	float deltaX=metaLines.xs-metaLines.xe;
	float deltaY=metaLines.ys-metaLines.ye;
	float angleLine = atan(deltaY/deltaX);
	std::vector<int> alignIndex(metaLines.points.size(),0);
	int count1=0;
	int count2=0;
	int count3=0;
	for (j=0;j<metaLines.points.size();++j)
	{
		int loc=metaLines.points[j].y*cols+metaLines.points[j].x;
		float anglePt=ptr[loc];
		if (abs(anglePt-angleLine)<angleOffset)
			count1++;
		if ((CV_PI-abs(anglePt-angleLine))<angleOffset)
			count2++;
		count3++;
	}

	int count=max(count1,count2);
	
	return probability(count3,count,p);
}