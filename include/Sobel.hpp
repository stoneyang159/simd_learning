
//	laviewpbt 共享于2017年7月25日。
//	相关文章见：http://www.cnblogs.com/Imageshop/p/7234463.html
//	绝对对你帮助用请给我买杯咖啡或者啤酒，非常感谢。



// stdafx.h : 标准系统包含文件的包含文件，或是经常使用但不常更改的特定于项目的包含文件

#pragma once					//	这是一个比较常用的C/C++杂注，只要在头文件的最开始加入这条杂注，就能够保证头文件只被编译一次。
//	#pragma once是编译器相关的，有的编译器支持，有的编译器不支持，具体情况请查看编译器API文档，不过现在大部分编译器都有这个杂注了。
#define WIN32_LEAN_AND_MEAN     //  从 Windows 头文件中排除极少使用的信息

#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <intrin.h>  				//	SSE函数需要的库
#include <iostream>

#define IM_STATUS_OK				0
#define IM_STATUS_INVALIDPARAMETER	1
#define IM_STATUS_OUTOFMEMORY		2
#define IM_STATUS_NULLREFRENCE		3
#define IM_STATUS_NOTIMPLEMENTED	4
#define IM_STATUS_NOTSUPPORTED		5
#define IM_STATUS_UNKNOWNERROR		6


#define IM_Max(a, b) (((a) > (b)) ? (a): (b))
#define IM_Min(a, b) (((a) > (b)) ? (b): (a))

int IM_Sobel(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride);
int IM_SobelTable(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride);
int IM_SobelSSE(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride);
