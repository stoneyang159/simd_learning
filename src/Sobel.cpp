#include "Sobel.hpp"


/// <summary>
/// 将整数限幅到字节数据类型。
/// </summary>
inline unsigned char IM_ClampToByte(int Value)			//	现代PC还是这样直接写快些
{
	if (Value < 0)
		return 0;
	else if (Value > 255)
		return 255;
	else
		return (unsigned char)Value;
	//return ((Value | ((signed int)(255 - Value) >> 31)) & ~((signed int)Value >> 31));
}

/// <summary>
/// 将整形的Value值限定在Min和Max内,可取Min或者Max的值。
/// </summary>
int IM_ClampI(int Value, int Min, int Max)
{
	if (Value < Min)
		return Min;
	else if (Value > Max)
		return Max;
	else
		return Value;
}

/// <summary>
/// Sobel边缘检测。
/// </summary>
/// <param name="Src">需要处理的源图像数据。</param>
/// <param name="Dest">处理后的图像数据。</param>
/// <param name="Width">图像的宽度。</param>
/// <param name="Height">图像的高度。</param>
///	<remarks>1、支持In-Place操作。</remarks>

//https://stackoverflow.com/questions/18217269/fast-transposition-of-an-image-and-sobel-filter-optimization-in-c-simd?rq=1
int IM_Sobel(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;
	if ((Src == NULL) || (Dest == NULL))			return IM_STATUS_NULLREFRENCE;
	if ((Width <= 0) || (Height <= 0))				return IM_STATUS_INVALIDPARAMETER;
	if ((Channel != 1) && (Channel != 3))			return IM_STATUS_INVALIDPARAMETER;

	// width + 2 for padding
	// kernel size = 3
	// channel = 1 or 3
	unsigned char *RowCopy = (unsigned char *)malloc((Width + 2) * 3 * Channel);

	if (RowCopy == NULL)	return IM_STATUS_OUTOFMEMORY;

	// pointer of first line, second line, third line 
	unsigned char *First = RowCopy;
	unsigned char *Second = RowCopy + (Width + 2) * Channel;
	unsigned char *Third = RowCopy + (Width + 2) * 2 * Channel;

	// memcpy(void *destin, void *source, unsigned n);
	memcpy(Second, Src, Channel);  // same padding，用第一个channel的灰度做padding 
	memcpy(Second + Channel, Src, Width * Channel);												//	拷贝数据到中间位置
	memcpy(Second + (Width + 1) * Channel, Src + (Width - 1) * Channel, Channel);

	memcpy(First, Second, (Width + 2) * Channel);											//	same padding, 第一行和第二行完全一样

	memcpy(Third, Src + Stride, Channel);													//	拷贝下一行数据
	memcpy(Third + Channel, Src + Stride, Width * Channel);
	memcpy(Third + (Width + 1) * Channel, Src + Stride + (Width - 1) * Channel, Channel);

	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Stride;
		if (Y != 0) // 如果不是第一行
		{
			// b, c, a  <-  a, b, c  后两行挪到前两行
			unsigned char *Temp = First; First = Second; Second = Third; Third = Temp;
		}
		if (Y == Height - 1)
		{
			// 如果是最后一行，第三行是第二行的拷贝
			memcpy(Third, Second, (Width + 2) * Channel);
		}
		else
		{  // 如果不是最后一行，将新的数据拷贝第三行
			memcpy(Third, Src + (Y + 1) * Stride, Channel);
			memcpy(Third + Channel, Src + (Y + 1) * Stride, Width * Channel);		//	由于备份了前面一行的数据，这里即使Src和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Src + (Y + 1) * Stride + (Width - 1) * Channel, Channel);
		}
		if (Channel == 1)
		{ // 如果是单通道
			for (int X = 0; X < Width; X++)
			{
				int GX = First[X] - First[X + 2] + (Second[X] - Second[X + 2]) * 2 + Third[X] - Third[X + 2];
				int GY = First[X] + First[X + 2] + (First[X + 1] - Third[X + 1]) * 2 - Third[X] - Third[X + 2];
				LinePD[X] = IM_ClampToByte(sqrtf(GX * GX + GY * GY + 0.0F));
			}
		}
		else
		{
			for (int X = 0; X < Width * 3; X++)
			{
				int GX = First[X] - First[X + 6] + (Second[X] - Second[X + 6]) * 2 + Third[X] - Third[X + 6];
				int GY = First[X] + First[X + 6] + (First[X + 3] - Third[X + 3]) * 2 - Third[X] - Third[X + 6];
				LinePD[X] = IM_ClampToByte(sqrtf(GX * GX + GY * GY + 0.0F));
			}
		}
	}
	free(RowCopy);
	return IM_STATUS_OK;
}


int IM_SobelTable(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;
	if ((Src == NULL) || (Dest == NULL))			return IM_STATUS_NULLREFRENCE;
	if ((Width <= 0) || (Height <= 0))				return IM_STATUS_INVALIDPARAMETER;
	if ((Channel != 1) && (Channel != 3))			return IM_STATUS_INVALIDPARAMETER;

	unsigned char *RowCopy = (unsigned char *)malloc((Width + 2) * 3 * Channel);
	if (RowCopy == NULL)	return IM_STATUS_OUTOFMEMORY;

	unsigned char *First = RowCopy;
	unsigned char *Second = RowCopy + (Width + 2) * Channel;
	unsigned char *Third = RowCopy + (Width + 2) * 2 * Channel;

	memcpy(Second, Src, Channel);
	memcpy(Second + Channel, Src, Width * Channel);													//	拷贝数据到中间位置
	memcpy(Second + (Width + 1) * Channel, Src + (Width - 1) * Channel, Channel);

	memcpy(First, Second, (Width + 2) * Channel);													//	第一行和第二行一样

	memcpy(Third, Src + Stride, Channel);													//	拷贝第二行数据
	memcpy(Third + Channel, Src + Stride, Width * Channel);
	memcpy(Third + (Width + 1) * Channel, Src + Stride + (Width - 1) * Channel, Channel);

	unsigned char Table[65026];
	for (int Y = 0; Y < 65026; Y++)		Table[Y] = (sqrtf(Y + 0.0f) + 0.5f);

	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Stride;
		if (Y != 0)
		{
			unsigned char *Temp = First; First = Second; Second = Third; Third = Temp;
		}
		if (Y == Height - 1)
		{
			memcpy(Third, Second, (Width + 2) * Channel);
		}
		else
		{
			memcpy(Third, Src + (Y + 1) * Stride, Channel);
			memcpy(Third + Channel, Src + (Y + 1) * Stride, Width * Channel);							//	由于备份了前面一行的数据，这里即使Src和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Src + (Y + 1) * Stride + (Width - 1) * Channel, Channel);
		}
		if (Channel == 1)
		{
			for (int X = 0; X < Width; X++)
			{
				int GX = First[X] - First[X + 2] + (Second[X] - Second[X + 2]) * 2 + Third[X] - Third[X + 2];
				int GY = First[X] + First[X + 2] + (First[X + 1] - Third[X + 1]) * 2 - Third[X] - Third[X + 2];
				LinePD[X] = Table[IM_Min(GX * GX + GY * GY, 65025)];
			}
		}
		else
		{
			for (int X = 0; X < Width * 3; X++)
			{
				int GX = First[X] - First[X + 6] + (Second[X] - Second[X + 6]) * 2 + Third[X] - Third[X + 6];
				int GY = First[X] + First[X + 6] + (First[X + 3] - Third[X + 3]) * 2 - Third[X] - Third[X + 6];
				LinePD[X] = Table[IM_Min(GX * GX + GY * GY, 65025)];
			}
		}
	}
	free(RowCopy);
	return IM_STATUS_OK;
}

int IM_SobelSSE(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride)
{
	int Channel = Stride / Width;
	if ((Src == NULL) || (Dest == NULL))			return IM_STATUS_NULLREFRENCE;
	if ((Width <= 0) || (Height <= 0))				return IM_STATUS_INVALIDPARAMETER;
	if ((Channel != 1) && (Channel != 3))			return IM_STATUS_INVALIDPARAMETER;

	unsigned char *RowCopy = (unsigned char *)malloc((Width + 2) * 3 * Channel);
	if (RowCopy == NULL)	return IM_STATUS_OUTOFMEMORY;

	unsigned char *First = RowCopy;
	unsigned char *Second = RowCopy + (Width + 2) * Channel;
	unsigned char *Third = RowCopy + (Width + 2) * 2 * Channel;

	memcpy(Second, Src, Channel);
	memcpy(Second + Channel, Src, Width * Channel);													//	拷贝数据到中间位置
	memcpy(Second + (Width + 1) * Channel, Src + (Width - 1) * Channel, Channel);

	memcpy(First, Second, (Width + 2) * Channel);													//	第一行和第二行一样

	memcpy(Third, Src + Stride, Channel);													//	拷贝第二行数据
	memcpy(Third + Channel, Src + Stride, Width * Channel);
	memcpy(Third + (Width + 1) * Channel, Src + Stride + (Width - 1) * Channel, Channel);

	int BlockSize = 8, Block = (Width * Channel) / BlockSize;
	unsigned char Table[65026];
	for (int Y = 0; Y < 65026; Y++)		Table[Y] = (sqrtf(Y + 0.0f) + 0.5f);

	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Stride;
		if (Y != 0)
		{
			unsigned char *Temp = First; First = Second; Second = Third; Third = Temp;
		}
		if (Y == Height - 1)
		{
			memcpy(Third, Second, (Width + 2) * Channel);
		}
		else
		{
			memcpy(Third, Src + (Y + 1) * Stride, Channel);
			memcpy(Third + Channel, Src + (Y + 1) * Stride, Width * Channel);							//	由于备份了前面一行的数据，这里即使Src和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Src + (Y + 1) * Stride + (Width - 1) * Channel, Channel);
		}
		if (Channel == 1)
		{
			for (int X = 0; X < Width; X++)
			{
				int GX = First[X] - First[X + 2] + (Second[X] - Second[X + 2]) * 2 + Third[X] - Third[X + 2];
				int GY = First[X] + First[X + 2] + (First[X + 1] - Third[X + 1]) * 2 - Third[X] - Third[X + 2];
				LinePD[X] = Table[IM_Min(GX * GX + GY * GY, 65025)];
			}
		}
		else
		{
			// sse对128位，16个字节的数据进行操作
			// 可同时优化4个float/int，或16个char
			// 函数命名规范：
			//	第一部分:
			//		_mm 为 sse,  _mm256 为avx
			//  第二部分：
			//      操作函数名称 _add _load 或 mul等
			//  第三部分：
			//      操作对象名及数据类型  
			//					 _ps 操作向量中所有的单精度数据
			//					 _pd 操作向量中所有的双精度数据
			//					 _pixx 操作向量中所有的xx位的有符号整型数据，向量寄存器长度为64位
			//					 _epixx 操作向量中所有的xx位的有符号整型数据，向量寄存器长度为128位
			//					 _epuxx 操作向量中所有的xx位的无符号整型数据，向量寄存器长度为128位
			//					 _ss 操作向量中第一个单精度数据
			//					 _si128 操作向量寄存器中第一个128位有符号整型
			// _mm_setzero_si128: sse2
			// Return vector of type __m128i with all elements set to zero.
			__m128i Zero = _mm_setzero_si128();
			for (int X = 0; X < Block * BlockSize; X += BlockSize)
			{
		/*		int GX = First[X] - First[X + 6] + (Second[X] - Second[X + 6]) * 2 + Third[X] - Third[X + 6];
				int GY = First[X] + First[X + 6] + (First[X + 3] - Third[X + 3]) * 2 - Third[X] - Third[X + 6];
				LinePD[X] = Table[IM_Min(GX * GX + GY * GY, 65025)];*/

				// __m128i _mm_loadl_epi64 (__m128i const* mem_addr): sse2
				// 加载p所指向的变量的低64位数据到返回值变量的低64位中，高64位赋值为0；
				// Load 64-bit integer from memory into the first element of dst.

				// __m128i _mm_unpacklo_epi8 (__m128i a, __m128i b)
				// Unpack and interleave 8-bit integers from the low half of a and b, and store the results in dst.
				// 返回值：
				// r0 := a0 ; r1 := b0
				// r2: = a1; r3: = b1
				// ...
				// r14 : = a7; r15: = b7
				__m128i* fp128 = (__m128i *)(First + X);
				__m128i fp064 = _mm_loadl_epi64(fp128);
				__m128i fp08 = _mm_unpacklo_epi8(fp064, Zero);
				std::cout << "fp128:" << fp128 << std::endl;
				std::cout << "fp064:" << &fp064 << std::endl;
				std::cout << "fp08:" << &fp08 << std::endl;
				system("pause");
				exit(0);


				__m128i FirstP0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(First + X)), Zero);
				__m128i FirstP1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(First + X + 3)), Zero);
				__m128i FirstP2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(First + X + 6)), Zero);

				__m128i SecondP0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(Second + X)), Zero);
				__m128i SecondP2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(Second + X + 6)), Zero);

				__m128i ThirdP0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(Third + X)), Zero);
				__m128i ThirdP1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(Third + X + 3)), Zero);
				__m128i ThirdP2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(Third + X + 6)), Zero);

				// GX = (((First[X] - First[X + 6]) + (Second[X] - Second[X + 6]) * 2) + (Third[X] - Third[X + 6]));
				__m128i GX16 = _mm_abs_epi16( // 取绝对值不会影响最后的结果，因为有平方
					_mm_add_epi16(
						_mm_add_epi16(
							_mm_sub_epi16(
								FirstP0,  //  First[X]
								FirstP2   //  First[X + 6]
							), 
							// _mm_slli_epi16(_m128i a, int count)
							// 返回一个_m128i的寄存器，将寄存器a中的8个16bit整数按照count进行相同的逻辑左移
							// 二进制左移n位等于乘2的n次方
							_mm_slli_epi16(  
								_mm_sub_epi16(
									SecondP0,  // Second[X]
									SecondP2   // Second[X + 6]
								), 
								1
							)
						),
						_mm_sub_epi16(
							ThirdP0,    //  Third[X]
							ThirdP2     //  Third[X + 6]
						)
					)
				);
				__m128i GY16 = _mm_abs_epi16(_mm_sub_epi16(_mm_add_epi16(_mm_add_epi16(FirstP0, FirstP2), _mm_slli_epi16(_mm_sub_epi16(FirstP1, ThirdP1), 1)), _mm_add_epi16(ThirdP0, ThirdP2)));


				__m128i GX32L = _mm_unpacklo_epi16(GX16, Zero);
				__m128i GX32H = _mm_unpackhi_epi16(GX16, Zero);
				__m128i GY32L = _mm_unpacklo_epi16(GY16, Zero);
				__m128i GY32H = _mm_unpackhi_epi16(GY16, Zero);

				__m128i ResultL = _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(_mm_add_epi32(_mm_mullo_epi32(GX32L, GX32L), _mm_mullo_epi32(GY32L, GY32L)))));
				__m128i ResultH = _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(_mm_add_epi32(_mm_mullo_epi32(GX32H, GX32H), _mm_mullo_epi32(GY32H, GY32H)))));

				_mm_storel_epi64((__m128i *)(LinePD + X), _mm_packus_epi16(_mm_packus_epi32(ResultL, ResultH), Zero));


				/*__m128i GXYL = _mm_unpacklo_epi16(GX16, GY16);
				__m128i GXYH = _mm_unpackhi_epi16(GX16, GY16);

				_mm_storel_epi64((__m128i *)(LinePD + X), _mm_packus_epi16(_mm_packus_epi32(GXYL, GXYH), Zero));*/
			}
		
			//for (int X = Block * BlockSize; X < Width * 3; X++)
			//{
			//	int GX = First[X] - First[X + 6] + (Second[X] - Second[X + 6]) * 2 + Third[X] - Third[X + 6];
			//	int GY = First[X] + First[X + 6] + (First[X + 3] - Third[X + 3]) * 2 - Third[X] - Third[X + 6];
			//	LinePD[X] = IM_ClampToByte(sqrtf(GX * GX + GY * GY + 0.0F));
			//	//std::cout << "debug LinePD[" << X << " ] :" << LinePD[X] << std::endl;
			//}
		}
	}
	free(RowCopy);
	return IM_STATUS_OK;
}

