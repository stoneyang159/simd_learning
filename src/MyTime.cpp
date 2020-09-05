#include "MyTime.h"

MyTime myTime;

std::chrono::steady_clock::time_point MyTime::getCurrTime() {
	now = std::chrono::steady_clock::now();
	return now;
};

std::chrono::duration<double> MyTime::getDuration() {
	now = std::chrono::steady_clock::now();
	std::chrono::duration<double> dura = std::chrono::duration_cast<std::chrono::duration<double>>(now - past);
	past = now;
	return dura;
};

void MyTime::PrintDuration(char* prefix) {
	std::chrono::duration<double> dura = getDuration();
	std::cout << prefix << ",cost time is : " << dura.count()<<std::endl;
};

void MyTime::getTimeStamp(char* prefix) {
	//定义毫秒级别的时钟类型
	//typedef std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> microClock_type;
	//获取当前时间点，windows system_clock是100纳秒级别的(不同系统不一样，自己按照介绍的方法测试)，所以要转换
	//microClock_type tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
	//转换为ctime.用于打印显示时间
	//time_t tt = std::chrono::system_clock::to_time_t(tp);
	//char _time[50];
	//ctime_s(_time, sizeof(_time), &tt);
	//std::cout << prefix << ",now time is : " << _time<<std::endl;
};
