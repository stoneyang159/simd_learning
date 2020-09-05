#ifndef __MY_TIME_H__
#define __MY_TIME_H__
#pragma once


#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>  

using namespace std;

class MyTime {
public:
	std::chrono::steady_clock::time_point now, past;

public:
	MyTime(){
		this->now = std::chrono::steady_clock::now();
		this->past = std::chrono::steady_clock::now();
	};
	~MyTime() {};

	std::chrono::duration<double> getDuration();
	std::chrono::steady_clock::time_point getCurrTime();
	void getTimeStamp(char* prefix="");
	void PrintDuration(char* prefix);
};

extern MyTime myTime;

#endif // __MY_TIME_H__