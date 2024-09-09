#include "mutils/timer.h"

using namespace ADU;

Timer* Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;