#include "mutils/timer.h"

using namespace ADU;

Timer* Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;

std::string ADU::getCurTime() {
    // get current time
    auto now = std::chrono::system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = std::chrono::system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "MD_%m_%d_HMS_%H_%M_%S"); // HH:MM:SS
    return oss.str();
}