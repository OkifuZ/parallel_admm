#pragma once

#include "mutils/cformat.h"

#include <chrono>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <map>
#include <algorithm>


namespace ADU {

    std::string getCurTime();

    class Timer {
    public:
        using ClockType = std::chrono::high_resolution_clock;

        struct Record {
            std::string tag;
            int us;

            Record(std::string&& tag_, int us_)
                : tag(std::move(tag_)), us(us_) {}
        };

    private:
        static Timer* current;
        static std::vector<Record> records;

        Timer* parent = nullptr;
        ClockType::time_point beg;
        ClockType::time_point end;
        std::string tag;

        Timer(std::string_view&& tag_, ClockType::time_point&& beg_) : parent(current), beg(beg_)
            , tag(current ? current->tag + " => " + (std::string)tag_ : tag_)
        {
            current = this;
        }

        void _destroy(ClockType::time_point&& end) {
            current = parent;
            auto diff = end - beg;
            int us = std::chrono::duration_cast
                <std::chrono::milliseconds>(diff).count();
            //<std::chrono::microseconds>(diff).count();
            records.emplace_back(std::move(tag), us);
        }

    public:
        Timer(std::string_view tag_) : Timer(std::move(tag_), ClockType::now()) {}
        ~Timer() { _destroy(ClockType::now()); }

        static auto const& getRecords() { return records; }
        static std::string getLog() {
            if (records.size() == 0) {
                return "";
            }

            std::string res;

            struct Statistic {
                int max_us = 0;
                int min_us = 0;
                int total_us = 0;
                int count_rec = 0;
            };
            std::map<std::string, Statistic> stats;
            for (auto const& [tag, us] : records) {
                auto& stat = stats[tag];
                stat.total_us += us;
                stat.max_us = std::max(stat.max_us, us);
                stat.min_us = !stat.count_rec ? us : std::min(stat.min_us, us);
                stat.count_rec++;
            }

            std::vector<std::pair<std::string, Statistic>> sortstats(stats.begin(), stats.end());
            std::sort(sortstats.begin(), sortstats.end(),
                [&](auto const& lhs, auto const& rhs) {
                    return lhs.second.total_us > rhs.second.total_us;
                });

            res += "   avg   |   min   |   max   |  total  | cnt | tag\n";
            for (auto const& [tag, stat] : sortstats) {
                res += cformat("%9d|%9d|%9d|%9d|%5d| %s\n",
                    stat.total_us / stat.count_rec,
                    stat.min_us, stat.max_us, stat.total_us,
                    stat.count_rec, tag.c_str());
            }
            return res;
        }
    };

}