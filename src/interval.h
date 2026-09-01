// interval.h - a simple [min, max] range, used for ray-t clipping and color
// clamping. Small utility class but it keeps intersection code readable.
#pragma once

#include <algorithm>
#include <limits>

class interval {
public:
    double min, max;

    interval() : min(+std::numeric_limits<double>::infinity()),
                 max(-std::numeric_limits<double>::infinity()) {}
    interval(double min_, double max_) : min(min_), max(max_) {}

    double size() const { return max - min; }
    bool contains(double x) const { return min <= x && x <= max; }
    bool surrounds(double x) const { return min < x && x < max; }
    double clamp(double x) const { return std::clamp(x, min, max); }

    static const interval empty, universe;
};

inline const interval interval::empty =
    interval(+std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());
inline const interval interval::universe =
    interval(-std::numeric_limits<double>::infinity(), +std::numeric_limits<double>::infinity());
