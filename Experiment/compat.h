#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>

template<typename T> using thrustDvec = thrust::device_vector<T>;
template<typename T> using thrustHvec = thrust::host_vector<T>;
typedef unsigned long ulong;

namespace cx {
struct timer {
    std::chrono::high_resolution_clock::time_point start;
    timer() : start(std::chrono::high_resolution_clock::now()) {}
    double lap_ms() {
        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - start).count();
        start = now;
        return ms;
    }
};
}
