cmake_minimum_required(VERSION 3.14)
project(HawkesHessianBenchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile the standalone benchmark directly
add_executable(benchmark_native benchmark_native.cpp)