cmake_minimum_required(VERSION 3.3)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    NeuralNetwork.c)

add_executable(NeuralNetwork ${SOURCE_FILES})