cmake_minimum_required(VERSION 3.10)
project(WaveSim3)

find_package(OpenCL REQUIRED)
find_package(SFML COMPONENTS system window graphics REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(WaveSim3 main.cpp)

target_link_libraries(WaveSim3 ${OpenCL_LIBRARIES} ${SFML_LIBRARIES})