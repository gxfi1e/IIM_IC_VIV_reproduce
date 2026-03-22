cmake_minimum_required(VERSION 3.15)
project(mysolver LANGUAGES C CXX)

add_executable(main2d example.cpp)

find_package(IBAMR 0.16.0 REQUIRED CONFIG)
target_link_libraries(main2d PRIVATE IBAMR::IBAMR2d)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${IBAMR_CXX_FLAGS}")