cmake_minimum_required(VERSION 3.21)
project(image_stitching)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "D:\\opencv\\mingw-build\\install")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(image_stitching
        main.cpp
        stitch.h
        stitch.cpp
        classify.h
        classify.cpp
        draw.h
        draw.cpp)


link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(image_stitching ${OpenCV_LIBS})
