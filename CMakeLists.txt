cmake_minimum_required(VERSION 2.8)
project(mtcnn-light)

set(QMAKE_CXXFLAGS "-std=c++11")
include_directories(/usr/include)
link_directories(/usr/lib)

find_package( OpenCV REQUIRED)

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( mtcnn-light ${DIR_SRCS})
target_link_libraries( mtcnn-light ${OpenCV_LIBS} )
target_link_libraries( mtcnn-light openblas)
