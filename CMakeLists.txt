# set project's name
PROJECT( CannyLine )

###############################################################################
# CMake settings
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
add_definitions(-std=c++11)
add_definitions(-o3)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# this command finds OpenCV libraries and sets all required variables
set(OpenCV_STATIC ON)


set(OpenCV_DIR  G:/CZCV/czcv_scan/3rd/build_cv/) 
find_package(OpenCV REQUIRED)
#set (OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/image_io/product/lib/cmake/opencv4/)
#find_package (OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
FILE(GLOB_RECURSE SRCS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

#message(STATUS "libraries:${FILE} ")


ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
