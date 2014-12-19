PROJECT(WORDVEC)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
SET(ROOT_PATH "${SRC_PATH}/../")

#aux_source_directory(./src SOURCE_FILES)
set(SOURCE_FILES 
  ${SRC_PATH}/utils.cc
  ${SRC_PATH}/vocabulary.cc
  ${SRC_PATH}/wordvec.cc
) 
  

MESSAGE("Application: WordVec")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g3 -Wall -fopenmp -fPIC -O2 -pg")


if(APPLE)
  set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
endif()

add_definitions(
  "-lpthread"
  "-fPIC"
  "-Wno-sign-compare"
  "-Wall"
)  

MESSAGE(STATUS "CXX FLAGS: ${CMAKE_CXX_FLAGS}")

INCLUDE_DIRECTORIES(
  ${ROOT_PATH}/include
  ${SRC_PATH}/include
)


SET(LIBS libgflags.a)
SET(EXECUTABLE_OUTPUT_PATH "${ROOT_PATH}/bin")
SET(LIBRARY_OUTPUT_PATH "${ROOT_PATH}/lib")

add_library(wv STATIC ${SOURCE_FILES})

ADD_EXECUTABLE(wordvec ${SRC_PATH}/main.cpp)
target_link_libraries(wordvec wv ${LIBS}) 

ADD_EXECUTABLE(distance "${SRC_PATH}/distance.cpp")
