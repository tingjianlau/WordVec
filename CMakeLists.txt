PROJECT(WORDVEC)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
SET(ROOT_PATH "${SRC_PATH}/../")

set(SOURCE_FILES 
  ${SRC_PATH}/utils.cc
  ${SRC_PATH}/vocabulary.cc
  ${SRC_PATH}/options.cc
  ${SRC_PATH}/wordvec.cc
) 

MESSAGE("Application: WordVec")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g3 -fopenmp -O2 -pg")


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


SET(LIBS
  libgflags.a
  libgtest.a
)
SET(EXECUTABLE_OUTPUT_PATH "${ROOT_PATH}/bin")
SET(LIBRARY_OUTPUT_PATH "${ROOT_PATH}/lib")

add_library(wv STATIC ${SOURCE_FILES})

ADD_EXECUTABLE(wordvec ${SRC_PATH}/main.cc)
target_link_libraries(wordvec wv ${LIBS}) 

ADD_EXECUTABLE(distance "${SRC_PATH}/distance.cc")

######################
#######Testing########
######################
set(TEST_SOURCE_FILES
  ${SRC_PATH}/main_test.cc
)
enable_testing()

add_executable(tmain ${TEST_SOURCE_FILES})
target_link_libraries(tmain wv ${LIBS})

add_test(NAME wordvec_test COMMAND tmain)