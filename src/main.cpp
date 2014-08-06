/*
 * main.cpp
 *
 *  Created on: 2014.7.29
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#include "vocabulary.hpp"
#include "wordvec.hpp"
#include "utils.hpp"
#include "gflags/gflags.h"
#include <omp.h>

using namespace std;

DEFINE_string(input_path, "", "file path of training data");
DEFINE_string(prefix, "data", "file prefix");
DEFINE_int32(thread_num, 4, "multi-thread number");
DEFINE_string(output_path, "word_vector.bin", "word vector model");

int main(int argc, char* argv[]) {

  // use google-flags to parse command line
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);

  printf("======================================================\n");
  printf("|                      WordVec                       |\n");
  printf("======================================================\n");

  // setting maximum number of processor for openMP
  printf("======================OpenMP==========================\n");
  int processor_num = omp_get_num_procs();
  printf("--Available Processor Num = %d\n", processor_num);
  printf("--Set Thread Num = %d\n", FLAGS_thread_num);
  omp_set_num_threads(FLAGS_thread_num);
  printf("======================================================\n");

  const string folder_path = FLAGS_input_path;
  vector<string> files;

  GetAllFiles(folder_path, files, FLAGS_prefix);
  WordVec wordvec;
  wordvec.Train(files);
  wordvec.SaveVector(FLAGS_output_path);
}
