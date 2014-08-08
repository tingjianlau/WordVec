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

DEFINE_string(train, "", "file path of training data");
DEFINE_string(prefix, "", "file prefix");
DEFINE_int32(threads, 4, "multi-thread number");
DEFINE_string(output, "word_vector.bin", "word vector model");
DEFINE_int32(hidden_size, 100, "neural num of hidden layers");
DEFINE_int32(window, 5, "sliding window size");
DEFINE_bool(cbow, true, "use Continuous Bag of Words model to train");
DEFINE_bool(skipgram, false, "use Skip-Gram model to train");
DEFINE_int32(sentence_size, 1000, "max sentence length");

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
  printf("--Set Thread Num = %d\n", FLAGS_threads);
  omp_set_num_threads(FLAGS_threads);
  printf("======================================================\n");

  printf("output path = %s\n", FLAGS_output.c_str());
  printf("input path = %s\n", FLAGS_train.c_str());

  // Read all files in training data folder
  vector<string> files;
  GetAllFiles(FLAGS_train, files, FLAGS_prefix);

  // Training word vector
  WordVec::ModelType model_type = WordVec::ModelType::CBOW;
  if (FLAGS_skipgram) {
    model_type = WordVec::ModelType::SKIP_GRAM;
  }
  WordVec wordvec(FLAGS_hidden_size, FLAGS_sentence_size, model_type, FLAGS_threads);
  wordvec.Train(files);
  wordvec.SaveVector(FLAGS_output);
}
