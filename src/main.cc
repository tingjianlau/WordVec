/*
 * main.cpp
 *
 *  Created on: 2014.7.29
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#include <omp.h>

#include "gflags/gflags.h"
#include "utils.h"
#include "vocabulary.h"
#include "wordvec.h"

using namespace std;

DEFINE_string(train, "", "file path of training data");
DEFINE_string(prefix, "", "file prefix");
DEFINE_int32(threads, 4, "multi-thread number");
DEFINE_string(output, "word_vector.bin", "word vector model output");
DEFINE_int32(hidden_size, 100, "neural num of hidden layers");
DEFINE_int32(window, 5, "sliding window size");
DEFINE_bool(cbow, true, "use Continuous Bag of Words model for training");
DEFINE_bool(skipgram, false, "use Skip-Gram model to train");
DEFINE_int32(sentence_size, 1000, "max sentence length");
DEFINE_int32(min_word_freq, 5, "the minimum word frequecy in vocabulary");
DEFINE_int32(iter, 1, "iteration for training the corpus");

namespace {

bool PopulateOptions(Options &options) {
  CHECK_EQ(FLAGS_cbow, FLAGS_skipgram);
  options.model_type = ModelType::kCBOW;
  if (FLAGS_skipgram) {
    options.model_type = ModelType::kSkipGram;
  }
  options.hidden_layer_size = FLAGS_hidden_size;
  options.max_sentence_size = FLAGS_sentence_size;
  options.thread_num = FLAGS_threads;
  options.windows_size = FLAGS_window;
  options.use_hierachical_softmax = true;
  options.use_negative_sampling = false;
  options.iter = FLAGS_iter;

  LOG(INFO) << "iter = " << options.iter;
  LOG(INFO) << "hidden_layer_size = " << options.hidden_layer_size;
  LOG(INFO) << "max_sentence_size = " << options.max_sentence_size;
  LOG(INFO) << "thread_num = " << options.thread_num;
  LOG(INFO) << "windows_size = " << options.windows_size;
  LOG(INFO) << "use_hierachical_softmax = " << options.use_hierachical_softmax;
  LOG(INFO) << "use_negative_sampling = " << options.use_negative_sampling;

  return true;
}

} // namespace

int main(int argc, char* argv[]) {

  // use google-flags to parse command line
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);

  printf("======================================================\n");
  printf("|                      WordVec                       |\n");
  printf("======================================================\n");

  // setting maximum number of processor for OpenMP
  printf("======================OpenMP==========================\n");
  const int processor_num = omp_get_num_procs();
  printf("--Available Processor Num = %d\n", processor_num);
  printf("--Set Thread Num = %d\n", FLAGS_threads);
  omp_set_num_threads(FLAGS_threads);
  printf("======================================================\n");

  printf("output path = %s\n", FLAGS_output.c_str());
  printf("training path = %s\n", FLAGS_train.c_str());

  // Read all files in training data folder
  vector<string> files;
  GetAllFiles(FLAGS_train, files, FLAGS_prefix);

  // Fill in wordvec options
  Options options;
  PopulateOptions(options);

  WordVec wordvec(options);

  // Training word vector
  wordvec.Train(files);

  // Save word vector model
  wordvec.SaveVector(FLAGS_output, true);
}
