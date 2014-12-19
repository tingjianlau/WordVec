/*
 * main.cpp
 *
 *  Created on: 2014.7.29
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#include <omp.h>

#include "vocabulary.h"
#include "wordvec.h"
#include "gflags/gflags.h"

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

// check whether a string is start with given prefix
bool StartWith(const std::string &word, const std::string &prefix) {
  if (prefix.size() == 0) {
    return true;
  }
  if (word.size() < prefix.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (word[i] != prefix[i])
      return false;
  }

  return true;
}

// a simple file operator to get all the files under given folder
void GetAllFiles(const std::string &folder_path, std::vector<std::string> &files,
                 const std::string &prefix) {
  files.clear();
  struct dirent* ent = NULL;
  DIR *pDir;
  pDir = opendir(folder_path.c_str());
  while ((ent = readdir(pDir)) != NULL) {
    if (ent->d_type == DT_REG) {
      std::string filename(ent->d_name);
      if (!StartWith(filename, prefix)) {
        continue;
      }
      char last_ch = folder_path.back();
      std::string file = folder_path + (last_ch == '/' ? "" : "/") + filename;
      files.push_back(file);
    }
  }
}

int main(int argc, char* argv[]) {

  // use google-flags to parse command line
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);

  printf("======================================================\n");
  printf("|                      WordVec                       |\n");
  printf("======================================================\n");

  // setting maximum number of processor for OpenMP
  printf("======================OpenMP==========================\n");
  int processor_num = omp_get_num_procs();
  printf("--Available Processor Num = %d\n", processor_num);
  printf("--Set Thread Num = %d\n", FLAGS_threads);
  omp_set_num_threads(FLAGS_threads);
  printf("======================================================\n");

  printf("output path = %s\n", FLAGS_output.c_str());
  printf("training path = %s\n", FLAGS_train.c_str());

  // Read all files in training data folder
  vector<string> files;
  GetAllFiles(FLAGS_train, files, FLAGS_prefix);

  // Select model type
  WordVec::ModelType model_type = WordVec::ModelType::CBOW;
  if (FLAGS_skipgram) {
    model_type = WordVec::ModelType::SKIP_GRAM;
  }
  WordVec wordvec(FLAGS_hidden_size, FLAGS_sentence_size, model_type, FLAGS_threads);

  // Training word vector
  wordvec.Train(files);

  // Save word vector model
  wordvec.SaveVector(FLAGS_output);
}
