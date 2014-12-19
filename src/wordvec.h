/*
 * wordvec.h
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef WORDVEC_H_
#define WORDVEC_H_

#include <cstring>
#include <cstdio>
#include <omp.h>

#include "vocabulary.h"
#include "options.h"
#include "utils.h"

using namespace std;

class WordVec {
 public:

  WordVec(Options option);

  ~WordVec();

  void InitializeNetwork();

  void Train(const std::vector<std::string> &files);


  // Training Continous Bag-of-Words model with one sentence, alpha is the learning rate
  void TrainCBOWModel(const vector<int> &sentence, real neu1[],
                      real neu1e[], int window_size, real alpha);

  // Training Skip-Gram model with one sentence, alpha is the learning rate
  void TrainSkipGramModel(const vector<int> &sentence, real neu1e[],
                          int window_size, real alpha);

  void TrainModelWithFile(const string &file_name);

  //save the word vector(the input synapses) to file
  void SaveVector(const string &output_file, bool binary_format);

 private:
  // sigmoid function
  real Sigmoid(double x) {
    return exp(x) / (1 + exp(x));
  }

  // configuration for word vector nueral networks
  int HIDDEN_LAYER_SIZE;
  int MAX_SENTENCE_SIZE;

  bool HIERACHICAL_SOFTMAX;
  bool NEGATIVE_SAMPLING;

  ModelType model_type_;
  const real start_alpha_ = 0.025;
  Vocabulary voc_;
  real* syn_in_;  //synapses for input layer
  real* syn_out_;  //synapses for output layer

  size_t word_count_total_;
  int thread_num_;

  Options opt_;
};

#endif /* WORDVEC_HPP_ */
