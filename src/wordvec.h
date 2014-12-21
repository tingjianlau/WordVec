/*
 * wordvec.h
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef WORDVEC_H_
#define WORDVEC_H_

#include <cstdio>
#include <memory>

#include "options.h"
#include "utils.h"
#include "vocabulary.h"

class WordVec {
 public:
  WordVec();

  WordVec(const Options &option);

  virtual ~WordVec();

  void Train(const std::vector<std::string> &files);

  void TrainModelWithFile(const std::string &file_name);

  //save the word vector(the input synapses) to file
  void SaveVector(const std::string &output_file, bool binary_format) const;

 private:
  void InitializeNetwork();
  // Training Continous Bag-of-Words model with one sentence, alpha is the learning rate
  void TrainCBOWModel(const std::vector<int> &sentence, real neu1[],
                      real neu1e[], int window_size, real alpha);

  // Training Skip-Gram model with one sentence, alpha is the learning rate
  void TrainSkipGramModel(const std::vector<int> &sentence, real neu1e[],
                          int window_size, real alpha);

  WordVec(const WordVec&);  // no copying!

  void operator=(const WordVec&);  // no copying!

  std::unique_ptr<Vocabulary> voc_;

  real* syn_in_;  //synapses for input layer

  real* syn_out_;  //synapses for output layer

  size_t word_count_total_;

  Options opt_;
};

#endif /* WORDVEC_HPP_ */
