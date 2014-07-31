/*
 * wordvec.hpp
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef WORDVEC_HPP_
#define WORDVEC_HPP_

#include <cstring>

#include "vocabulary.hpp"

typedef float real;

class WordVec {
public:
  WordVec(size_t hidden_layer_size = 100, size_t max_sentence_size = 10) :
      HIDDEN_LAYER_SIZE(hidden_layer_size), MAX_SENTENCE_SIZE(max_sentence_size) {

    HIERACHICAL_SOFTMAX = true;
    NEGATIVE_SAMPLING = false;

    CBOW = true;
    SKIP_GRAM = false;
    InitializeNetwork();
  }

  ~WordVec() {
    delete[] _syn0;
    delete[] _syn1;
  }

  void LoadVocabulary(const string &file_name) {
    FILE* fin = fopen(file_name.c_str(), "r");
    voc.LoadVocabFromTrainFile(fin);
    voc.HuffmanEncoding();
  }

  void InitializeNetwork() {
    // Initialize synapses for input layer
    _syn0 = new real[voc.Size() * HIDDEN_LAYER_SIZE];

    for (size_t hid_idx = 0; hid_idx < HIDDEN_LAYER_SIZE; ++hid_idx) {
      for (size_t in_idx = 0; in_idx < voc.Size(); in_idx++) {
        // use random value (0,1) to initialize the input synapsis
        _syn0[in_idx * HIDDEN_LAYER_SIZE + hid_idx] = (rand() / (real) RAND_MAX
            - 0.5) / HIDDEN_LAYER_SIZE;
      }
    }

    // Initialize synapses for output layer
    if (HIERACHICAL_SOFTMAX) {
      _syn1 = new real[voc.Size() * HIDDEN_LAYER_SIZE * sizeof(real)];
    }
    //TODO: Negative Sampling Network Initialize

  }

  void TrainModelWithFile(const string &file_name, uint64_t &random_seed) {
    int window = 5;
    uint64_t next_rand = random_seed;
    FILE *fi = fopen(file_name.c_str(), "r");
    if (fi == NULL) {
      fprintf(stderr, "No such training file: %s", file_name.c_str());
    }

    // Initialize neure
    real* neu1 = new real[HIDDEN_LAYER_SIZE];
    // Initialize neure error
    real* neu1e = new real[HIDDEN_LAYER_SIZE];

    vector<uint32_t> sentence;
    vector<string> sen;
    string word;
    while (!feof(fi)) {
      //TODO: adatively tune alpha
      sentence.clear();
      if (sentence.empty()) {
        // read enough words to consititude a sentence
        while (sentence.size() < MAX_SENTENCE_SIZE && !feof(fi)) {
          Vocabulary::ReadWord(word, fi);
          sen.push_back(word);
          int word_idx = voc.GetWordIndex(word);
          if (word_idx == -1) {
            continue;
          }
          sentence.push_back(word_idx);
          // TODO: do subsampling to discards frequent words
        }
        printf("sentence:\n");
        for (size_t i = 0; i < sen.size(); ++i) {
          printf("%s ", sen[i].c_str());
        }
        printf("\n\n\n");
      }

      memset(neu1, 0, HIDDEN_LAYER_SIZE * sizeof(real));
      memset(neu1e, 0, HIDDEN_LAYER_SIZE * sizeof(real));
      next_rand = next_rand * (uint64_t) 25214903917 + 11;
      // TODO: curr_window need to random substract
      int window_size = window;

      int n = sentence.size() - 1;
      for (int curr = 0; curr <= n; ++curr) { //iterate every word of sentence
        uint32_t word_predict = sentence[curr];
        // determine sentence windows size
        int w_left = max(0, curr - window_size);
        int w_right = min(n - 1, curr + window_size);
        // in -> hidden
        for (int w = w_left; w <= w_right; ++w) {
          if (w == curr) {
            continue; // if w position equal to curr, skip it
          }
          size_t word_idx = sentence[w];
          for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; hl++) {
            neu1[hl] += _syn0[hl + word_idx * HIDDEN_LAYER_SIZE];
          }
        }
        // hierachical softmax
        if (HIERACHICAL_SOFTMAX) {
          for (size_t code_idx = 0; code_idx < voc[word_predict].code.size();
              ++code_idx) {
            real f = 0;
            size_t out_idx = voc[word_predict].point[code_idx];
            for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; ++hl) {
              f += neu1[hl] * _syn1[hl + out_idx];
            }
            real g = (1 - voc[word_predict].code[code_idx] - f) * alpha;
            for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; ++hl) {
              neu1e[hl] += g * _syn1[hl + out_idx];
            }
            for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; ++hl) {
              _syn1[hl + out_idx] += g * neu1[hl];
            }
          }
        }

        //hidden -> in
        for (int w = w_left; w <= w_right; ++w) {
          if (w == curr) {
            continue; // if w position equal to curr, skip it
          }
          size_t word_idx = sentence[w];
          for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; hl++) {
            _syn0[hl + word_idx * HIDDEN_LAYER_SIZE] += neu1e[hl];
          }
        }
      }
    }
  }

  // logistic function
  real logit(double x) {
    return exp(x) / (1 + exp(x));
  }

  real alpha = 0.0025;
  //configuration for wordvec nueral networks
  const size_t MAX_SENTENCE_SIZE;
  const size_t HIDDEN_LAYER_SIZE;

  bool HIERACHICAL_SOFTMAX;
  bool NEGATIVE_SAMPLING;
  bool CBOW;
  bool SKIP_GRAM;

private:
  Vocabulary voc;
  real* _syn0;  //synapses for input layer
  real* _syn1;  //synapses for output layer
};

#endif /* WORDVEC_HPP_ */
