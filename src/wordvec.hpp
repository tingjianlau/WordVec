/*
 * wordvec.hpp
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef WORDVEC_HPP_
#define WORDVEC_HPP_

#include <cstring>
#include <cstdio>

#include "vocabulary.hpp"

typedef float real;

class WordVec {
public:
  WordVec(size_t hidden_layer_size = 200, size_t max_sentence_size = 1000) :
      HIDDEN_LAYER_SIZE(hidden_layer_size), MAX_SENTENCE_SIZE(max_sentence_size) {

    HIERACHICAL_SOFTMAX = true;
    NEGATIVE_SAMPLING = false;

    CBOW = true;
    SKIP_GRAM = false;

    _syn0 = _syn1 = NULL;
  }

  ~WordVec() {
    delete[] _syn0;
    delete[] _syn1;
  }

  void InitializeNetwork() {
    // Initialize synapses for input layer
    _syn0 = new real[voc.Size() * HIDDEN_LAYER_SIZE];

    for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; ++hl) {
      for (size_t il = 0; il < voc.Size(); il++) {
        // use random value (0,1) to initialize the input synapsis
        _syn0[il * HIDDEN_LAYER_SIZE + hl] = (rand() * 1.0 / (real) RAND_MAX)
            / HIDDEN_LAYER_SIZE;
      }
    }

    // Initialize synapses for output layer
    if (HIERACHICAL_SOFTMAX) {
      _syn1 = new real[voc.Size() * HIDDEN_LAYER_SIZE];
      memset(_syn1, 0, voc.Size() * HIDDEN_LAYER_SIZE * sizeof(real));
    }
    //TODO: Negative Sampling Network Initialize
  }

  void Train(const string &train_file_name) {
    voc.LoadVocabFromTrainFile(train_file_name);
    voc.ReduceVocab();
    voc.HuffmanEncoding();

    InitializeNetwork();

    TrainModelWithFile(train_file_name);
  }

  void TrainModelWithFile(const string &file_name) {
    int window = 5;
    real alpha = start_alpha;
    // variable for statistic
    size_t word_count = 0, last_word_count = 0, word_count_actual = 0;
    clock_t start = clock(), now;

    FILE *fi = fopen(file_name.c_str(), "r");
    if (fi == NULL) {
      fprintf(stderr, "No such training file: %s", file_name.c_str());
    }

    // Initialize neure and neure error
    real* neu1 = new real[HIDDEN_LAYER_SIZE];
    real* neu1e = new real[HIDDEN_LAYER_SIZE];

    vector<uint32_t> sentence;
    string sen;
    string word;

    while (!feof(fi)) {
      //TODO: adatively tune alpha
      if (word_count - last_word_count > 10000) {
        word_count_actual += word_count - last_word_count;
        last_word_count = word_count;
        now = clock();
        printf(
            "Alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk\r",
            alpha,
            word_count_actual * 100.0 / (voc.TrainWordCount() + 1),
            word_count_actual * 1.0
                / ((now - start + 1) / CLOCKS_PER_SEC * 1000.0));
        fflush(stdout);
        alpha = start_alpha
            * max(0.001, (1 - word_count_actual * 1.0 / voc.TrainWordCount()));
      }

      sentence.clear();
      sen = "";
      if (sentence.empty()) {
        // read enough words to consititude a sentence
        while (sentence.size() < MAX_SENTENCE_SIZE && !feof(fi)) {
          Vocabulary::ReadWord(word, fi);
          sen += word + " ";
          int word_idx = voc.GetWordIndex(word);
          if (word_idx == -1) {
            continue;
          }
          ++word_count;
          sentence.push_back(word_idx);
          // TODO: do subsampling to discards frequent words
        }

      }

      // TODO: curr_window need to random substract
      int window_size = window;

      int n = sentence.size();
      //iterate every word of sentence
      for (int curr = 0; curr < n; ++curr) {
        // curr points to the word to be predict
        uint32_t word_predict = sentence[curr];

        // determine sentence windows size
        int w_left = max(0, curr - window_size);
        int w_right = min(n - 1, curr + window_size);
        // clear neu1 and neu1e when predict words change
        memset(neu1, 0, HIDDEN_LAYER_SIZE * sizeof(real));
        memset(neu1e, 0, HIDDEN_LAYER_SIZE * sizeof(real));

        //printf("word_to_predict = %ld window_left = %d, windows_right = %d\n", word_predict, w_left, w_right);
        // in -> hidden
        for (int w = w_left; w <= w_right; ++w) {
          if (w == curr) {
            continue; // if w position equal to the word to predict, skip it
          }
          size_t word_idx = sentence[w];
          for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; hl++) {
            neu1[hl] += _syn0[hl + word_idx * HIDDEN_LAYER_SIZE];
          }
        }
        // hierachical softmax
        if (HIERACHICAL_SOFTMAX) {
          //iteratte every huffman code of the word to be predict
          for (size_t code_idx = 0; code_idx < voc[word_predict].code.size();
              ++code_idx) {
            real f = 0;
            size_t output_idx = voc[word_predict].point[code_idx]
                * HIDDEN_LAYER_SIZE;
            for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; ++hl) {
              f += neu1[hl] * _syn1[hl + output_idx];
            }

            f = sigmoid(f);
            real g = (1 - voc[word_predict].code[code_idx] - f) * alpha;
            for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; ++hl) {
              neu1e[hl] += g * _syn1[hl + output_idx];
            }
            for (size_t hl = 0; hl < HIDDEN_LAYER_SIZE; ++hl) {
              _syn1[hl + output_idx] += g * neu1[hl];
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

    delete[] neu1;
    delete[] neu1e;
  }

  void SaveVector(const string &output_file, bool binary = true) {
    FILE* fo = fopen(output_file.c_str(), "wb");
    fprintf(fo, "%lld %lld\n", (long long) voc.Size(),
            (long long) HIDDEN_LAYER_SIZE);
    for (size_t i = 0; i < voc.Size(); ++i) {
      fprintf(fo, "%s ", voc[i].word.c_str());
      for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
        if (binary) {
          fwrite(&_syn0[i * HIDDEN_LAYER_SIZE + j], sizeof(real), 1, fo);
        } else {
          fprintf(fo, "%lf ", _syn0[i * HIDDEN_LAYER_SIZE + j]);
        }
      }
      fprintf(fo, "\n");
    }
  }

  // sigmoid function
  real sigmoid(double x) {
    return exp(x) / (1 + exp(x));
  }

  real start_alpha = 0.025;
  //configuration for wordvec nueral networks
  const size_t HIDDEN_LAYER_SIZE;
  const size_t MAX_SENTENCE_SIZE;

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
