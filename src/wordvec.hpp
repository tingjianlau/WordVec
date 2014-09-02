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
#include <omp.h>
#include "vocabulary.hpp"

typedef float real;

class WordVec {
public:
  enum ModelType {
    CBOW, SKIP_GRAM
  };

  // NOTE: max_sentence_size determines the cache size
  // Default model is CBOW for it's faster then Skip-Gram
  WordVec(size_t hidden_layer_size = 100, size_t max_sentence_size = 1000,
          ModelType model_type = CBOW, int thread_num = 4) :
      HIDDEN_LAYER_SIZE(hidden_layer_size), MAX_SENTENCE_SIZE(max_sentence_size) {

    HIERACHICAL_SOFTMAX = true;
    NEGATIVE_SAMPLING = false;

    _syn0 = _syn1 = NULL;
    _model_type = model_type;
    _word_count_all_threads = 0;
    _thread_num = thread_num;
  }

  ~WordVec() {
    delete[] _syn0;
    delete[] _syn1;
  }

  void InitializeNetwork() {
    // Initialize synapses for input layer
    _syn0 = new real[_voc.Size() * HIDDEN_LAYER_SIZE];

    for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
      for (size_t xi = 0; xi < _voc.Size(); xi++) {
        // use random value (0,1) to initialize the input synapses
        _syn0[xi * HIDDEN_LAYER_SIZE + h] = (rand() * 1.0 / (real) RAND_MAX)
            / HIDDEN_LAYER_SIZE;
      }
    }

    // Initialize synapses for output layer
    if (HIERACHICAL_SOFTMAX) {
      _syn1 = new real[_voc.Size() * HIDDEN_LAYER_SIZE];
      memset(_syn1, 0, _voc.Size() * HIDDEN_LAYER_SIZE * sizeof(real));
    }
    // TODO: Negative Sampling Network Initialize
    // Negative Samlpling is one of the trick of word2vec, but will not change the result greatly
    // Turn negative sampling off is he default configuration of training word2vec, so here not implemented
  }

  void Train(const vector<string> &files) {
    //loading vocabulary needs to read all files
    _voc.LoadVocabFromTrainFiles(files);
    _voc.ReduceVocab();
    _voc.HuffmanEncoding();

    InitializeNetwork();
    _word_count_all_threads = 0;
    double start = omp_get_wtime();
#pragma omp parallel for
    for (size_t i = 0; i < files.size(); ++i) {
      TrainModelWithFile(files[i]);
    }
    double cost_time = omp_get_wtime() - start;
    printf("Training Time: %lf sec\n", cost_time);
    printf("Training Speed: words/thread/sec: %.1fk\n",
           _voc.TrainWordCount() / cost_time / _thread_num / 1000);
  }

  // Training Continous Bag-of-Words model with one sentence, alpha is learning rate
  void TrainCBOWModel(const vector<uint32_t> &sentence, real neu1[],
                      real neu1e[], int window_size, real alpha) {
    int sentence_len = sentence.size();
    //iterate every word in a sentence
    for (int w_target_idx = 0; w_target_idx < sentence_len; ++w_target_idx) {
      // curr points to the word to be predict
      uint32_t target_word = sentence[w_target_idx];

      // determine sentence windows range w_left and w_right
      int w_left = max(0, w_target_idx - window_size);
      int w_right = min(sentence_len - 1, w_target_idx + window_size);
      // clear neu1 and neu1e when predicted words change
      memset(neu1, 0, HIDDEN_LAYER_SIZE * sizeof(real));
      memset(neu1e, 0, HIDDEN_LAYER_SIZE * sizeof(real));

      // in -> hidden
      for (int w = w_left; w <= w_right; ++w) {
        if (w == w_target_idx) {
          continue; // if w position equal to the target word index, skip it
        }
        size_t xi = sentence[w] * HIDDEN_LAYER_SIZE;
        for (size_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
          neu1[h] += _syn0[h + xi];
        }
      }
      // Hierachical softmax
      if (HIERACHICAL_SOFTMAX) {
        // iterate every Huffman code of the word to be predict
        for (size_t c_idx = 0; c_idx < _voc[target_word].code.size(); ++c_idx) {
          real f = 0;
          size_t xo = _voc[target_word].point[c_idx] * HIDDEN_LAYER_SIZE;
          for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
            f += neu1[h] * _syn1[h + xo];
          }

          f = Sigmoid(f);
          //real gradient = (1 - _voc[target_word].code[c_idx] - f) ;
          real gradient = _voc[target_word].code[c_idx] - f;
          for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
            neu1e[h] += alpha * gradient * _syn1[h + xo];
          }
          for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
            _syn1[h + xo] += alpha * gradient * neu1[h];
          }
        }
      }
      // TODO: Negative Sampling
      // hidden -> in
      for (int w = w_left; w <= w_right; ++w) {
        if (w == w_target_idx) {
          continue; // if w position equal to curr, skip it
        }
        size_t word_idx = sentence[w];
        for (size_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
          _syn0[h + word_idx * HIDDEN_LAYER_SIZE] += neu1e[h];
        }
      }
    }
  }

  // Training Skip-Gram with one sentence, alpha is learning rate
  void TrainSkipGramModel(const vector<uint32_t> &sentence, real neu1e[],
                          int window_size, real alpha) {
    int sentence_len = sentence.size();
    //iterate every word of sentence
    for (int w_input_idx = 0; w_input_idx < sentence_len; ++w_input_idx) {
      uint32_t word_input = sentence[w_input_idx];

      size_t xi = word_input * HIDDEN_LAYER_SIZE;
      // determine sentence windows range w_left and w_right
      int w_left = max(0, w_input_idx - window_size);
      int w_right = min(sentence_len - 1, w_input_idx + window_size);
      // clear neu1 and neu1e when predict words change

      for (int w = w_left; w <= w_right; ++w) {
        if (w == w_input_idx) {
          continue; // if w position equal to the target word index, skip it
        }
        memset(neu1e, 0, HIDDEN_LAYER_SIZE * sizeof(real));
        size_t target_word = sentence[w];

        // hierachical softmax
        if (HIERACHICAL_SOFTMAX) {
          // iterate every Huffman code of the word to be predict
          for (size_t c_idx = 0; c_idx < _voc[target_word].code.size();
              ++c_idx) {
            real f = 0;
            size_t xo = _voc[target_word].point[c_idx] * HIDDEN_LAYER_SIZE;
            for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
              f += _syn0[h + xi] * _syn1[h + xo];
            }

            f = Sigmoid(f);
            real gradient = (1 - _voc[target_word].code[c_idx] - f);
            for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
              neu1e[h] += alpha * gradient * _syn1[h + xo];
            }
            for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
              _syn1[h + xo] += alpha * gradient * _syn0[h + xi];
            }
          }
        }
        // hidden -> input
        for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h)
          _syn0[h + xi] += neu1e[h];
      }
    }
  }

  void TrainModelWithFile(const string &file_name) {
    int window = 5;
    real alpha = _start_alpha;
    // variable for statistic
    size_t word_count_curr_thread = 0, last_word_count_curr_thread = 0;
    FILE *fi = fopen(file_name.c_str(), "r");
    if (fi == NULL) {
      fprintf(stderr, "No such training file: %s", file_name.c_str());
    }

    // Initialize neure and neure error
    real* neu1 = new real[HIDDEN_LAYER_SIZE];
    real* neu1e = new real[HIDDEN_LAYER_SIZE];

    vector<uint32_t> sentence;
    string word;

    while (!feof(fi)) {
      if (word_count_curr_thread - last_word_count_curr_thread > 10000) {
#pragma omp critical (word_count)
        {
          _word_count_all_threads += word_count_curr_thread
              - last_word_count_curr_thread;
        }
        last_word_count_curr_thread = word_count_curr_thread;
        printf("Alpha: %f  Progress: %.2f%%\r", alpha,
               _word_count_all_threads * 100.0 / (_voc.TrainWordCount() + 1));
        fflush(stdout);
        alpha = _start_alpha
            * max(0.001,
                  (1 - _word_count_all_threads * 1.0 / _voc.TrainWordCount()));
      }

      sentence.clear();
      if (sentence.empty()) {
        // read enough words to consititude a sentence
        while (sentence.size() < MAX_SENTENCE_SIZE && !feof(fi)) {
          bool eol = Vocabulary::ReadWord(word, fi);
          int word_idx = _voc.GetWordIndex(word);
          if (word_idx == -1) {
            continue;
          }
          ++word_count_curr_thread;
          sentence.push_back(word_idx);
          // TODO: do subsampling to discards high-frequent words
          // This is another trick of word2vec, but will not influence the final result
          if (eol) {
            break;
          }
        }
      }
      // finish read sentence
      if (_model_type == CBOW) {
        TrainCBOWModel(sentence, neu1, neu1e, window, alpha);
      } else if (_model_type == SKIP_GRAM) {
        TrainSkipGramModel(sentence, neu1, window, alpha);
      }
    }

    delete[] neu1;
    delete[] neu1e;
  }

  //save the word vector(the input synapses) to file
  void SaveVector(const string &output_file, bool binary_format = true) {
    FILE* fo = fopen(output_file.c_str(), "wb");
    fprintf(fo, "%lld %lld\n", (long long) _voc.Size(),
            (long long) HIDDEN_LAYER_SIZE);
    for (size_t i = 0; i < _voc.Size(); ++i) {
      fprintf(fo, "%s ", _voc[i].word.c_str());
      for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
        if (binary_format) {
          fwrite(&_syn0[i * HIDDEN_LAYER_SIZE + j], sizeof(real), 1, fo);
        } else {
          fprintf(fo, "%lf ", _syn0[i * HIDDEN_LAYER_SIZE + j]);
        }
      }
      fprintf(fo, "\n");
    }
  }

  // configuration for word vector nueral networks
  const size_t HIDDEN_LAYER_SIZE;
  const size_t MAX_SENTENCE_SIZE;

  bool HIERACHICAL_SOFTMAX;
  bool NEGATIVE_SAMPLING;

private:
  // sigmoid function
  real Sigmoid(double x) {
    return exp(x) / (1 + exp(x));
  }

  ModelType _model_type;
  real _start_alpha = 0.025;
  Vocabulary _voc;
  real* _syn0;  //synapses for input layer
  real* _syn1;  //synapses for output layer
  size_t _word_count_all_threads;
  int _thread_num;
}
;

#endif /* WORDVEC_HPP_ */
