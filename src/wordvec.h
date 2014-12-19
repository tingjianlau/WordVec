/*
 * wordvec.hpp
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
#include "utils.h"

using namespace std;

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

    syn_in_ = syn_out_ = NULL;
    model_type_ = model_type;
    word_count_total_ = 0;
    thread_num_ = thread_num;
  }

  ~WordVec() {
    delete[] syn_in_;
    delete[] syn_out_;
  }

  void InitializeNetwork() {
    // Initialize synapses for input layer
    syn_in_ = new real[voc_.Size() * HIDDEN_LAYER_SIZE];

    for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
      for (size_t xi = 0; xi < voc_.Size(); xi++) {
        // use random value (0,1) to initialize the input synapses
        syn_in_[xi * HIDDEN_LAYER_SIZE + h] = RandReal();
      }
    }

    // Initialize synapses for output layer
    if (HIERACHICAL_SOFTMAX) {
      syn_out_ = new real[voc_.Size() * HIDDEN_LAYER_SIZE];
      memset(syn_out_, 0, voc_.Size() * HIDDEN_LAYER_SIZE * sizeof(real));
    }
    // TODO: Negative Sampling Network Initialize
    // Negative Samlpling is one of the trick of word2vec, but will not improve
    // the result greatly. Turning negative sampling off is he default 
    // configuration of training word2vec, so here not implemented
  }

  void Train(const vector<string> &files) {
    //loading vocabulary needs to read all files
    voc_.LoadVocabFromTrainFiles(files);
    voc_.ReduceVocab();
    voc_.HuffmanEncoding();

    InitializeNetwork();
    word_count_total_ = 0;
    double start = omp_get_wtime();
#pragma omp parallel for
    for (size_t i = 0; i < files.size(); ++i) {
      TrainModelWithFile(files[i]);
    }
    double cost_time = omp_get_wtime() - start;
    printf("Training Time: %lf sec\n", cost_time);
    printf("Training Speed: words/thread/sec: %.1fk\n",
           voc_.TrainWordCount() / cost_time / thread_num_ / 1000);
  }

  // Training Continous Bag-of-Words model with one sentence, alpha is the learning rate
  void TrainCBOWModel(const vector<int> &sentence, real neu1[],
                      real neu1e[], int window_size, real alpha) {
    int sentence_len = sentence.size();
    //iterate every word in a sentence
    for (int w_target_idx = 0; w_target_idx < sentence_len; ++w_target_idx) {
      // curr points to the word to be predict
      int target_word = sentence[w_target_idx];

      // determine sentence windows range w_left and w_right
      int w_left = max(0, w_target_idx - window_size);
      int w_right = min(sentence_len - 1, w_target_idx + window_size);
      // clear neu1 and neu1e when predicted words change
      memset(neu1, 0, HIDDEN_LAYER_SIZE * sizeof(real));
      memset(neu1e, 0, HIDDEN_LAYER_SIZE * sizeof(real));

      // update from input layer -> hidden layer
      for (int w = w_left; w <= w_right; ++w) {
        if (w == w_target_idx) {
          continue; // if w position equal to the target word index, skip it
        }
        size_t xi = sentence[w] * HIDDEN_LAYER_SIZE;
        for (size_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
          neu1[h] += syn_in_[h + xi];
        }
      }
      // Hierachical softmax
      if (HIERACHICAL_SOFTMAX) {
        // iterate every Huffman code of the word to be predict
        for (size_t c_idx = 0; c_idx < voc_[target_word].code_.size(); ++c_idx) {
          real f = 0;
          size_t xo = voc_[target_word].output_node_id_[c_idx] * HIDDEN_LAYER_SIZE;
          for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
            f += neu1[h] * syn_out_[h + xo];
          }

          f = Sigmoid(f);
          //real gradient = (1 - _voc[target_word].code[c_idx] - f) ;
          real gradient = voc_[target_word].code_[c_idx] - f;
          for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
            neu1e[h] += alpha * gradient * syn_out_[h + xo];
          }
          for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
            syn_out_[h + xo] += alpha * gradient * neu1[h];
          }
        }
      }
      // TODO: Negative Sampling
      // update from hidden layer -> input layer
      for (int w = w_left; w <= w_right; ++w) {
        if (w == w_target_idx) {
          continue; // if w position equal to curr, skip it
        }
        size_t word_idx = sentence[w];
        for (size_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
          syn_in_[h + word_idx * HIDDEN_LAYER_SIZE] += neu1e[h];
        }
      }
    }
  }

  // Training Skip-Gram model with one sentence, alpha is the learning rate
  void TrainSkipGramModel(const vector<int> &sentence, real neu1e[],
                          int window_size, real alpha) {
    int sentence_len = sentence.size();
    //iterate every word in sentence
    for (int w_input_idx = 0; w_input_idx < sentence_len; ++w_input_idx) {
      int word_input = sentence[w_input_idx];

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
          for (size_t c_idx = 0; c_idx < voc_[target_word].code_.size();
              ++c_idx) {
            real f = 0;
            size_t xo = voc_[target_word].output_node_id_[c_idx] * HIDDEN_LAYER_SIZE;
            for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
              f += syn_in_[h + xi] * syn_out_[h + xo];
            }

            f = Sigmoid(f);
            // the gradient formular for word2vec
            real gradient = (1 - voc_[target_word].code_[c_idx] - f);
            for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
              neu1e[h] += alpha * gradient * syn_out_[h + xo];
            }
            for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h) {
              syn_out_[h + xo] += alpha * gradient * syn_in_[h + xi];
            }
          }
        }
        // hidden -> input
        for (size_t h = 0; h < HIDDEN_LAYER_SIZE; ++h)
          syn_in_[h + xi] += neu1e[h];
      }
    }
  }

  void TrainModelWithFile(const string &file_name) {
    int window = 5;
    real alpha = start_alpha_;
    // variable for statistic
    size_t word_count_curr_thread = 0, last_word_count_curr_thread = 0;
    FILE *fi = fopen(file_name.c_str(), "r");
    if (fi == NULL) {
      LOG(FATAL) << "No such training file: " << file_name << endl;
    }

    // Initialize neuron and neuron error
    real* neu1 = new real[HIDDEN_LAYER_SIZE];
    real* neu1e = new real[HIDDEN_LAYER_SIZE];

    vector<int> sentence;
    string word;

    while (!feof(fi)) {
      if (word_count_curr_thread - last_word_count_curr_thread > 10000) {
#pragma omp critical (word_count)
        {
          word_count_total_ += word_count_curr_thread
              - last_word_count_curr_thread;
        }
        last_word_count_curr_thread = word_count_curr_thread;
        printf("Alpha: %f  Progress: %.2f%%\r", alpha,
               word_count_total_ * 100.0 / (voc_.TrainWordCount() + 1));
        fflush(stdout);

        // decay alpha according to training progress
        alpha = start_alpha_
            * max(0.001,
                  (1 - word_count_total_ * 1.0 / voc_.TrainWordCount()));
      }

      sentence.clear();
      if (sentence.empty()) {
        // read enough words to consititude a sentence
        while (sentence.size() < MAX_SENTENCE_SIZE && !feof(fi)) {
          bool eol = Vocabulary::ReadWord(word, fi);
          int word_idx = voc_.GetWordIndex(word);
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
      if (model_type_ == CBOW) {
        TrainCBOWModel(sentence, neu1, neu1e, window, alpha);
      } else if (model_type_ == SKIP_GRAM) {
        TrainSkipGramModel(sentence, neu1, window, alpha);
      }
    }

    delete[] neu1;
    delete[] neu1e;
  }

  //save the word vector(the input synapses) to file
  void SaveVector(const string &output_file, bool binary_format = true) {
    FILE* fo = fopen(output_file.c_str(), "wb");
    fprintf(fo, "%lld %lld\n", (long long) voc_.Size(),
            (long long) HIDDEN_LAYER_SIZE);
    for (size_t i = 0; i < voc_.Size(); ++i) {
      fprintf(fo, "%s ", voc_[i].word_.c_str());
      for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
        if (binary_format) {
          fwrite(&syn_in_[i * HIDDEN_LAYER_SIZE + j], sizeof(real), 1, fo);
        } else {
          fprintf(fo, "%lf ", syn_in_[i * HIDDEN_LAYER_SIZE + j]);
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

  ModelType model_type_;
  const real start_alpha_ = 0.025;
  Vocabulary voc_;
  real* syn_in_;  //synapses for input layer
  real* syn_out_;  //synapses for output layer

  size_t word_count_total_;
  int thread_num_;
};

#endif /* WORDVEC_HPP_ */
