/*
 * wordvec.hpp
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

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
    // Initialize synapsis for input layer
    _syn0 = new real[voc.Size() * HIDDEN_LAYER_SIZE];

    for (size_t hid_idx = 0; hid_idx < HIDDEN_LAYER_SIZE; ++hid_idx) {
      for (size_t in_idx = 0; in_idx < voc.Size(); in_idx++) {
        _syn0[in_idx * HIDDEN_LAYER_SIZE + hid_idx] = (rand() / (real) RAND_MAX
            - 0.5) / HIDDEN_LAYER_SIZE;
      }
    }

    // Initialize synapsis for output layer
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
    string word;
    while (!feof(fi)) {
      if (sentence.empty()) {
        // read enough words to consititude a sentence
        while (sentence.size() < MAX_SENTENCE_SIZE && !feof(fi)) {
          Vocabulary::ReadWord(word, fi);
          uint32_t word_idx = voc.GetWordIndex(word);
          if (word_idx == -1)
            continue;
          sentence.push_back(word_idx);
          // TODO: do subsampling to discards frequent words
        }
      }
      memset(neu1, 0, HIDDEN_LAYER_SIZE * sizeof(real));
      memset(neu1e, 0, HIDDEN_LAYER_SIZE * sizeof(real));
      next_rand = next_rand * (uint64_t) 25214903917 + 11;
      if (CBOW) {
        // in -> hidden
      }
    }
  }

  //configuration for wordvec nueral networks
  const size_t MAX_SENTENCE_SIZE;
  const size_t HIDDEN_LAYER_SIZE;

  bool HIERACHICAL_SOFTMAX;
  bool NEGATIVE_SAMPLING;
  bool CBOW;
  bool SKIP_GRAM;

private:
  Vocabulary voc;
  real* _syn0;  //synapsis for input layer
  real* _syn1;  //synapsis for output layer
};

real logit(double x) {
  return exp(x) / (1 + exp(x));
}
