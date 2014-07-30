/*
 * wordvec.cpp
 *
 *  Created on: 2014年7月29日
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cmath>
#include <pthread.h>

#include <unordered_map>
#include <vector>

using namespace std;

typedef float real;

struct Word {
  size_t freq;
  int *point;
  string word;
  char* code;
  int codelen;

  Word(string w, size_t f) :
      freq(f), word(w) {
    point = NULL;
    code = NULL;
    codelen = 0;
  }

  bool operator <(const Word &rhs) const {
    return this->freq > rhs.freq;
  }
};

class Vocabulary {
public:
  const size_t MIN_WORD_FREQ;

  Vocabulary(size_t min_word_freq = 5) :
      MIN_WORD_FREQ(min_word_freq) {
  }

  bool AddWord(const string &word) {
    if (word2pos.find(word) == word2pos.end()) {
      vocab.push_back(Word(word, 1));
      word2pos[word] = vocab.size() - 1;
    } else {
      vocab[word2pos[word]].freq++;
    }

    return true;
  }

  void LoadVocabFromTrainFile(FILE *fin) {
    clock_t start = clock();
    size_t train_words = 0;
    string word;
    while (!feof(fin)) {
      ReadWord(word, fin);
      if (word.size() == 0) {
        continue;
      }
      ++train_words;
      if (train_words % 100000 == 0) {
        printf("process %lu K\r", train_words / 1000);
        fflush(stdout);
      }
      AddWord(word);
    }
    printf("Cost %lf second to load training file\n",
           (clock() * 1.0 - start) / CLOCKS_PER_SEC);

    sort(vocab.begin(), vocab.end());
    ReduceVocab();

    printf("Vocabulary Size = %lu\nWords in Training File = %lu\n",
           word2pos.size(), train_words);
  }

  size_t Size() {
    return vocab.size();
  }

  void ReduceVocab() {
    int last_idx = vocab.size() - 1;
    while (last_idx-- >= 0) {
      if (vocab[last_idx].freq < MIN_WORD_FREQ) {
        word2pos.erase(vocab[last_idx].word);
        vocab.pop_back();
      } else {
        break;
      }
    }
  }

  uint32_t GetWordIndex(const string &word) {
    if (word2pos.find(word) != word2pos.find(word)) {
      return word2pos[word];
    }
    return -1;
  }

  static void ReadWord(string &word, FILE* fin) {
    word.clear();
    char ch;
    while (!feof(fin)) {
      ch = fgetc(fin);
      if (ch == 13 || ch == 9) {
        continue; //skip '\r' and
      }
      if (ch == ' ' || ch == '\t' || ch == '\n') {
        return;
      }
      word.push_back(ch);
    }
    // TODO: Truncate Too Long Word?
  }

private:
  unordered_map<string, uint32_t> word2pos;
  vector<Word> vocab;

};

class WordVec {
public:

  WordVec(int hidden_layer_size = 100, size_t max_sentence_size = 10) :
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
    real* neu1e = new real[HIDDEN_LAYER_SIZE];

    vector<uint32_t> sentence;
    string word;
    while (!feof(fi)) {
      if (sentence.empty()) {
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

      }
    }
  }

  //configure
  const size_t MAX_SENTENCE_SIZE;
  const int HIDDEN_LAYER_SIZE;

  bool HIERACHICAL_SOFTMAX;
  bool NEGATIVE_SAMPLING;
  bool CBOW;
  bool SKIP_GRAM;

private:
  Vocabulary voc;
  real* _syn0;  //synapsis for input layer
  real* _syn1;  //synapsis for output layer
  real _alpha = 0.025;
  real _start_alpha;
};

int main() {
  WordVec wordvec;
  const string file_name = "/Users/Zeyu/word2vec-master/data/text8";
  wordvec.LoadVocabulary(file_name);
}
