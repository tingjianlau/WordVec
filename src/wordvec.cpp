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
  int64_t freq;
  int *point;
  string word;
  char* code;
  int codelen;
  Word(string w, int64_t f) :
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
  Vocabulary(size_t min_word_freq = 5) :
      _min_word_freq(min_word_freq) {
    // initialize hash table
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
      if (vocab[last_idx].freq < _min_word_freq) {
        word2pos.erase(vocab[last_idx].word);
        vocab.pop_back();
      } else {
        break;
      }
    }
  }

  void ReadWord(string &word, FILE* fin) {
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

  const size_t _min_word_freq;
};

class WordVec {
public:
  WordVec(int hidden_layer_size) :
      _hidden_layer_size(hidden_layer_size) {

    _hierachical_softmax = true;
    _negative_sampling = false;

    InitializeNetwork();
  }

  ~WordVec() {

  }

  void LoadVocabulary(const string &file_name) {
    FILE* fin = fopen(file_name.c_str(), "r");
    voc.LoadVocabFromTrainFile(fin);
  }

  void InitializeNetwork() {
    posix_memalign((void **) &syn0, 128,
                   voc.Size() * _hidden_layer_size * sizeof(real));
    if (syn0 == NULL) {
      fprintf(stderr, "Memory allocation failed!\n");
      exit(1);
    }

    for (size_t hid_idx = 0; hid_idx < _hidden_layer_size; ++hid_idx) {
      for (size_t in_idx = 0; in_idx < voc.Size(); in_idx++) {
        syn0[in_idx * _hidden_layer_size + hid_idx] = (rand() / (real) RAND_MAX
            - 0.5) / _hidden_layer_size;
      }
    }

    if (_hierachical_softmax) {
      posix_memalign((void **) &syn1, 128,
                     voc.Size() * _hidden_layer_size * sizeof(real));
      if (syn1 == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
      }
    }

    //TODO: Negative Sampling Network Initialize
  }

private:
  Vocabulary voc;
  real* syn0;
  real* syn1;
  real alpha = 0.025;
  real start_alpha;

  bool _hierachical_softmax;
  bool _negative_sampling;

  const int _hidden_layer_size;
};

int main() {
  WordVec wordvec;
  const string file_name = "/Users/Zeyu/word2vec-master/data/text8";
  wordvec.LoadVocabulary(file_name);
}
