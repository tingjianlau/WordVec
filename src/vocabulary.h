/*
 * vocabulary.h
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "gflags/gflags.h"
#include "utils.h"

struct Word {
  int freq;
  std::vector<int> output_node_id;
  std::string word;
  std::vector<char> code;

  Word(const std::string &word, size_t freq) :
      freq(freq), word(word) {
  }

  // high frequency word frist
  bool operator <(const Word &rhs) const {
    return freq > rhs.freq;
  }
};

// Data structure for huffman tree
struct HuffmanTreeNode {
  int freq;    // the frequency sum of each node
  int parent;  // if the parent == NOPARENT(-1) means root
  int idx;     // node index
  char code;   // huffman code for each node

  HuffmanTreeNode(int freq, int parent, int idx) :
    freq(freq), parent(parent), idx(idx), code(0) {
  }

  bool operator <(const HuffmanTreeNode &rhs) const {
    return freq > rhs.freq;
  }
};

class Vocabulary {
 public:
  Vocabulary();

  virtual ~Vocabulary();

  Word& operator[](size_t index);

  bool AddWord(const std::string &word);

  static Vocabulary* CreateVocabFromTrainFiles(const std::vector<std::string> &files);

  void HuffmanEncoding();

  size_t Size() const {
    return vocab_.size();
  }

  // Remove low frequency words in vocabulary
  // Moreover, after sorting the vocabulary, the word->index hash need to be rebuild
  void ReduceVocab();

  int GetWordIndex(const std::string &word) const;

  int GetTrainWordCount() const {
    return train_word_count_;
  }

 private:
  std::unordered_map<std::string, int> word2pos_;

  std::vector<Word> vocab_;

  int train_word_count_;
};

#endif // vocabulary.h
