/*
 * vocabulary.hpp
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <cstdio>
#include <string>
#include <vector>
#include <queue>
#include <assert.h>
#include <algorithm>
#include <unordered_map>

#include "utils.h"
#include "gflags/gflags.h"


struct Word {
  int freq_;
  std::vector<int> output_node_id_;
  std::string word_;
  std::vector<char> code_;

  Word(std::string word, size_t freq) :
      freq_(freq), word_(word) {
  }

  // high frequency word frist
  bool operator <(const Word &rhs) const {
    return freq_ > rhs.freq_;
  }
};

// Data structure for huffman tree
struct HuffmanTreeNode {
  int freq_;    // the frequency sum of each node
  int parent_;  // if the parent == NOPARENT(-1) means root
  char code_;  // huffman code for each node
  int idx_;   // node index

  HuffmanTreeNode(int freq, int parent, int idx) :
          freq_(freq), parent_(parent), idx_(idx) {
    code_ = 0;
  }

  bool operator <(const HuffmanTreeNode &rhs) const {
    return freq_ > rhs.freq_;
  }
};

DECLARE_int32(min_word_freq);

class Vocabulary {
public:
  Vocabulary();

  Word& operator[](size_t index);

  bool AddWord(const std::string &word);

  void LoadVocabFromTrainFiles(const std::vector<std::string> &files);

  void LoadVocabFromTrainFile(FILE *fin);

  void HuffmanEncoding();

  size_t Size() {
    return vocab_.size();
  }

  // Remove low frequency words in vocabulary
  // Moreover, after sorting the vocabulary, the word->index hash need to be rebuild
  void ReduceVocab();

  size_t GetWordIndex(const std::string &word);

  // Read word by word from text, return true if read end of file(EOF) or '\n'
  static bool ReadWord(std::string &word, FILE* fin);

  int TrainWordCount() {
    return train_word_count_;
  }

private:
  std::unordered_map<std::string, int> word2pos_;
  std::vector<Word> vocab_;
  int train_word_count_;

  const int NO_PARENT = -1;
};
#endif /* VOCABULARY_HPP_ */
