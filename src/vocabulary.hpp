/*
 * vocabulary.hpp
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */
#ifndef VOCABULARY_HPP_
#define VOCABULARY_HPP_

#include <cstdio>
#include <string>
#include <vector>
#include <queue>
#include <assert.h>
#include <algorithm>
#include <unordered_map>

using namespace std;

struct Word {
  size_t freq;
  vector<size_t> point;
  string word;
  vector<char> code;

  Word(string w, size_t f) :
      freq(f), word(w) {
  }

  bool operator <(const Word &rhs) const {
    return freq > rhs.freq;
  }
};

class Vocabulary {
public:
  // Data structure for huffman tree
  struct HuffmanTreeNode {
    size_t _freq;
    int _parent;
    char _code;
    size_t _idx;

    HuffmanTreeNode(size_t freq, int parent, int idx) :
        _freq(freq), _parent(parent), _idx(idx) {
      _code = 0;
    }

    bool operator <(const HuffmanTreeNode &rhs) const {
      return _freq > rhs._freq;
    }
  };
  const size_t MIN_WORD_FREQ;

  Vocabulary(size_t min_word_freq = 5) :
      MIN_WORD_FREQ(min_word_freq) {
  }

  Word& operator[](size_t index) {
    assert(index >= 0 && index < vocab.size());
    return vocab[index];
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

  void LoadVocabFromTrainFile(const string &file_name) {
    FILE* fin = fopen(file_name.c_str(), "r");
    LoadVocabFromTrainFile(fin);
    fclose(fin);
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

    printf("Vocabulary Size = %lu\nWords in Training File = %lu\n",
           word2pos.size(), train_words);

  }

  void HuffmanEncoding() {
    vector<HuffmanTreeNode> nodes;
    // Heap structure for building huffman tree, min frequency pop out first
    priority_queue<HuffmanTreeNode> heap;
    // Firstly, every word in vocabulary is a huffman tree node
    // Secondly, every tree node should put into a heap
    for (auto iter = vocab.begin(); iter != vocab.end(); ++iter) {
      int node_idx = nodes.size();
      nodes.push_back(HuffmanTreeNode(iter->freq, -1, node_idx));
      heap.push(nodes.back());
    }

    while (!heap.empty()) {
      // retrieve 2 nodes from heap
      auto min_node1 = heap.top();
      heap.pop();
      if (heap.empty()) { //if heap is empty means huffman tree has built
        break;
      }
      auto min_node2 = heap.top();
      heap.pop();

      // merge two minimum frequency nodes to a new huffman tree node
      // currently its parent is -1
      size_t new_node_idx = nodes.size();
      auto new_node = HuffmanTreeNode(min_node1._freq + min_node2._freq, -1,
                                      new_node_idx);
      nodes.push_back(new_node);
      heap.push(new_node);
      // assign Huffman code
      nodes[min_node1._idx]._code = 0;
      nodes[min_node2._idx]._code = 1;
      // assign parent index
      nodes[min_node1._idx]._parent = new_node_idx;
      nodes[min_node2._idx]._parent = new_node_idx;
    }

    // Encoding every word in vocabulary
    for (size_t i = 0; i < vocab.size(); ++i) {
      size_t idx = i;
      // Generate the Huffman code from leaf to root, it's the same as from root to leaf
      // If idx equal to -1 means reach Huffman tree root
      while (idx != -1 && nodes[idx]._parent != -1) {
        vocab[i].code.push_back(nodes[idx]._code);
        // vocab's point is a Huffman code mapping to output layer
        // Huffman coding mapping just reflects the frequency information
        vocab[i].point.push_back(idx % vocab.size());
        idx = nodes[idx]._parent;
      }

      /***************This is a extremely hidden TRICK!*******************
       * When we doing huffman encoding mapping, we need to make sure:
       * every word has a same common output layer node connection!!!
       ******************************************************************/

      vocab[i].point.push_back(vocab.size() - 2);
      reverse(vocab[i].code.begin(), vocab[i].code.end());
      reverse(vocab[i].point.begin(), vocab[i].point.end());
      printf("word=%s freq=%lu code=", vocab[i].word.c_str(), vocab[i].freq);
      for (int j = 0; j < vocab[i].code.size(); ++j) {
        printf("%d", (int) (vocab[i].code[j]));
      }
      printf("\n");
      for (int j = 0; j < vocab[i].point.size(); ++j) {
        printf("%lu ", vocab[i].point[j]);
      }
      printf("\n");
    }
  }

  size_t Size() {
    return vocab.size();
  }

  // Remove low frequency words in vocabulary
  // Moreover, after sorting the vocabulary, the word->index hash need to be rebuilt
  void ReduceVocab() {
    printf("Reducing Vocabulary...\n");
    sort(vocab.begin(), vocab.end());

    while (vocab.back().freq < MIN_WORD_FREQ) {
      vocab.pop_back();
    }
    word2pos.clear();

    for (size_t i = 0; i < vocab.size(); ++i) {
      word2pos[vocab[i].word] = i;
    }
    printf("Recuded Vocabulary Size = %lu\n", vocab.size());
  }

  int GetWordIndex(const string &word) {
    if (word2pos.find(word) != word2pos.end()) {
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
  }

private:
  unordered_map<string, size_t> word2pos;
  vector<Word> vocab;
};
#endif /* VOCABULARY_HPP_ */
