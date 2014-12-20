#include "vocabulary.h"

using namespace std;

Vocabulary::Vocabulary() : train_word_count_(0) {
}

Word& Vocabulary::operator[](size_t index) {
    CHECK_GE(index, 0);
    CHECK_LT(index, vocab_.size());
    return vocab_[index];
}

bool Vocabulary::AddWord(const string &word) {
    if (word2pos_.find(word) == word2pos_.end()) {
        vocab_.emplace_back(Word(word, 1));
        word2pos_[word] = vocab_.size() - 1;
    } else {
        vocab_[word2pos_[word]].freq++;
    }

    return true;
}

void Vocabulary::LoadVocabFromTrainFiles(const vector<string> &files) {
    clock_t start = clock();
    for (int i = 0; i < files.size(); ++i) {
        printf("loading %s\n", files[i].c_str());
        FILE* fin = fopen(files[i].c_str(), "r");
        LoadVocabFromTrainFile(fin);
        fclose(fin);
    }

    printf("Cost %lf second to load training file\n",
            (clock() * 1.0 - start) / CLOCKS_PER_SEC);

    printf("Vocabulary Size = %lu\nWords in Training File = %d\n",
            word2pos_.size(), train_word_count_);

}

void Vocabulary::LoadVocabFromTrainFile(FILE *fin) {
    string word;
    while (!feof(fin)) {
        ReadWord(word, fin);
        if (word.size() == 0) {
            continue;
        }
        ++train_word_count_;

        if (train_word_count_ % 100000 == 0) {
            printf("process %d K words\r", train_word_count_ / 1000);
            fflush(stdout);
        }

        AddWord(word);
    }
}

// This is a clearer implementation of building Huffman Tree than google
// word2vec
void Vocabulary::HuffmanEncoding() {
    vector<HuffmanTreeNode> nodes;
    // Heap structure for building huffman tree, min frequency pop out first
    priority_queue<HuffmanTreeNode> heap;
    // Firstly, every word in vocabulary is a huffman tree node
    // Secondly, every tree node should put into a heap
    for (auto iter = vocab_.begin(); iter != vocab_.end(); ++iter) {
        int node_idx = nodes.size();
        nodes.emplace_back(HuffmanTreeNode(iter->freq, kNoParent, node_idx));
        heap.push(nodes.back());
    }

    while (!heap.empty()) {
        // retrieve 2 nodes from heap every time
        auto min_node1 = heap.top();
        heap.pop();
        if (heap.empty()) { //if heap is empty means huffman tree has built
            break;
        }
        auto min_node2 = heap.top();
        heap.pop();

        // merge two minimum frequency nodes to a new huffman tree node
        // at first its parent is -1
        int new_node_idx = nodes.size();
        auto new_node = HuffmanTreeNode(min_node1.freq + min_node2.freq, kNoParent,
                new_node_idx);
        nodes.push_back(new_node);

        heap.push(new_node);
        // assign Huffman code
        nodes[min_node1.idx].code = 0;
        nodes[min_node2.idx].code = 1;
        // assign parent index
        nodes[min_node1.idx].parent = new_node_idx;
        nodes[min_node2.idx].parent = new_node_idx;
    }
    nodes.back().code = 1;  // assign the huffman ROOT code
    // encoding every word in vocabulary
    int root_index = nodes.back().idx;
    for (int i = 0; i < vocab_.size(); ++i) {
        int idx = i;
        // Generate the Huffman code from leaf to root, it's the same as from
        // root to leaf. If idx equal to -1 means reach Huffman tree root
        while (nodes[idx].parent != kNoParent) {
            vocab_[i].code.push_back(nodes[idx].code);
            // vocab's point is a Huffman code mapping to output layer
            // Huffman coding mapping just reflects the frequency information
            vocab_[i].output_node_id.push_back(idx % vocab_.size());
            idx = nodes[idx].parent;
        }
        /***************Below is a hidden TRICK!***************************
        * When we doing huffman encoding mapping, we need to make sure:
        * every word's huffman code must contains the huffman tree root!
        * if you loss the mapping of huffman tree root, the result is terrible!!
        ******************************************************************/
        vocab_[i].output_node_id.push_back(root_index % vocab_.size());  // TRICK!

        reverse(vocab_[i].code.begin(), vocab_[i].code.end());
        reverse(vocab_[i].output_node_id.begin(), vocab_[i].output_node_id.end());
    }
}

// Remove low frequency words in vocabulary
// Moreover, after sorting the vocabulary, the word->index hash need to be rebuild
void Vocabulary::ReduceVocab() {
    printf("Reducing Vocabulary...\n");
    sort(vocab_.begin(), vocab_.end());

    while (vocab_.back().freq < FLAGS_min_word_freq) {
        vocab_.pop_back();
    }
    word2pos_.clear();
    // rebuild word->index mapping
    for (int i = 0; i < vocab_.size(); ++i) {
        word2pos_[vocab_[i].word] = i;
    }
    printf("Recuded Vocabulary Size = %lu\n", vocab_.size());
}

int Vocabulary::GetWordIndex(const string &word) {
    if (word2pos_.find(word) != word2pos_.end()) {
        return word2pos_[word];
    }

    return -1;
}

