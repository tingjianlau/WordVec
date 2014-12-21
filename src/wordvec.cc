#include "wordvec.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <omp.h>

using namespace std;

namespace {
const real start_alpha_ = 0.025;

inline real Sigmoid(double x) {
  return exp(x) / (1 + exp(x));
}
}

WordVec::WordVec() {
  syn_in_ = syn_out_ = nullptr;
  word_count_total_ = 0;
}

WordVec::WordVec(const Options &options) : opt_(options) {
  syn_in_ = syn_out_ = nullptr;
  word_count_total_ = 0;
}

WordVec::~WordVec() {
  delete[] syn_in_;
  delete[] syn_out_;
}

void WordVec::InitializeNetwork() {
  CHECK(voc_ != nullptr);
  // Initialize synapses for input layer
  syn_in_ = new real[voc_->Size() * opt_.hidden_layer_size];

  for (int h = 0; h < opt_.hidden_layer_size; ++h) {
    for (int xi = 0; xi < voc_->Size(); xi++) {
      // use random value (0,1) to initialize the input synapses
      syn_in_[xi * opt_.hidden_layer_size + h] = RandReal();
    }
  }

  // Initialize synapses for output layer
  if (opt_.use_hierachical_softmax) {
    syn_out_ = new real[voc_->Size() * opt_.hidden_layer_size];
    memset(syn_out_, 0, voc_->Size() * opt_.hidden_layer_size * sizeof(real));
  }
// TODO: Negative Sampling Network Initialize
// Negative Samlpling is one of the trick of word2vec, but will not improve
// the result greatly. Turning negative sampling off is he default config
}

void WordVec::Train(const vector<string> &files) {
  //loading vocabulary needs to read all files
  voc_.reset(Vocabulary::CreateVocabFromTrainFiles(files));
  voc_->ReduceVocab();
  voc_->HuffmanEncoding();

  InitializeNetwork();
  word_count_total_ = 0;
  double start = omp_get_wtime();
  // iterate the corpus
  for (int epoch = 0; epoch < opt_.iter; ++epoch) {
#pragma omp parallel for
    for (int i = 0; i < files.size(); ++i) {
      TrainModelWithFile(files[i]);
    }
  }
  double cost_time = omp_get_wtime() - start;
  printf("Training Time: %lf sec\n", cost_time);
  printf("Training Speed: words/thread/sec: %.1fk\n",
      voc_->GetTrainWordCount() / cost_time / opt_.thread_num / 1000);
}

// Training Continous Bag-of-Words model with one sentence, alpha is the learning rate
void WordVec::TrainCBOWModel(const vector<int> &sentence, real neu1[],
    real neu1e[], int window_size, real alpha) {
  CHECK(voc_ != nullptr);
  CHECK(syn_in_ != nullptr);
  CHECK(syn_out_ != nullptr);

  int sentence_len = sentence.size();
  //iterate every word in a sentence
  for (int w_target_idx = 0; w_target_idx < sentence_len; ++w_target_idx) {
    // curr points to the word to be predict
    int target_word = sentence[w_target_idx];

    // determine sentence windows range w_left and w_right
    int w_left = max(0, w_target_idx - window_size);
    int w_right = min(sentence_len - 1, w_target_idx + window_size);
    // clear neu1 and neu1e when predicted words change
    memset(neu1, 0, opt_.hidden_layer_size * sizeof(real));
    memset(neu1e, 0, opt_.hidden_layer_size * sizeof(real));

    // update from input layer -> hidden layer
    for (int w = w_left; w <= w_right; ++w) {
      if (w == w_target_idx) {
        continue; // if w position equal to the target word index, skip it
      }
      int xi = sentence[w] * opt_.hidden_layer_size;
      for (int h = 0; h < opt_.hidden_layer_size; h++) {
        neu1[h] += syn_in_[h + xi];
      }
    }
    // Hierachical softmax
    if (opt_.use_hierachical_softmax) {
      // iterate every Huffman code of the word to be predict
      for (int c_idx = 0; c_idx < (*voc_)[target_word].code.size(); ++c_idx) {
        real f = 0;
        int xo = (*voc_)[target_word].output_node_id[c_idx] * opt_.hidden_layer_size;
        for (int h = 0; h < opt_.hidden_layer_size; ++h) {
          f += neu1[h] * syn_out_[h + xo];
        }

        f = Sigmoid(f);
        //real gradient = (1 - _voc[target_word].code[c_idx] - f) ;
        real gradient = (*voc_)[target_word].code[c_idx] - f;
        for (int h = 0; h < opt_.hidden_layer_size; ++h) {
          neu1e[h] += alpha * gradient * syn_out_[h + xo];
        }
        for (int h = 0; h < opt_.hidden_layer_size; ++h) {
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
      int word_idx = sentence[w];
      for (int h = 0; h < opt_.hidden_layer_size; h++) {
        syn_in_[h + word_idx * opt_.hidden_layer_size] += neu1e[h];
      }
    }
  }
}

// Training Skip-Gram model with one sentence, alpha is the learning rate
void WordVec::TrainSkipGramModel(const vector<int> &sentence, real neu1e[],
    int window_size, real alpha) {
  CHECK(voc_ != nullptr);
  CHECK(syn_in_ != nullptr);
  CHECK(syn_out_ != nullptr);

  int sentence_len = sentence.size();
  //iterate every word in sentence
  for (int w_input_idx = 0; w_input_idx < sentence_len; ++w_input_idx) {
    int word_input = sentence[w_input_idx];

    int xi = word_input * opt_.hidden_layer_size;
    // determine sentence windows range w_left and w_right
    int w_left = max(0, w_input_idx - window_size);
    int w_right = min(sentence_len - 1, w_input_idx + window_size);
    // clear neu1 and neu1e when predict words change

    for (int w = w_left; w <= w_right; ++w) {
      if (w == w_input_idx) {
        continue; // if w position equal to the target word index, skip it
      }
      memset(neu1e, 0, opt_.hidden_layer_size * sizeof(real));
      int target_word = sentence[w];

      // hierachical softmax
      if (opt_.use_hierachical_softmax) {
        // iterate every Huffman code of the word to be predict
        for (int c_idx = 0; c_idx < (*voc_)[target_word].code.size();
             ++c_idx) {
          real f = 0;
          int xo = (*voc_)[target_word].output_node_id[c_idx] * opt_.hidden_layer_size;
          for (int h = 0; h < opt_.hidden_layer_size; ++h) {
            f += syn_in_[h + xi] * syn_out_[h + xo];
          }

          f = Sigmoid(f);
          // the gradient formular for word2vec
          real gradient = (1 - (*voc_)[target_word].code[c_idx] - f);
          for (int h = 0; h < opt_.hidden_layer_size; ++h) {
            neu1e[h] += alpha * gradient * syn_out_[h + xo];
          }
          for (int h = 0; h < opt_.hidden_layer_size; ++h) {
            syn_out_[h + xo] += alpha * gradient * syn_in_[h + xi];
          }
        }
      }
      // hidden -> input
      for (int h = 0; h < opt_.hidden_layer_size; ++h)
        syn_in_[h + xi] += neu1e[h];
    }
  }
}

void WordVec::TrainModelWithFile(const string &file_name) {
  int window = 5;
  real alpha = start_alpha_;
  // variable for statistic
  int word_count_curr_thread = 0, last_word_count_curr_thread = 0;
  FILE *fi = fopen(file_name.c_str(), "r");
  FileCloser fcloser(fi);
  if (fi == NULL) {
    LOG(FATAL) << "No such training file: " << file_name << endl;
  }

  // Initialize neuron and neuron error
  real* neu1 = new real[opt_.hidden_layer_size];
  real* neu1e = new real[opt_.hidden_layer_size];

  vector<int> sentence;
  string word;

  int train_word_total = voc_->GetTrainWordCount() * opt_.iter;

  while (!feof(fi)) {
    if (word_count_curr_thread - last_word_count_curr_thread > 10000) {
#pragma omp critical (word_count)
      {
        word_count_total_ += word_count_curr_thread
            - last_word_count_curr_thread;
      }
      last_word_count_curr_thread = word_count_curr_thread;
      printf("Alpha: %f  Progress: %.2f%%\r", alpha,
          word_count_total_ * 100.0 / (train_word_total + 1));
      fflush(stdout);

      // decay alpha according to training progress
      alpha = start_alpha_
          * max(0.001,
          (1 - word_count_total_ * 1.0 / train_word_total));
    }

    sentence.clear();
    if (sentence.empty()) {
      // read enough words to consititude a sentence
      while (sentence.size() < opt_.max_sentence_size && !feof(fi)) {
        bool eol = ReadWord(word, fi);
        int word_idx = voc_->GetWordIndex(word);
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
    if (opt_.model_type == kCBOW) {
      TrainCBOWModel(sentence, neu1, neu1e, window, alpha);
    } else if (opt_.model_type == kSkipGram) {
      TrainSkipGramModel(sentence, neu1, window, alpha);
    }
  }

  delete[] neu1;
  delete[] neu1e;
}

//save the word vector(the input synapses) to file
void WordVec::SaveVector(const string &output_file, bool binary_format = true) const {
  // TODO: Check succeed to open file or not.
  FILE* fo = fopen(output_file.c_str(), "wb");
  FileCloser fcloser(fo);
  fprintf(fo, "%lld %lld\n", (long long) voc_->Size(),
      (long long) opt_.hidden_layer_size);
  for (int i = 0; i < voc_->Size(); ++i) {
    fprintf(fo, "%s ", (*voc_)[i].word.c_str());
    for (int j = 0; j < opt_.hidden_layer_size; ++j) {
      if (binary_format) {
        fwrite(&syn_in_[i * opt_.hidden_layer_size + j], sizeof(real), 1, fo);
      } else {
        fprintf(fo, "%lf ", syn_in_[i * opt_.hidden_layer_size + j]);
      }
    }
    fprintf(fo, "\n");
  }
}
