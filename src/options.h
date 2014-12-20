#ifndef OPTIONS_H_
#define OPTIONS_H_

enum ModelType {
  kCBOW = 0x01,
  kSkipGram = 0x02
};

struct Options {
  int hidden_layer_size;

  int max_sentence_size;

  int thread_num;

  int windows_size;

  int iter;

  ModelType model_type;

  bool use_hierachical_softmax;

  bool use_negative_sampling;

  Options();
};

#endif // options.h
