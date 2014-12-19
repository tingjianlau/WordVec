#ifndef OPTIONS_H_
#define OPTIONS_H_

enum ModelType {
  kCBOW = 0x01,
  kSKIP_GRAM = 0x02
};

struct Options {
  int hidden_layer_size;

  int max_sentence_size;

  int thread_num;

  int windows_size;

  ModelType model_type;

  bool use_hierachical_softmax;

  bool use_negative_sampling;

  Options();
};

#endif
