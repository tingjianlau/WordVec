/*
 * options.cc
 *
 *  Created on: 2014.7.29
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#include "options.h"

Options::Options()
    : hidden_layer_size(100),
      max_sentence_size(1000),
      thread_num(4),
      iter(1),
      model_type(ModelType::kCBOW),
      use_hierachical_softmax(true),
      use_negative_sampling(false) {
}


