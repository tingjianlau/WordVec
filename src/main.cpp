/*
 * main.cpp
 *
 *  Created on: 2014.7.29
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#include "vocabulary.hpp"
#include "wordvec.hpp"

using namespace std;

int main() {
  WordVec wordvec;
  const string file_name = "/Users/Zeyu/WordVec/data/text8";
  wordvec.LoadVocabulary(file_name);
  uint64_t seed = 10;
  wordvec.TrainModelWithFile(file_name, seed);
}
