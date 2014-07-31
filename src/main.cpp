/*
 * main.cpp
 *
 *  Created on: 2014.7.29
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <queue>
#include <cmath>
#include <algorithm>
#include <pthread.h>

#include <unordered_map>
#include <vector>
#include "vocabulary.hpp"
#include "wordvec.hpp"

using namespace std;

int main() {
  WordVec wordvec;
  const string file_name = "/Users/Zeyu/WordVec/data/text8";
  wordvec.LoadVocabulary(file_name);
}
