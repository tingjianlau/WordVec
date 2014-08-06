/*
 * vocabulary.hpp
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string>

using namespace std;

bool StartWith(const string &word, const string &prefix) {
  if (word.size() < prefix.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (word[i] != prefix[i])
      return false;
  }

  return true;
}

void GetAllFiles(const string &folder_path, vector<string> &files,
                 const string &prefix) {
  files.clear();
  struct dirent* ent = NULL;
  DIR *pDir;
  pDir = opendir(folder_path.c_str());
  while ((ent = readdir(pDir)) != NULL) {
    if (ent->d_type == DT_REG) {
      string filename(ent->d_name);
      if (!StartWith(filename, prefix)) {
        continue;
      }
      string file = folder_path + (folder_path.back() == '/' ? "" : "/")
          + filename;
      files.push_back(file);
      printf("%s\n", file.c_str());
    }
  }
}

#endif
