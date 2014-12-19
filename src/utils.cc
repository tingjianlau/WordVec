#include "utils.h"

#include <cmath>

using namespace std;

char kSegmentFaultCauser[] = "Used to cause artificial segmentation fault";

bool StartWith(const std::string &word, const std::string &prefix) {
  if (prefix.size() == 0) {
    return true;
  }
  if (word.size() < prefix.size()) {
    return false;
  }
  for (int i = 0; i < prefix.size(); ++i) {
    if (word[i] != prefix[i])
      return false;
  }

  return true;
}

// a simple file operator to get all the files under given folder
void GetAllFiles(const std::string &folder_path, std::vector<std::string> &files,
    const std::string &prefix) {
  files.clear();
  struct dirent* ent = NULL;
  DIR *pDir;
  pDir = opendir(folder_path.c_str());
  while ((ent = readdir(pDir)) != NULL) {
    if (ent->d_type == DT_REG) {
      std::string filename(ent->d_name);
      if (!StartWith(filename, prefix)) {
        continue;
      }
      char last_ch = folder_path.back();
      std::string file = folder_path + (last_ch == '/' ? "" : "/") + filename;
      files.push_back(file);
    }
  }
}

real Sigmoid(double x) {
  return exp(x) / (1 + exp(x));
}

// Read word by word from text, return true if read end of file(EOF) or '\n'
bool ReadWord(string &word, FILE* fin) {
  word.clear();
  char ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13 || ch == 9) {
      continue; //skip '\r' and
    }
    if (ch == ' ' || ch == '\t') {
      return false;
    }
    if (ch == '\n') {
      return true;  // if read a '\n' that
    }
    word.push_back(ch);
  }
  return true;
}