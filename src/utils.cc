#include "utils.h"

using namespace std;

char kSegmentFaultCauser[] = "Used to cause artificial segmentation fault";

//// check whether a string is start with given prefix
//bool StartWith(const string &word, const string &prefix) {
//  if (prefix.size() == 0) {
//    return true;
//  }
//  if (word.size() < prefix.size()) {
//    return false;
//  }
//  for (size_t i = 0; i < prefix.size(); ++i) {
//    if (word[i] != prefix[i])
//      return false;
//  }
//
//  return true;
//}
//
//// a simple file operator to get all the files under given folder
//void GetAllFiles(const string &folder_path, vector<string> &files,
//                 const string &prefix) {
//  files.clear();
//  struct dirent* ent = NULL;
//  DIR *pDir;
//  pDir = opendir(folder_path.c_str());
//  while ((ent = readdir(pDir)) != NULL) {
//    if (ent->d_type == DT_REG) {
//      string filename(ent->d_name);
//      if (!StartWith(filename, prefix)) {
//        continue;
//      }
//      char last_ch = folder_path.back();
//      string file = folder_path + (last_ch == '/' ? "" : "/") + filename;
//      files.push_back(file);
//    }
//  }
//}
