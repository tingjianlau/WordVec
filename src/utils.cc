#include "utils.h"

using namespace std;

char kSegmentFaultCauser[] = "Used to cause artificial segmentation fault";


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
