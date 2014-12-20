/*
 * utils.h 
 *
 *  Created on: 2014.7.31
 *      Author: Zeyu Chen(zeyuchen@outlook.com)
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <dirent.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

// CHECK operator utilities
extern char kSegmentFaultCauser[];

#define CHECK(a) if (!(a)) {                                            \
    std::cerr << "CHECK failed "                                        \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_EQ(a, b) if (!((a) == (b))) {                             \
    std::cerr << "CHECK_EQ failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_GT(a, b) if (!((a) > (b))) {                              \
    std::cerr << "CHECK_GT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_LT(a, b) if (!((a) < (b))) {                              \
    std::cerr << "CHECK_LT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_GE(a, b) if (!((a) >= (b))) {                             \
    std::cerr << "CHECK_GE failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_LE(a, b) if (!((a) <= (b))) {             \
    std::cerr << "CHECK_LE failed "                     \
              << __FILE__ << ":" << __LINE__ << "\n"    \
              << #a << " = " << (a) << "\n"             \
              << #b << " = " << (b) << "\n";            \
    *kSegmentFaultCauser = '\0';                        \
  }                                                     \
                                                      \

// mini version of glog, to avoid too much library dependency
enum LogSeverity { INFO, WARNING, ERROR, FATAL }; // Log Level

class Logger {
 public:
  Logger(LogSeverity ls, const std::string& file, int line)
      : ls_(ls), file_(file), line_(line) {}

  std::ostream& stream() const {
    return std::cerr << file_ << " (" << line_ << ") : " ;
  }

  virtual ~Logger() {
    if (ls_ == FATAL) {
      *::kSegmentFaultCauser = '\0';
    }
  }

 private:
  LogSeverity ls_;
  std::string file_;
  int line_;
};

#define LOG(ls) Logger(ls, __FILE__, __LINE__).stream()

// define POD type of this project
// if we want to use double to imporve precsion, just change here
typedef float               real;
typedef int                 int32;
typedef uint32_t            uint32;
typedef int64_t             int64;
typedef uint64_t            uint64;

// Generate a random float value in the range of [0,1) from the
// uniform distribution.
inline real RandReal() {
    return rand() / static_cast<real>(RAND_MAX);
}

// Generate a random integer value in the range of [0,bound) from the
// uniform distribution.
inline int RandInt(int bound) {
    // NOTE: Do NOT use rand() % bound, which does not approximate a
    // discrete uniform distribution will.
    return static_cast<int>(RandReal() * bound);
}

bool ReadWord(std::string &word, FILE* fin);

class FileCloser {
 public:
  FileCloser(FILE* f) : f_(f) {
  }

  virtual ~FileCloser() {
    fclose(f_); 
  }

 private:
  FILE* f_;

  FileCloser(const FileCloser&);  // no copying!
  void operator=(const FileCloser&);  // no copying!
};

#endif  // utils.h
