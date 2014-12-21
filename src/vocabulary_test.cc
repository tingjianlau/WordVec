#include <string>
#include <gtest/gtest.h>
#include <gflags/gflags.h>

#include "options.h"
#include "utils.h"
#include "wordvec.h"
#include "vocabulary.h"

using namespace std;

TEST(TestVocabulary, TestAddWord) {
  Vocabulary vocab;
  vocab.AddWord("chenzeyu");
  ASSERT_EQ(vocab.Size(), 1);
  vocab.AddWord("wordvec");
  ASSERT_EQ(vocab.Size(), 2);
  int idx1 = vocab.GetWordIndex("chenzeyu");
  ASSERT_EQ(0, idx1);
  int idx2 = vocab.GetWordIndex("wordvec");
  ASSERT_EQ(1, idx2);
  string w1 = vocab[idx1].word;
  ASSERT_EQ(w1, "chenzeyu");
  string w2 = vocab[idx2].word;
  ASSERT_EQ(w2, "wordvec");
}


TEST(TestVocabulary, TestAddLargeWord) {
  int tot_word = 10000000;
  Vocabulary vocab;
  for(int i = 0;i < tot_word;++i) {
    vocab.AddWord("abc");
  }
  ASSERT_EQ(tot_word, vocab[0].freq);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
