#include <string>
#include <gtest/gtest.h>
#include <gflags/gflags.h>

#include "options.h"
#include "utils.h"
#include "wordvec.h"

using namespace std;


TEST(TestUtil, TestStrStartWith) {
  ASSERT_EQ(true, StartWith("chenzeyu", ""));
  ASSERT_EQ(true, StartWith("chenzeyu", "chen"));
  ASSERT_EQ(false, StartWith("chenzeyu", "xx"));
}

TEST(TestUtil, TestSigmoid) {
  ASSERT_EQ(0.5, Sigmoid(0.0));
}

class TestWordVec : public ::testing::Test {
protected:
  TestWordVec() {

  }

  virtual ~TestWordVec() {
  }

  virtual void SetUp() {
    Options options;
  }

  virtual void TearDown() {
  }

  WordVec wordvec;
};


int main(int argc, char **argv) {
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}