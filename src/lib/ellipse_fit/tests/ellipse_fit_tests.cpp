#include <gtest/gtest.h>
TEST(LolTest, LolLolTest) {
   double x = 1.0;
   ASSERT_EQ(1.0,x);
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
