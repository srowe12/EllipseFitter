#include <gtest/gtest.h>
#include <math.h>
#include <armadillo>

#include "../ellipse_fit.h"

TEST(EllipseFitTests, AssembleMatricesTest) {
   // Draw an ellipse
   size_t num_samples = 20;
   std::vector<double> x(num_samples);
   std::vector<double> y(num_samples);
   
   double angle_delta = 2*M_PI/(num_samples-1);

   
   double a = 4; //Semi major axis
   double b = 3; //Semi minor axis
   // Sample the ellipse with points (x,y) = (a*cos(t), b*sin(t))
   double t = 0;
   for (size_t i = 0; i<num_samples; ++i) {
       x[i] = a*cos(t);
       y[i] = b*sin(t);
       t += angle_delta;
   }
    
   arma::vec xvec(x);
   arma::vec yvec(y); 
   
   arma::mat M = AssembleMatrices(xvec,yvec); 
   
   arma::mat expected_M{ {-1.865454545454545454, 0, 1.04931818181818181}, { 0, -3.42, 0}, {3.3163636363636363, 0, -1.86545454545454}};
   expected_M *=1e2;
   arma::mat diff = M-expected_M;
   double error = sqrt(accu(diff % diff));
   EXPECT_NEAR(0, error, 1e-12);
   
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
