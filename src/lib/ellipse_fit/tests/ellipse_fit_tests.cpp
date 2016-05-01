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
   arma::mat M(3,3); 
   arma::mat T(3,3);
   AssembleMatrices(xvec,yvec,M,T); 
   
   arma::mat expected_M{ {-1.865454545454545454, 0, 1.04931818181818181}, { 0, -3.42, 0}, {3.3163636363636363, 0, -1.86545454545454}};
   arma::mat expected_T{{-.18181818181818181, 0, .1022727272727272},{0,0,0},{-8.3636363636363,0,-4.295454545454}};
   expected_M *=1e2;
   arma::mat diff_M = M-expected_M;
   arma::mat diff_T = T-expected_T;
   double error_T = sqrt(accu(diff_M % diff_M));
   double error_M = sqrt(accu(diff_T % diff_T));
   EXPECT_NEAR(0, error_M, 1e-12);
   EXPECT_NEAR(0, error_T, 1e-12);
   
}

TEST(EllipseFitTests, ComputeOptimalEigenvectorTest) {

   arma::mat M{ {-1.865454545454545454, 0, 1.04931818181818181}, { 0, -3.42, 0}, {3.3163636363636363, 0, -1.86545454545454}};
   M *=1e2;
   arma::vec expected_optimal_eigenvector{-.490261239632559,0, -.871575537124549};

   arma::vec optimal_eigenvector = ComputeOptimalEigenvector(M);

   arma::vec diff = optimal_eigenvector - expected_optimal_eigenvector;
   EXPECT_NEAR(0, arma::norm(diff), 1e-12);
}

TEST(EllipseFitTests, FitEllipse) {

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

   std::vector<double> ellipse_coefs = FitEllipse(x,y);
   std::vector<double> ellipse_coefs_expected{-.490261239632559,0,-.871575537124549,0,0,7.844179834120943};
   for(size_t i =0; i<ellipse_coefs_expected.size(); ++i) {
       // They may differ by a sign, so normalize by the constant coefficient to handle sign
       EXPECT_NEAR(ellipse_coefs_expected[i]/ellipse_coefs_expected[5], ellipse_coefs[i]/ellipse_coefs[5],1e-12);
    }
}

TEST(EllipseFitTests, FitEllipseRotated) {

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
   // Rotate points by 45 degrees;
   double angle_radians = 45*M_PI/180.0; //degrees
   double cos_angle = cos(angle_radians);
   double sin_angle = sin(angle_radians);
   arma::mat R{{cos_angle, -sin_angle},{sin_angle,cos_angle}};
   arma::rowvec xvec(x);
   arma::rowvec yvec(y);

   arma::mat data(2,num_samples);
   data.row(0) = xvec;
   data.row(1) = yvec;
   
   arma::mat data_rotated = R*data;
   xvec = data_rotated.row(0);
   yvec = data_rotated.row(1);
   std::vector<double> x_in = arma::conv_to<std::vector<double>>::from(xvec);
   std::vector<double> y_in = arma::conv_to<std::vector<double>>::from(yvec);
   

   std::vector<double> ellipse_coefs = FitEllipse(x_in,y_in);
   std::vector<double> ellipse_coefs_expected{-.65743960379771, .368166178126752, -.657439603797772,0,0,7.573704235750322};
   for(size_t i =0; i<ellipse_coefs_expected.size(); ++i) {
       // They may differ by a sign, so normalize by the constant coefficient to handle sign
       EXPECT_NEAR(ellipse_coefs_expected[i]/ellipse_coefs_expected[5], ellipse_coefs[i]/ellipse_coefs[5],1e-12);
    }
}

TEST(EllipseFitTests, FitEllipseRotatedAndShifted) {

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
   // Rotate points by 45 degrees;
   double angle_radians = 45*M_PI/180.0; //degrees
   double cos_angle = cos(angle_radians);
   double sin_angle = sin(angle_radians);
   arma::mat R{{cos_angle, -sin_angle},{sin_angle,cos_angle}};
   arma::rowvec xvec(x);
   arma::rowvec yvec(y);

   arma::mat data(2,num_samples);
   data.row(0) = xvec;
   data.row(1) = yvec;
   
   arma::mat data_rotated = R*data;
   xvec = data_rotated.row(0);
   yvec = data_rotated.row(1);
   
   std::vector<double> x_in = arma::conv_to<std::vector<double>>::from(xvec);
   std::vector<double> y_in = arma::conv_to<std::vector<double>>::from(yvec);
   //Now shift the x values by 1 and the y values by 2 to give it a center of (1,2);
   double x_center = 1;
   double y_center = 2;
   for (size_t i =0; i<x_in.size(); ++i) {
       x_in[i] += x_center;
       y_in[i] += y_center;
   }

   std::vector<double> ellipse_coefs = FitEllipse(x_in,y_in);
   std::vector<double> ellipse_coefs_expected{-.65743960379771, .368166178126752, -.657439603797772,.578546851342042,2.261592237064336,5.022838573014967};
   for(size_t i =0; i<ellipse_coefs_expected.size(); ++i) {
       // They may differ by a sign, so normalize by the constant coefficient to handle sign
       EXPECT_NEAR(ellipse_coefs_expected[i]/ellipse_coefs_expected[5], ellipse_coefs[i]/ellipse_coefs[5],1e-12);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
