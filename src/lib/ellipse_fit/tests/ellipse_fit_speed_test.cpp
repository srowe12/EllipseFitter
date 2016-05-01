#include "../ellipse_fit.h"

#include <iostream>
#include <chrono>

int main() {
   //Build some data

   
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

   // Test speed of code here:
   auto start = std::chrono::steady_clock::now();
   
   std::vector<double> coefs = FitEllipse(x_in,y_in); 

   auto end = std::chrono::steady_clock::now();

   std::cout << "The first coef is" << coefs[0] << std::endl;

   auto diff = end - start;

   std::cout << std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;

   arma::vec xin(x);
   arma::vec yin(y); 
   arma::mat M(3,3); 
   arma::mat T(3,3);

   start = std::chrono::steady_clock::now();
   AssembleMatrices(xin,yin,M,T); 
   end = std::chrono::steady_clock::now();
   diff = end - start;
   std::cout << "Current Matrix Assembly:" << std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;
  
   arma::mat M2(3,3); 
   arma::mat T2(3,3);

   start = std::chrono::steady_clock::now();
   AssembleMatricesFast(xin,yin,M2,T2); 
   end = std::chrono::steady_clock::now();
   diff = end - start;
   std::cout << "Alternative Matrix Assembly" <<  std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;
}
