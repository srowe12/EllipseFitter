#include "../ellipse_fit.h"

#include <iostream>
#include <chrono>

void AssembleMatricesFast(const arma::vec& x, const arma::vec& y, arma::mat::fixed<3,3>& M, arma::mat::fixed<3,3>& T);

arma::vec ComputeOptimalEigenvectorDynamic(const arma::mat::fixed<3,3>& M);

void AssembleMatricesFast(const arma::vec& x, const arma::vec& y, arma::mat::fixed<3,3>& M, arma::mat::fixed<3,3>& T){
    size_t nrows = x.size();
    arma::mat D(nrows,6);
    D.col(0) = x % x;
    D.col(1) = x % y;
    D.col(2) = y % y;
    D.col(3) = x;
    D.col(4) = y;
    D.col(5) = arma::ones(nrows,1);
    arma::mat S = D.t() * D;

    const auto& S1 = S.submat(0,0,2,2);
    const auto& S2 = S.submat(0,3,2,5);
    const auto& S3 = S.submat(3,3,5,5);   
    T = -inv(S3)*S2.t();
    M = S1 + S2*T;
    
    ///@todo may be faster to multiply through by .5 then by -2 on row(1)
    M.row(2) *= .5;
    M.row(1) *=  -1;
    M.row(0) *= .5;
    M.swap_rows(0,2);

}

arma::vec ComputeOptimalEigenvectorDynamic(const arma::mat& M) {
   arma::cx_vec eigval;
   arma::cx_mat eigvec_complex;
 
   eig_gen(eigval, eigvec_complex, M);
   ///@todo Hmmm, this seems gross having to cast as a real vector.
   arma::mat eigvec = real(eigvec_complex);
   arma::rowvec cond = 4*eigvec.row(0) % eigvec.row(2) - eigvec.row(1) % eigvec.row(1);
   return eigvec.cols(arma::find(cond > 0));

}

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
   
   std::array<double,6> ellipse_coefs = FitEllipse(x_in,y_in);

   auto end = std::chrono::steady_clock::now();

   std::cout << "The first coef is" << ellipse_coefs[0] << std::endl;

   auto diff = end - start;

   std::cout << std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;

   arma::vec xin(x);
   arma::vec yin(y); 
   arma::mat::fixed<3,3> M; 
   arma::mat::fixed<3,3> T;

   start = std::chrono::steady_clock::now();
   AssembleMatrices(xin,yin,M,T); 
   end = std::chrono::steady_clock::now();
   diff = end - start;
   std::cout << "Current Matrix Assembly:" << std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;
  
   arma::mat::fixed<3,3> M2; 
   arma::mat::fixed<3,3> T2;

   start = std::chrono::steady_clock::now();
   AssembleMatricesFast(xin,yin,M2,T2); 
   end = std::chrono::steady_clock::now();
   diff = end - start;
   std::cout << "Alternative Matrix Assembly" <<  std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;


   arma::mat Meig{ {-1.865454545454545454, 0, 1.04931818181818181}, { 0, -3.42, 0}, {3.3163636363636363, 0, -1.86545454545454}};
   start = std::chrono::steady_clock::now();
   arma::vec optimal_eigenvector = ComputeOptimalEigenvectorDynamic(Meig);
   end = std::chrono::steady_clock::now();
   diff = end - start;
   std::cout << "Compute Eigenvalue with Dynamic matrix" <<  std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;

   arma::mat::fixed<3,3> Meigfixed{ {-1.865454545454545454, 0, 1.04931818181818181}, { 0, -3.42, 0}, {3.3163636363636363, 0, -1.86545454545454}};
   start = std::chrono::steady_clock::now();
   arma::vec optimal_eigenvector2 = ComputeOptimalEigenvector(Meigfixed);
   end = std::chrono::steady_clock::now();
   diff = end - start;
   std::cout << "Compute Eigenvalue with Fixed Matrix" <<  std::chrono::duration<double, std::milli> (diff).count() << " ms" << std::endl;
}
