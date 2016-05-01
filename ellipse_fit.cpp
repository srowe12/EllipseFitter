#include "ellipse_fit.h"

arma::mat

std::vector<double> FitEllipse(const std::vector<double>& x, const std::vector<double>& y)
{
  
}

void AssembleMatrices(const arma::vec& x; const arma::vec& y) {
    arma::mat D1(nrows, 3);
    D1.col(0) = x % x; // x*x
    D1.col(1) = x % y; // x*y
    D1.col(2) = y % y; // y*y
    // Assemble linear chunk matrix 
    arma::mat D2(nrows,3);
    D2.col(0) = x;
    D2.col(1) = y;
    D2.col(2) = arma::ones(nrows,1);


    arma::mat S1 = D1.t()*D1;
    arma::mat S2 = D1.t()*D2;
    arma::mat S3 = D2.t()*D2;
    
    arma::mat M = S1 - S2*T.inv()*S2.t();
    
    //Multiply by C1.inv();
    ///@todo may be faster to multiply through by .5 then by -2 on row(1)
    M.row(2) *= .5;
    M.row(1) *=  -1;
    M.row(0) *= .5;
    
    // Also return T! 
    return M;

}

int main() {
   std::vector<double> x;
   arma::mat lolmat{ {1,0},{0,1}};
   return 0;
}