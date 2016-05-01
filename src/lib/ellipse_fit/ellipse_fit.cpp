#include "ellipse_fit.h"

std::vector<double> FitEllipse(const std::vector<double>& x, const std::vector<double>& y)
{
   std::vector<double> lol(6,0); 
   return lol;
}

arma::mat AssembleMatrices(const arma::vec& x, const arma::vec& y) {
    size_t nrows = x.size();

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
    arma::mat T = -inv(S3)*S2.t();
    arma::mat M = S1 + S2*T;
    
    ///@todo may be faster to multiply through by .5 then by -2 on row(1)
    M.row(2) *= .5;
    M.row(1) *=  -1;
    M.row(0) *= .5;
    M.swap_rows(0,2);
    // Also return T! 
    return M;

}

arma::vec ComputeOptimalEigenvector(const arma::mat& M) {
   arma::cx_vec eigval;
   arma::cx_mat eigvec_complex;
 
   eig_gen(eigval, eigvec_complex, M);
   ///@todo Hmmm, this seems gross having to cast as a real vector.
   arma::mat eigvec = real(eigvec_complex);
   arma::rowvec cond = 4*eigvec.row(0) % eigvec.row(2) - eigvec.row(1) % eigvec.row(1);
   return eigvec.cols(arma::find(cond > 0));

}
