#include "ellipse_fit.h"

std::vector<double> FitEllipse(const std::vector<double>& x, const std::vector<double>& y)
{
   arma::mat M,T;
   AssembleMatrices(arma::vec(x), arma::vec(y), M,T);
   arma::vec optimal_eigenvector = ComputeOptimalEigenvector(M);
   arma::vec linear_coefs = T*optimal_eigenvector;

   std::vector<double> ellipse_coefs{optimal_eigenvector(0), optimal_eigenvector(1), optimal_eigenvector(2), linear_coefs(0), linear_coefs(1), linear_coefs(2)};
   return ellipse_coefs;
}


void AssembleMatrices(const arma::vec& x, const arma::vec& y, arma::mat& M, arma::mat& T) {
    size_t nrows = x.size();
    arma::mat D(nrows,6);
    D.col(0) = x % x;
    D.col(1) = x % y;
    D.col(2) = y % y;
    D.col(3) = x;
    D.col(4) = y;
    D.col(5) = arma::ones(nrows,1);
    arma::mat S = D.t() * D;

    auto S1 = S.submat(0,0,2,2);
    auto S2 = S.submat(0,3,2,5);
    auto S3 = S.submat(3,3,5,5);   
    T = -inv(S3)*S2.t();
    M = S1 + S2*T;
    
    ///@todo may be faster to multiply through by .5 then by -2 on row(1)
    M.row(2) *= .5;
    M.row(1) *=  -1;
    M.row(0) *= .5;
    M.swap_rows(0,2);

}

void AssembleMatricesFast(const arma::vec& x, const arma::vec& y, arma::mat& M, arma::mat& T) {
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

arma::vec ComputeOptimalEigenvector(const arma::mat& M) {
   arma::cx_vec eigval;
   arma::cx_mat eigvec_complex;
 
   eig_gen(eigval, eigvec_complex, M);
   ///@todo Hmmm, this seems gross having to cast as a real vector.
   arma::mat eigvec = real(eigvec_complex);
   arma::rowvec cond = 4*eigvec.row(0) % eigvec.row(2) - eigvec.row(1) % eigvec.row(1);
   return eigvec.cols(arma::find(cond > 0));

}
