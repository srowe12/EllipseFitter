#ifndef ELLIPSE_FIT_H
#define ELLIPSE_FIT_H

#include <armadillo>
#include <vector>

std::vector<double>  FitEllipse(const std::vector<double>& x, const std::vector<double>& y);

void AssembleMatrices(const arma::vec& x, const arma::vec& y, arma::mat& M, arma::mat& T);
void AssembleMatricesFast(const arma::vec& x, const arma::vec& y, arma::mat& M, arma::mat& T);

arma::vec ComputeOptimalEigenvector(const arma::mat& M);

#endif // ELLIPSE_FIT_H
