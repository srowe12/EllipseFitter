#ifndef ELLIPSE_FIT_H
#define ELLIPSE_FIT_H

#include <armadillo>
#include <vector>

std::vector<double>  FitEllipse(const std::vector<double>& x, const std::vector<double>& y);

void AssembleMatrices(const arma::vec& x, const arma::vec& y, arma::mat::fixed<3,3>& M, arma::mat::fixed<3,3>& T);

arma::vec ComputeOptimalEigenvector(const arma::mat::fixed<3,3>& M);

#endif // ELLIPSE_FIT_H
