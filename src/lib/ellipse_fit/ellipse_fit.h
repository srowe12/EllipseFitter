#ifndef ELLIPSE_FIT_H
#define ELLIPSE_FIT_H
#include <armadillo>
#include <vector>

std::vector<double>  FitEllipse(const std::vector<double>& x, const std::vector<double>& y);

arma::mat AssembleMatrices(const arma::vec& x, const arma::vec& y); 

#endif // ELLIPSE_FIT_H
