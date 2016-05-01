# Numerically Stable Ellipse Fitting
Our goal is to fit an ellipse to a sequence of data points in 2D. The algorithm can be found in "Numerically Stable Direct Least Squares Fitting of Ellipses" by Halir and Flusser. This paper is an improvement on a method in "Direct Least Squares Fitting of Ellipses" by Fitzgibbon, Pilu, and Fisher.

Our goal was to provide a C++ function with an API leveraging just common STL objects (std::vectors) and using an underlying numerical linear algebra package within (Armadillo in this case).
# Requirements
We have the following requirements:
* CMake version 2.8 at least
* Armadillo (version 6.4 was used in our tests)
* GoogleTest 
* Pthread (GoogleTest seems to require this on my Ubuntu 14.04)
* C++11 compliant compiler
