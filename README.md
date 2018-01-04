# Required Libaray
## Centos
- CentOS release 6.9 (Final)
## GSL
- The GNU Scientific Library (GSL) is a numerical library for C and C++ programmers
- version 1.16
- https://www.gnu.org/software/gsl/
## Eigen
- https://bitbucket.org/eigen/eigen
## Boost
- ver 1.58.0

# data set
## example data
- https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale

# Sample Command
- make
- ./main --train ~/data/libsvm/dense/australian.train \  
  --test ~/data/libsvm/dense/australian.valid \  
  --out_path result \  
  --out_fname SVRG_L2.dat \  
  --step_size 0.1 \  
  --lambda 1.0e-5 \  
  --convergence_threshold 1.0e-4 \  
  --convergence_threshold_count 1

# Sample Result
- in sample folder(result), there is a Batch VS SGD LogLoss Plot.
