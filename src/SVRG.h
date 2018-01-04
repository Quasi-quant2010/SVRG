#ifndef SVRG_H
#define SVRG_H

#include <math.h>

#include "utils.h"
#include "read_file.h"

class SVRG{
public:
  arg_params* option_args;     // option_parser's args
  /* <- option_args
   float lambda;                // regularization
   float eta;                   // learning rate
   float epsilon;               // adagrad's epsilon
   float clip;                  // threshold in clipping
   string clip_method;           // clipping method, clipping, squeese, euclidian
   unsigned int iteration;       // max iteration
   unsigned int mini_batch;      // mini_batch size
   float convergence_threshold; // early stopping
   unsigned int batch_sgd;       // if 1, optimization is SGD
  */

  SVRG(size_t, size_t,
	  arg_params*);

  void train(gsl_rng*, FILE*,
	     RMatrixXf*, RowVectorXf*,
	     RowVectorXf*, float*);
  /*
    _r : 
    _fnamae :
    _X : designe matrix
    _y : label
    _w : feature weight
    _w_level : feature weight level
   */
  //void predict(MatrixXd& _x,VectorXd& _l,vector<float>& ret);

  ~SVRG()
    {
      // do nothing
    }

private:
  size_t N;                        // the num of samples
  size_t D;                        // the num of features
  //RowVectorXf E;                // gradient for adaptive learning rate


  /* ---------------- Main ------------------------- */
  void opt_dense(gsl_rng* _r, FILE* _fname,
		 RMatrixXf* _X, RowVectorXf* _y,
		 RowVectorXf* _w, float* _w_level);

  /* ---------------- SVRG ------------------------- */
  float get_learnig_rate(const arg_params* _option_args, 
  			  float _cur_iter);
  float get_max(float a, float b);
  float get_min(float a, float b);

  /* ---------------- Loss Function ------------------- */
  float LogLikelihood(const RMatrixXf *__X, const RowVectorXf *__y,
		       const RowVectorXf *__w, const float *__w_level);

  /* ---------------- Elementary Function ------------------- */
  void init_vector(float* a, size_t len);
  float inner_product(const RMatrixXf *__X, size_t i, const RowVectorXf *__w);
  float sigmoid(float z);
};

#endif
