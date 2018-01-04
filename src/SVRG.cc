#include "SVRG.h"


SVRG::SVRG(size_t _N, size_t _D,
	   arg_params* _option_args):
  /*
     N : Num of data
     D : Num of feature
   */
  N(_N), 
  D(_D),
  option_args(_option_args)
/*
  E(RowVectorXf::Zero(_D+1))   // adaptive learning rate initialize,
                               // diagonal matrix. Therefore, VectorXd
                               //  feature-weight-dim + weight-level
                               //          _D         +      1
*/
{}


void SVRG::train(gsl_rng *r, FILE *fname, 
		 RMatrixXf *X, RowVectorXf *y,
		 RowVectorXf *w, float *w_level)
{
  // init
  float max_grad = 0.;

  // SVRG
  /*
    Lazy Update for Sparse Data
  if (this->cli_params->sparse)
    opt_sparse(r, fname,
	       X, y, w, w_level);
  */
  // Normal Update for Not Sparse Data
  if (this->option_args->sparse)
    printf("Lazy Update\n");
  else
    opt_dense(r, fname,
	      X, y, w, w_level);
  //Batch(fname, 
  //	X, y, w, w_level);
  /*
  if ( this->option_args->batch_sgd == 1) {
  } else if ( this->option_args->batch_sgd == 0) {
  }
  */
}


/* ---------------- Main ------------------------- */
void SVRG::opt_dense(gsl_rng* _r, FILE *_fname,
		     RMatrixXf *_X, RowVectorXf *_y,
		     RowVectorXf *_w, float *_w_level)
{
  //unsigned int i, j; //i is sample, j is feature index
  float learning_rate;
  float before_loss, after_loss, cur_loss_rate;
  uint16_t cur_iter;
  float inner_product_, pred;
  uint8_t bool_count = 0;

  RowVectorXf w_before = RowVectorXf::Zero(this->D); // before feature weight
  float w_level_before = 0.0f;
  
  cur_loss_rate = this->option_args->convergence_threshold + 1.0;
  before_loss = 1.0;
  cur_iter = 0;
  //cur_loss_rate >= this->option_args->convergence_threshold
  while (bool_count < this->option_args->convergence_threshold_count) {

    // 0. select feature weight update options
    uint8_t num_inner_loop = 0;
    if (this->option_args->update_option == 1)
      num_inner_loop = this->option_args->num_inner_loop;
    if (this->option_args->update_option == 2)
      num_inner_loop = gsl_rng_uniform_int(_r, this->option_args->num_inner_loop+1);

    // 1. save previsous feature weight and weight level
    for (size_t j=0; j < _w->size(); ++j)
      w_before[j] = _w->coeffRef(j);
    w_level_before = *_w_level;


    // 2. calculate error
    float *error = (float *)malloc(sizeof(float) * this->N);
    init_vector(error, this->N);
    for (size_t i=0; i < this->N; ++i) {
      inner_product_ = 0.0f; pred = 0.0f;
      inner_product_ = inner_product(_X, i, _w);
      pred = sigmoid(inner_product_ + *_w_level);
      error[i] = pred - (float)_y->coeffRef(i);                                     // error
    }


    // 3. calculate full gradient : \tilda{\mu_w}1w and \tilda{\mu_w_level}
    //  3.1 \tilda{\mu}
    float *tilda_mu_w = (float *)malloc(sizeof(float) * this->D);
    init_vector(tilda_mu_w, this->D);
    float grad;
    for (size_t j=0; j < this->D; ++j) {
      for (size_t i=0; i < this->N; ++i) {
	grad = 0.0f;
	grad = _X->coeffRef(i,j) * error[i];                               // grad_i_j = _X->coeffRef(i,j) * error[i];
	tilda_mu_w[j] += grad;
      }
      tilda_mu_w[j] /= (float)this->N;
    }
    //  3.2 \tilda{\mu_w_level}
    float tilda_mu_w_level = 0.0f;
    for (size_t i=0; i < this->N; ++i)
      tilda_mu_w_level += error[i];
    tilda_mu_w_level /= (float)this->N;


    // 4. inner loop
    RowVectorXf w_tmp = RowVectorXf::Zero(this->D);
    for (size_t j=0; j < _w->size(); ++j)
      w_tmp[j] = _w->coeffRef(j);
    float w_level_tmp = *_w_level;

    uint16_t cur_inner_loop = 0;
    while (cur_inner_loop < num_inner_loop) {

      // 4.1 select single sample uniformly
      uint32_t idx = gsl_rng_uniform_int(_r, this->N);
      
      // 4.2 update feature weight
      float error_cur, error_pre;

      //  gradient for cur error
      inner_product_ = 0.0f; pred = 0.0f; error_cur = 0.0f;
      inner_product_ = inner_product(_X, idx, &w_tmp);
      pred = sigmoid(inner_product_ + w_level_tmp);
      error_cur = pred - (float)_y->coeffRef(idx);

      // gradient for previous error
      inner_product_ = 0.0f; pred = 0.0f; error_pre = 0.0f;
      inner_product_ = inner_product(_X, idx, &w_before);
      pred = sigmoid(inner_product_ + w_level_before);
      error_pre = pred - (float)_y->coeffRef(idx);

      float grad_cur = 0.0f, grad_pre = 0.0f;
      for (size_t j=0; j < this->D; ++j) {
	// calculate gradient with current feature weight
	grad_cur = _X->coeffRef(idx, j) * error_cur;
	
	// calculate gradient with previous feature weight
	grad_pre = _X->coeffRef(idx, j) * error_pre;
	
	// update feature weight
	//tilda_mu -> tilda_mu.coeffRef(j)
	w_tmp.coeffRef(j) -= this->option_args->step_size * (tilda_mu_w[j] + \
							     (grad_cur - grad_pre) + \
							     this->option_args->lambda * w_tmp.coeffRef(j)); // L2-regularization
      }

      // 4.3 update feature level
      grad_cur = 0.0f; grad_pre = 0.0f;
      //  4.3.1 cur
      grad_cur = error_cur;
      //  4.3.2 previsous
      grad_pre = error_pre;

      w_level_tmp -= this->option_args->step_size * (tilda_mu_w_level + \
						     (grad_cur - grad_pre) + \
						     this->option_args->lambda * w_level_tmp);              // L2-regularization

      cur_inner_loop += 1;
    } // over inner loop

    // 4.4 update feature weight and feature weight level
    for (size_t j=0; j < _w->size(); ++j)
      _w->coeffRef(j)= w_tmp.coeffRef(j);
    *_w_level = w_level_tmp;


    // 5. likelihood
    after_loss = LogLikelihood(_X, _y, _w, _w_level);
    fprintf(_fname, "%d\t%f\n", cur_iter, after_loss);
    //fprintf(stdout, "%d\t%f\n", cur_iter, after_loss);


    // 6. next iteration bool
    if (cur_iter == this->option_args->max_iter) break;

    cur_loss_rate = (float)(fabs(before_loss - after_loss) / fabs(before_loss));
    if (cur_loss_rate <= this->option_args->convergence_threshold)
      bool_count += 1;
    else
      bool_count = 0;

    before_loss = after_loss;
    cur_iter += 1;
  }// over while  

}// over mini_batch_train


float SVRG::get_max(float a, float b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

float SVRG::get_min(float a, float b)
{
  if (a > b) {
    return b;
  } else {
    return a;
  }
}

//void SVRG::EuclideanSqueezing(float *grad)
//{
//}

float SVRG::get_learnig_rate(const arg_params* _option_args,
			      float _cur_iter)
{
  float learning_rate = 0.0;

  learning_rate = _option_args->step_size / sqrt(_cur_iter);
  
  return learning_rate;
}

 
/* ---------------- Loss Function ------------------- */
float SVRG::LogLikelihood(const RMatrixXf *__X, const RowVectorXf *__y,
			  const RowVectorXf *__w, const float *__w_level)
{
  float loss = 0.0;
  for (size_t i = 0; i < this->N; i++) {

    float inner_product_ = inner_product(__X, i, __w);
    float pred = sigmoid(inner_product_ + *__w_level);
    if (__y->coeffRef(i) == 1) {
      loss += log(pred);
    } else if (__y->coeffRef(i) == 0) {
      loss += log(1.0 - pred);
    }
  }

  return -loss / (float)this->N +					\
    0.5 * this->option_args->lambda * (__w->squaredNorm() +		\
				       sqrt(pow(*__w_level,2.0)));
  
}


/* ---------------- Elementary Function ------------------- */
float SVRG::inner_product(const RMatrixXf *__X, size_t i, const RowVectorXf *__w){
  float tmp = __X->row(i).dot(*__w);
  return tmp;
}

float SVRG::sigmoid(float z)
{
  if (z >  6.) {
    return 1.0;
  } else if (z < -6.) {
    return 0.0;
  } else {
    return 1.0 / (1 + exp(-z));
  }
 }

void SVRG::init_vector(float* a, size_t len)
{
  for (size_t i=0; i < len; i++)
    a[i] = 0.0;
}

/*
float SVRG::Acc(vector<float>& pred,VectorXd &l){
  int t =0;
  float loss =0;
  for(int i = 0; i< pred.size();i++){
    loss += (l(i) - pred[i]) * (l(i) - pred[i]);
    int s = pred[i] > 0.5 ? 1 : 0;
    if(s == l(i)){
      t++;
    }
  }
  //return (float)t/pred.size();
  return loss / pred.size();
}

void SVRG::predict(MatrixXd& _x,VectorXd& _l,vector<float>& ret){
  for(int i = 0; i < _x.rows(); i++){
    float inner_product_ = inner_product(_x, i);
    float pred = sigmoid(inner_product_ + this->w_level);
    ret.push_back(pred);
  }
}
*/
