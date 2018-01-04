#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <boost/algorithm/string/join.hpp>

#include "src/utils.h"
#include "src/read_file.h"
#include "src/SVRG.h"

int main(int argc, char **argv)
{

  // 1. option parser
  arg_params* cli_params = (arg_params*)malloc(sizeof(arg_params));
  read_args(argc, argv, cli_params);
  show_args(cli_params);

  // 2. read train data and initialize feature weight vector
  std::string line_delimiter, line_delimiter_between, line_delimiter_within;
  line_delimiter.clear(); line_delimiter = " ";
  line_delimiter_between.clear(); line_delimiter_between = ":";
  line_delimiter_within.clear(); line_delimiter_within = ":";

  // Train
  size_t feature_dim = 0;
  size_t train_size = 0;
  size_t *train_size_ptr = &train_size;
  feature_dim = get_feature_length(cli_params->train,
				   line_delimiter, line_delimiter_between, line_delimiter_within);
  train_size = get_data_length(cli_params->train);
  fprintf(stderr, "Train:(N,D)=(%zu,%zu)", train_size, feature_dim);
  RMatrixXf X_train = RMatrixXf::Zero(train_size,feature_dim);     // design matrix
  RowVectorXf y_train = RowVectorXf::Zero(train_size);             // label
  load_data(&X_train, &y_train, 
	    cli_params->train, 
	    line_delimiter, line_delimiter_between, line_delimiter_within);
  // Test
  unsigned int test_size = 0;
  unsigned int *test_size_ptr = &test_size;
  test_size = get_data_length(cli_params->test);
  RMatrixXf X_test = RMatrixXf::Zero(test_size,feature_dim);
  RowVectorXf y_test = RowVectorXf::Zero(test_size);
  load_data(&X_test, &y_test, 
	    cli_params->test, 
	    line_delimiter, line_delimiter_between, line_delimiter_within);
  cout << "Train:" << train_size << " Test:" << test_size << endl;

  // 3. main
  SVRG svrg = SVRG(X_train.rows(), X_train.cols(),
		   cli_params);
  
  const gsl_rng_type *T;
  gsl_rng *r;
  T = gsl_rng_mt19937;        // random generator
  r = gsl_rng_alloc(T);       // random gererator pointer
  gsl_rng_set(r, time(NULL)); // initialize seed for random generator by sys clock

  // 3.1 Train
  std::vector<std::string> my_arr;
  my_arr.push_back(cli_params->out_path);
  my_arr.push_back(cli_params->out_fname);
  std::string joined = boost::algorithm::join(my_arr, "/");

  FILE *output;
  if ( (output = fopen(joined.c_str(), "w")) == NULL ) {
    printf("can not make output file");
    exit(1);
  }
  RowVectorXf w = RowVectorXf::Ones(feature_dim);  // feature weight. initialize
  w /= (float)(feature_dim + 1);
  float w_level = 1.0f / (float)(feature_dim + 1); // feature weight level
  float *w_level_ptr = &w_level;
  svrg.train(r, output,
	     &X_train, &y_train,
	     &w, w_level_ptr);
  fclose(output);

  //vector<float> adgscores;
  //adglr.predict(X_test, y_test, adgscores);
  //cout<<"=== Adagrad ==="<<endl;
  //cout <<"Accuracy:"<< adglr.Acc(adgscores, y_test) << endl;

  gsl_rng_free(r);
}
