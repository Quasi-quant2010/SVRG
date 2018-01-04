#ifndef __READ_FILE_H__
#define __READ_FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <random>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
using namespace std;

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <Eigen/Core>
using namespace Eigen;

#include "utils.h"

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RMatrixXf;

void 
split(const string&, const string, vector<string>&);

size_t
get_data_length(const char*);

size_t
get_max(size_t*, size_t*);

size_t
get_feature_length(const char*, 
		   std::string, std::string, std::string);

void 
load_data(RMatrixXf*, RowVectorXf*,
	  char*, 
	  std::string, std::string, std::string);

void
show_data_mat(RMatrixXf*);

void
show_data_vec(RowVectorXf*);

#endif //__READ_FILE_H__
