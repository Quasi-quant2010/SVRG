#include "read_file.h"

void 
split(const string &str, const string delimiter, vector<string> &array)
{
  std::string::size_type idx = str.find_first_of(delimiter);

  if ( idx != std::string::npos) {
    array.push_back(str.substr(0, idx));
    split(str.substr(idx+1), delimiter, array);
  } else {
    array.push_back(str);
  }
}


size_t
get_data_length(const char *filename)
{
  //read file
  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  ifstream in(tmp_string);
  string s;

  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }

  size_t count = 0;
  while (std::getline(in, s)) {
    istringstream iss(s);
    string line;
    //iss >> line;
    count += 1;
  }
  in.close();


  return count;
}


size_t
get_max(size_t* a, size_t* b)
{
  if (a > b) {
    return *a;
  } else {
    return *b;
  }
}


size_t
get_feature_length(const char *filename, 
		   string line_delimiter, 
		   string line_delimiter_between, 
		   string line_delimiter_within)
{

  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  std::string line_buffer;

  std::ifstream in;
  in.open(filename, std::ios::in);

  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }

  size_t max_feature_id = 0;
  size_t feature_id, feature_length;
  float feature_score;
  vector<string> my_arr, my_arr2;
  while ( std::getline(in, line_buffer) ) {
    // trim
    boost::trim_right(line_buffer);
    my_arr.clear();
    //split(line_buffer, line_delimiter, my_arr);
    boost::split(my_arr, line_buffer, boost::is_any_of(line_delimiter));
    feature_length = (size_t)(my_arr.size() - 1);// -1 is label, -1 1:0.1 2:0.4
    string key, value;
    for (size_t j = 1; j < feature_length + 1;  ++j) {
      my_arr2.clear();
      boost::split(my_arr2, my_arr[j], boost::is_any_of(line_delimiter_between));
      key.clear(); key = my_arr2[0]; feature_id = (size_t)atoi(key.c_str());
      value.clear(); value = my_arr2[1]; feature_score = atof(value.c_str());
      max_feature_id = get_max(&max_feature_id, &feature_id);
    }

  }
  in.close();

  return max_feature_id;
}

void 
load_data(RMatrixXf *X, RowVectorXf *y,
	  char *filename, 
	  std::string line_delimiter, std::string line_delimiter_between, std::string line_delimiter_within)
{

  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  std::string line_buffer;

  std::ifstream in;
  in.open(filename, std::ios::in);

  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }


  unsigned int feature_id, feature_length;
  float feature_score;
  size_t i = 0;
  vector<string> my_arr, my_arr2;
  while ( std::getline(in, line_buffer) ) {
    // trim
    boost::trim_right(line_buffer);
    my_arr.clear();
    //split(line_buffer, line_delimiter, my_arr);
    boost::split(my_arr, line_buffer, boost::is_any_of(line_delimiter));

    // insert data
    size_t label =  (size_t)atoi(my_arr[0].c_str());
    if (label > 0) {
      y->coeffRef(i) = 1;
    } else {
      y->coeffRef(i) = 0;
    }    
    feature_length = (size_t)(my_arr.size() - 1);// -1 is label, -1 1:0.1 2:0.4
    string key, value;
    for (size_t j = 1; j < feature_length + 1;  ++j) {
      my_arr2.clear();
      boost::split(my_arr2, my_arr[j], boost::is_any_of(line_delimiter_between));
      key.clear(); key = my_arr2[0]; feature_id = (size_t)atoi(key.c_str());
      value.clear(); value = my_arr2[1]; feature_score = atof(value.c_str());
      X->coeffRef(i, feature_id - 1) = feature_score;
    }

    i += 1;
  }
  in.close();

}

void 
show_data_mat(RMatrixXf *X)
{
  printf("\n");
  for (size_t i = 0; i < X->rows(); i++)
    for (size_t j = 0; j < X->cols(); j++)
      fprintf(stdout, "%d:%d %f %p\n", i, j, X->coeffRef(i,j), &X->coeffRef(i,j));
}

void
show_data_vec(RowVectorXf *y)
{
  for (size_t i = 0; i < y->size(); i++)
    fprintf(stdout, "%d %f %p\n", i, y->coeffRef(i), &y->coeffRef(i));
}
