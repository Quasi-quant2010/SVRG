#ifndef __UTIL__H__
#define __UTIL__H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define max_len 1024

typedef std::unordered_map<unsigned int, float> feature_vector;

typedef struct _samples{
  int click;
  feature_vector fv;
} samples;

typedef struct {
  char* train;
  char* test;
  char* out_path;
  char* out_fname;
  float step_size;
  float lambda;
  uint8_t num_inner_loop;
  uint8_t mini_batch_size;
  float convergence_threshold;
  uint8_t convergence_threshold_count;
  uint8_t max_iter;
  uint8_t sparse;
  uint8_t update_option;
} arg_params;

char* string2char(std::string);
std::string char2string(const char* cstr);
void read_args(int, char**, arg_params*);
void show_args(arg_params*);
bool isEmpty(const char*);
bool isNULL(const char*);

#endif //__UTIL__H__
