#include "utils.h"

static struct option options[] =
  {
    {"train", required_argument, NULL, 'a'},
    {"test", required_argument, NULL, 'b'},
    {"out_path", required_argument, NULL, 'c'},
    {"out_fname", required_argument, NULL, 'd'},
    {"step_size", required_argument, NULL, 'e'},
    {"lambda", required_argument, NULL, 'f'},
    {"num_inner_loop", required_argument, NULL, 'g'},
    {"mini_batch_size", required_argument, NULL, 'h'},
    {"convergence_threshold", required_argument, NULL, 'i'},
    {"convergence_threshold_count", required_argument, NULL, 'm'},
    {"max_iter", required_argument, NULL, 'j'},
    {"sparse", required_argument, NULL, 'k'},
    {"update_option", required_argument, NULL, 'l'},
    {0, 0, 0, 0}
  };

char* string2char(std::string str)
{
  char* cstr = new char[max_len + 1]; 
  strcpy(cstr, str.c_str());
  return cstr;
}

std::string char2string(const char* cstr)
{
  std::string str(cstr);
  return str;
}

void read_args(int argc_, char **argv_, arg_params *cli_param)
{
  
  // default values for input options
  std::string tmp_fname;
  tmp_fname.clear(); tmp_fname = "/home/tanakai/data/libsvm/australian/train.txt";
  cli_param->train = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = "/home/tanakai/data/libsvm/australian/train.txt";
  cli_param->test = string2char(tmp_fname);

  tmp_fname.clear(); tmp_fname = "/home/tanakai/projects/git/c/regret/SVRG/result"; 
  cli_param->out_path = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = "SVRG_L2.dat"; 
  cli_param->out_fname = string2char(tmp_fname);
  
  // numeric setting params
  cli_param->step_size = 1.0f;
  cli_param->lambda = 0.1f;
  cli_param->num_inner_loop = 10;
  cli_param->mini_batch_size = 1;
  cli_param->convergence_threshold = 0.001f;
  cli_param->convergence_threshold_count = 5;
  cli_param->max_iter = 100;
  cli_param->sparse = 0;
  cli_param->update_option = 1;                  // default update option is 1. 
                                                 // option 2 is to set \tilda{w_s} = w_t for randomly chosen t in {0, ..., m-1}


  // command line options
  int dummy, index;
  while( (dummy = getopt_long(argc_, argv_, "abcdefghijk", options, &index)) != -1 ){
    switch(dummy){
    case 'a':
      cli_param->train = optarg;
      break;
    case 'b':
      cli_param->test = optarg;
      break;
    case 'c':
      cli_param->out_path = optarg;
      break;
    case 'd':
      cli_param->out_fname = optarg;
      break;
    case 'e':
      cli_param->step_size = (float)atof(optarg);
      break;
    case 'f':
      cli_param->lambda = (float)atof(optarg);
      break;
    case 'g':
      cli_param->num_inner_loop = (uint8_t)atoi(optarg);
      break;
    case 'h':
      cli_param->mini_batch_size = (uint8_t)atoi(optarg);
      break;
    case 'i':
      cli_param->convergence_threshold = (float)atof(optarg);
      break;
    case 'm':
      cli_param->convergence_threshold_count = (uint8_t)atoi(optarg);
      break;
    case 'j':
      cli_param->max_iter = (uint8_t)atoi(optarg);
      break;
    case 'k':
      cli_param->sparse = (uint8_t)atoi(optarg);
      break;
    case 'l':
      cli_param->update_option = (uint8_t)atoi(optarg);
      break;
    default:
      printf("Error: An unkown option\n");
      exit(1);
    }
  }
}

void show_args(arg_params *cli_param)
{
  char* pt1 = cli_param->train;
  //printf("%s %p %p\n", cli_param->train, pt1, &(cli_param->train));

  fprintf(stderr, "[SVRG Setting Params]\n");
  fprintf(stderr, "\t Trainfile               : %s\n",   cli_param->train);
  fprintf(stderr, "\t Testfile                : %s\n",  cli_param->test);
  fprintf(stderr, "\t OutPath                 : %s\n",   cli_param->out_path);
  fprintf(stderr, "\t OutFname                : %s\n",    cli_param->out_fname);
  fprintf(stderr, "\t InitialStepSize         : %1.3e\n",    cli_param->step_size);
  fprintf(stderr, "\t L2-Lambda               : %1.3e\n",    cli_param->lambda);
  fprintf(stderr, "\t Sparse                  : %d\n",    cli_param->sparse);
  fprintf(stderr, "\t F-WeightUpdateOption    : %d\n",    cli_param->update_option);
  fprintf(stderr, "\t NumInnerLoop            : %d\n",    cli_param->num_inner_loop);
  fprintf(stderr, "\t MiniBatchSize           : %d\n",    cli_param->mini_batch_size);
  fprintf(stderr, "\t ConvergenceRate         : %1.3e\n",    cli_param->convergence_threshold);
  fprintf(stderr, "\t ConvergenceCount        : %d\n",    cli_param->convergence_threshold_count);
  fprintf(stderr, "\t MaxIter                 : %d\n",    cli_param->max_iter);
}

bool isEmpty(const char *s)
{
  if (s==NULL ) return false;

  if( s!=NULL || strlen(s) == 0 ) return true;

  return false;
}

bool isNULL(const char *s)
{
  if( s==NULL ) return true;
  return false;
}
