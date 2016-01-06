#ifndef LOGISTIC
#define LOGISTIC

#include "LogisticEmbed_common.h"
double** logistic_embed_allinone(PDATA pd, PARAS myparas, char* embedding_file, int* l);
double** logistic_embed_with_tags(PDATA pd, TDATA td,  PARAS myparas, char* embedding_file);
PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile);
double compute_regularization_term(PARAS myparas, double** X, double** dl_x, int num_samples, int dim, int* low_indices, int* upper_indices, PDATA* pd, int reg_type);
double R4_distance_metric(PARAS myparas, double** X, int x_ind, int y_ind, int k);

double** logistic_embed_triple_dependency(PDATA pd, PARAS myparas, char* embeddingfile, int* l);
double** logistic_embed_sp_mt(PDATA pd, PARAS myparas, char* embeddingfile, int* l);
#endif
