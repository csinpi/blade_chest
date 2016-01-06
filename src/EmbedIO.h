#ifndef EMBED_IO
#define EMBED_IO
#include "HLogisticEmbed.h"
#include <stdio.h>

//Output part
void fprint_vec(double* vec, int len, FILE* fp);
void fprint_int_vec(int* vec, int len, FILE* fp);
void fprint_mat(double** mat, int m, int n, FILE* fp);
void fprint_separator(FILE* fp);
void write_ClusterEmbedding_to_file(CEBD ce, char* filename);

//Input part
void read_int_vec(FILE* fp, int** return_vec, int* length);
void read_double_vec(FILE* fp, double** return_vec, int* length);
void read_double_mat(FILE* fp, double*** return_mat, int* m, int* n);
CEBD read_ClusterEmbedding_file(char* filename);
//Utility
void error_in_read(char* str);
int is_in_float_char_set(char c);
int is_in_int_char_set(char c);
int parse_int_line(char* str, int** list);
int parse_double_line(char* str, double** list);
int extract_tail_int(char* str, char* prefix);
int extract_head_int(char* str, int* pos_of_nondigit);
char* find_after_prefix(char* str, char* prefix);
double extract_ending_double(char* str, char* substr);

//Other
int find_cluster_song_list(int* member_vec, int k, int cluster_idx, int** cluster_idx_list);
#endif
