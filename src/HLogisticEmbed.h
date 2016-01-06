#ifndef HLOGISTIC_EMBED
#define HLOGISTIC_EMBED
#include "LogisticEmbed_common.h"
#include "PairHashTable.h"
#include "LogisticPred.h"

typedef struct HParas
{
	int use_random_init;
	//int* idx_vec;
	int num_clusters;
	int num_subset_songs;
	int verbose;
	PHASH* tcount;
}
HPARAS;

typedef struct ClusterEmbedding
{
	int d;
	int num_clusters;
	int num_songs;
	int num_portals;
	int* num_songs_in_each_cluster;
	int** cluster_song_relation;
	int* assignment;
	double*** X_by_cluster;
	double** bias_by_cluster;
	int inter_cluster_transition_type;
	int** preferred_portal_array; // |C| * k matrix
}
CEBD;


PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile);

PHASH* create_hash_transition_pair_table(PDATA pd, PARAS myparas, int shift);
//Embedd the subset of songs indicated by the idx_vec
//idx_vec is of length k (num of songs), and each element of value 1 indicates that that song is in the subset. 0 if otherwise
//if use_random_init is true, we use random initilization for X. Else, we use the existing values in X as initialization. 
double logistic_embed_component(PARAS myparas, double** X, double* bias, HPARAS hparas, int* seconds_used);
double logistic_embed_component_ps(PARAS myparas, double** X, double* bias, HPARAS hparas, int* seconds_used);
double logistic_test_component(double** X, double* bias, PHASH* tcount, int d);
void logistic_embed_component_fixed_framework(PARAS myparas, double** X, HPARAS hparas);
double** hierachical_logistic_embed(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters);
double** hierachical_logistic_embed_fix_framework(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters);
CEBD lme_by_cluster(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters);
CEBD lme_by_cluster_uniportal(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters, int num_portals);
void lme_by_cluster_test(PDATA pd, CEBD ce, TEST_PARAS myparas);
void compute_two_tables(int num_as, int num_internal_songs, int d, double** as_to_internal_table, double** internal_to_as_table, double** X, double* bias);
double reassign_external_songs(PHASH* tcount_total, int num_as, int num_internal_songs, int* internal_song_list, int num_external_songs, int* external_song_list, double** as_to_internal_table, double** internal_to_as_table, int* assignment);
void settle_down_internal_songs(double** X, int num_internal_songs, int* internal_song_list, int d, int num_songs, int num_as, int* assignment);
void settle_down_unconnected_external_songs(int num_as, int num_songs, int* assignment, PHASH* tcount_total);
void build_song_cluster_ttable(double** ttable, int* assignment, int num_songs, int num_clusters, PHASH* tcount);
//void free_HParas(HPARAS hparas);


void sort_tpair(HELEM* list_to_sort, int length);

void initial_clustering(PHASH* tcount, int M);

PHASH* create_upper_hash(PDATA pd, PARAS myparas, int num_clusters, int* member_vec);
//build the lower level hash table for the cluster with index cluster_idx
//void create_lower_hash(PDATA pd, PARAS myparas, int num_clusters, int* member_vec, int cluster_idx, int* num_songs_in_cluster, double** local_hub_trans_count, PHASH** tcount);
PHASH* create_lower_hash(PDATA pd, PARAS myparas, int* cluster_idx_list, int cluster_song_count, double* local_hub_trans_count);

PHASH* create_lower_hash_fix_framework(PDATA pd, PARAS myparas, int* cluster_idx_list, int cluster_song_count, int* member_vec, int num_clusters, int current_cluster_id);

//int find_cluster_song_list(int* member_vec, int k, int cluster_idx, int** cluster_idx_list);

void decompose_X(double** X, double** X_decomposed[], int d, int k, int num_clusters, int* cluster_song_relation[], int num_songs_in_each_cluster[]);

void recompose_X(double** X, double** X_decomposed[], int d, int k, int num_clusters, int* cluster_song_relation[], int num_songs_in_each_cluster[], double* local_counts_array[], double** upper_level_hubs);

void count_local_hub_song_transitions(PHASH* tcount, int num_clusters, int current_cluster_id, double* parray);

int count_local_song_song_transtions(PHASH* tcount, int num_clusters);

void compute_transition_stats(PHASH* tcount, int num_clusters, int* ss, int* hs, int* hh, int *total);

void free_ClusterEmbedding(CEBD ce);
void random_assign_cluster(int* assignment, int k, int num_clusters);
void altenate_assign_cluster(int* assignment, int k, int num_clusters);
void greedy_assign_cluster(int* assignment, int k, int num_clusters, PDATA pd);
void greedy_assign_cluster_with_constraints(int* assignment, int k, int num_clusters, PDATA pd, int upper, int lower);
PHASH* create_cluster_hash(PDATA pd, int transition_range, int* cluster_idx_list, int cluster_song_count, int* member_vec, int num_clusters, int current_cluster_id, int type);
void find_transitions_to_each_cluster(int local_song_idx, PHASH* tcount, double* tvec, int num_clusters, int current_cluster_idx, int type);
void find_trans_cluster_log_prob(double** hubs, int num_clusters, int d, int current_cluster_idx, double* p);
//int reached_checkpoint(int rank, int size, int idx, int k, int num_checkpoints);
int count_intra_cluster_transitions(int* assignment, int k, int num_clusters, PDATA pd);
PHASH* create_hash_from_pd(PDATA pd);
PHASH* create_inter_hash_from_pd(PDATA pd, int* assignment);

void find_preferred_portals(CEBD ce, PHASH* total_tcount);
void build_tcount_array(CEBD ce, HPARAS* hparas_array, PHASH* total_tcount);
int count_inter_transition(CEBD ce, PHASH* total_tcount);
void preclustering_by_lme(PDATA pd, PARAS myparas, int num_clusters, int num_portals, int* output_assignment);
void preclustering_with_aggregated_song(PDATA pd, PARAS myparas, int num_as, double internal_percentage, int* output_assignment);
void random_init_aggregated_song(int num_songs, int num_as, double internal_percentage, int* assignment);
void weighted_random_init_aggregated_song(int num_songs, int num_as, double internal_percentage, int* assignment, int* weights);
void max_init_aggregated_song(int num_songs, int num_as, double internal_percentage, int* assignment, int* weights);
PHASH* build_tcount_with_aggregate_songs(PHASH* tcount_total, int* assignment, int num_as, int num_internal_songs, int* internal_song_list);
#ifdef MPI
#include <mpi.h>
double logistic_embed_component_with_comm(PARAS myparas, double** X, double* bias, HPARAS hparas, MPI_Comm comm, int* seconds_used);
CEBD lme_by_cluster_MPI(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters);
#endif

#endif

