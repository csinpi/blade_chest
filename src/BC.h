#ifndef GAME_EMBEDDING
#define GAME_EMBEDDING
#include "GamePairHash.h"
typedef struct Match
{
    int pa; //first player
    int pb; //second player
	int pw; //winning player
}
MATCH;

typedef struct Matches
{
    int pa; //first player
    int pb; //second player
	int na; //Number of wins the first player gets
	int nb; //Number of wins the second player gets
}
MATCHES;

typedef struct GameRecords 
{
    int num_players;
	int num_games;
	MATCH* all_games;
	int* test_mask; //Masks that indicates whether a game if for training, validation or testing.
	char** all_players; //Names for all players.
	int with_mask; //Whether the dataset contatins the mask or not.
}
GRECORDS;

typedef struct GameEmbedding 
{
    int k; //number of players
	int d; //dimensionality of the blade/chest vectors
	int modeltype; //dist or inner model
	double* ranks; //the bias terms
	double** tvecs; //chest vectors
	double** hvecs; //blade vectors
	int rankon; //whether using the bias terms or not
}
GEMBEDDING;

GRECORDS read_game_records_data(char* filename, int verbose);
PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile);
void extract_game_detal_from_line(char* str, int* pa, int* pb, int* na, int* nb);
void free_game_records_data(GRECORDS grs);
double* BTL_train_test(GRECORDS grs, PARAS myparas);
double safe_log_logit(double a);
GEMBEDDING game_embedding_train_test(GRECORDS grs, PARAS myparas);
void free_game_embedding(GEMBEDDING gebd);
void init_game_embedding(GEMBEDDING* p_gebd, int k, int d, int rankon, int modeltype);
void copy_game_embedding(GEMBEDDING* p_dest, GEMBEDDING* p_src);
void shuffle_match_array(MATCH* array, size_t n);
void shuffle_gele_array(GELE* array, size_t n);
int random_in_range (unsigned int min_val, unsigned int max_val);
double matchup_fun(GEMBEDDING gebd, int a, int b, int modeltype);
double vec_diff(const double* x, const double* y, int length);
double logistic_fun(double a);
void write_GameEmbedding_to_file(GEMBEDDING ge, char* filename);
void compute_ll_obj(double* avg_ll, double* obj, int num_used, GELE* training_games_aggregated, GEMBEDDING main_embedding, PARAS myparas);




//Related to some weak baselines. Not used any more.
typedef struct BaselineModel
{
	int num_players;
	int d;
	double** X;
	GPHASH sigma;
}
BModel;

void init_BModel(BModel* p_bm, int num_players, int d, GPHASH* aggregated_games);
void free_BModel(BModel* p_bm);
BModel train_baseline_model(GPHASH* aggregated_games, PARAS myparas, int verbose);
void copy_BModel(BModel* src, BModel* dest);
double matchup_fun_bm(BModel bm, int a, int b, double* sigma_ab);
void test_baseline_model(BModel bm, GRECORDS grs, int* test_mask);
double matchup_matrix_recover_error(double** true_matchup_mat, double** predict_matchup_mat, int k);
int sign_fun(double a);

#endif
