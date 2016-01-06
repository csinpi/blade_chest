#ifndef LOGISTIC_PRED
#define LOGISTIC_PRED
typedef struct
{
    int method;
    int allow_self_transition;
    int fast_collection;
    int radius;
    int num_points;
    int output_distr;
    int underflow_correction;
    char tagfile[200];
    char train_test_hash_file[200];
    int range;
    char bias_ebd_filename[200];
    char square_dist_filename[200];
    char song_ebd_filename[200];
    char tag_ebd_filename[200];
    int use_hash_TTable;
    int transition_range;
    int test_with_order_two;
} 
TEST_PARAS;

TEST_PARAS parse_test_paras(int argc, char* argv[], char* testfile, char* embedfile, char* trainfile);
#endif
