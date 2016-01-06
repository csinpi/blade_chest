#ifndef PAIR_HASH_TABLE
#define PAIR_HASH_TABLE

//Transition pair
typedef struct
{
    int fr;
    int to;
}
TPAIR;

//element for hash table
typedef struct helem
{
    TPAIR key;
    double val;
    struct helem* pnext; //Pointer for tracking the transition pairs with the same from song
}
HELEM;

typedef struct
{
    int length; //Total number of available elements
    int num_used;
    int num_songs;
    HELEM* p;
    HELEM** p_first_tran;
}
PHASH;

PHASH* create_empty_hash(int l, int num_songs);
void free_hash(PHASH* ph);
int is_null_entry(TPAIR tp);
int pair_equal(TPAIR tp1, TPAIR tp2);
long hash_fun(TPAIR tp);
int exist_in_hash(PHASH* ph, TPAIR tp);
void add_entry(PHASH* ph, HELEM elem);
double retrieve_value(PHASH* ph, TPAIR tp);
void update_with(PHASH* ph, int idx, double add_value);
double retrieve_value_with_idx(PHASH* ph, int idx);
void build_same_song_index(PHASH* ph);
void one_more_count(PHASH* ph, TPAIR temp_pair, double val);
double get_pair_value(PHASH* ph, TPAIR tp);
double get_undirected_transition_value(PHASH* ph, int fr, int to);
void naive_print_ph_table(PHASH* ph);
TPAIR* build_transition_array_for_ps(PHASH* ph, int* num_trans);
#endif
