#ifndef TRIPLE_HASH_TABLE
#define TRIPLE_HASH_TABLE

//Transition Triple 
typedef struct
{
    int prev;
    int fr;
    int to;
}
TTRIPLE;

//triple element for hash table
typedef struct
{
    TTRIPLE key;
    double val;
}
THELEM;

typedef struct
{
    int length; //Total number of available elements
    int num_used;
    THELEM* p;
}
THASH;

THASH* t_create_empty_hash(int l);
void t_free_hash(THASH* th);
int t_is_null_entry(TTRIPLE tt);
int t_triple_equal(TTRIPLE tt1, TTRIPLE tt2);
int t_hash_fun(TTRIPLE tt);
int t_exist_in_hash(THASH* th, TTRIPLE tt);
void t_add_entry(THASH* th, THELEM elem);
double t_retrieve_value(THASH* th, TTRIPLE tp);
void t_update_with(THASH* th, int idx, double add_value);
double t_retrieve_value_with_idx(THASH* th, int idx);
#endif
