#ifndef GAME_PAIR_HASH
#define GAME_PAIR_HASH
typedef struct game_pair
{
    int fst;
    int snd;
}
GPAIR;

typedef struct
{
	GPAIR key;
	GPAIR val;
}
GELE;

typedef struct
{
    int length; //Total number of available elements
    int num_used;
    int num_players;
    GELE* array;
}
GPHASH;

GPHASH create_gp_hash(int l, int num_players);
void free_gp_hash(GPHASH gp);
int gele_is_empty(GELE g);
long gp_hash_fun(GPAIR gp);
int gp_equal(GPAIR gp1, GPAIR gp2);
int find_gp_key(GPAIR gp, GPHASH gp_hash, int* ending_point);
void update_with_gele(GELE gele, GPHASH* p_gp_hash);
void print_gp_hash(GPHASH gp_hash);
void copy_gp_hash(GPHASH* src, GPHASH* dest);




#endif
