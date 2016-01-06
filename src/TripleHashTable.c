#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "TripleHashTable.h"


THASH* t_create_empty_hash(int l)
{
    THASH* th = (THASH*)malloc(sizeof(THASH));
    th -> length = l;
    th -> num_used = 0;
    th -> p = (THELEM*) malloc(l * sizeof(THELEM));
    int i;
    for(i = 0; i < l; i++)
    {
	((th -> p) + i) -> key.prev = -1;
	((th -> p) + i) -> key.fr = -1;
	((th -> p) + i) -> key.to = -1;
	((th -> p) + i) -> val = 0.0;
    }
    return th;
}

void t_free_hash(THASH* th)
{
    free(th -> p);
    free(th);
}

int t_is_null_entry(TTRIPLE tt)
{
    return(tt.prev == -1 && tt.fr == -1 && tt.to == -1);
}

int t_triple_equal(TTRIPLE tt1, TTRIPLE tt2)
{
    return (tt1.prev == tt2.prev && tt1.fr == tt2.fr && tt1.to == tt2.to);
}

int t_hash_fun(TTRIPLE tt)
{
    return fabs((fabs(tt.prev) + 1) * (fabs(tt.fr) + 1)) * ((fabs(tt.to) + 1));
}

//Return the idx of the element, -1 if not exist
int t_exist_in_hash(THASH* th, TTRIPLE tt)
{
    int start_idx = hash_fun(tt) % (th -> length);
    int idx = start_idx;
    TTRIPLE temp_triple;
    while(1)
    {
	temp_triple = (th -> p)[idx].key;
	if(is_null_entry(temp_triple))
	    return -1;
	else if(pair_equal(temp_triple, tt))
	    return idx;
	else
	{
	    idx = (idx + 1) % (th -> length);
	    if(idx == start_idx)
		return -1;
	}
    }
}

void t_add_entry(THASH* th, THELEM elem)
{
    if(th -> num_used == th -> length)
    {
	printf("The hash table is already full.\n");
	exit(1);
    }
    int start_idx = hash_fun(elem.key) % (th -> length);
    int idx = start_idx; 
    while(!is_null_entry((th -> p)[idx].key))
	idx = (idx + 1) % (th -> length);
    (th -> p)[idx] = elem;
    (th -> num_used) += 1;

}

double t_retrieve_value(THASH* th, TTRIPLE tp)
{
    int start_idx = hash_fun(tp) % (th -> length);
    int idx = start_idx;
    while(!pair_equal((th -> p)[idx].key, tp))
	idx = (idx + 1) % (th -> length);
    return (th -> p)[idx].val;
}

void t_update_with(THASH* th, int idx, double add_value)
{
    ((th -> p) + idx) -> val += add_value;
}

double t_retrieve_value_with_idx(THASH* th, int idx)
{
    return ((th -> p) + idx) -> val;
}
