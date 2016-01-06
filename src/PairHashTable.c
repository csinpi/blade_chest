#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "PairHashTable.h"


PHASH* create_empty_hash(int l, int num_songs)
{
	PHASH* ph = (PHASH*)malloc(sizeof(PHASH));
	ph -> num_songs = num_songs;
	ph -> length = l;
	ph -> num_used = 0;
	ph -> p = (HELEM*) malloc(l * sizeof(HELEM));
	int i;
	for(i = 0; i < l; i++)
	{
		((ph -> p) + i) -> key.fr = -1;
		((ph -> p) + i) -> key.to = -1;
		((ph -> p) + i) -> val = 0.0;
		((ph -> p) + i) -> pnext = NULL;
	}

	ph -> p_first_tran = (HELEM**)malloc(num_songs * sizeof(HELEM*));
	for(i = 0; i < num_songs; i++)
		(ph -> p_first_tran)[i] = NULL;
	return ph;
}

void free_hash(PHASH* ph)
{
	free(ph -> p);
	free(ph -> p_first_tran);
	free(ph);
}

int is_null_entry(TPAIR tp)
{
	return(tp.fr == -1 && tp.to == -1);
}

int pair_equal(TPAIR tp1, TPAIR tp2)
{
	return (tp1.fr == tp2.fr && tp1.to == tp2.to);
}

long hash_fun(TPAIR tp)
{
	return fabs((fabs(tp.fr) + 1) ) * ((fabs(tp.to) + 1));
}

//Return the idx of the element, -1 if not exist
int exist_in_hash(PHASH* ph, TPAIR tp)
{
	int start_idx = hash_fun(tp) % (ph -> length);
	int idx = start_idx;
	TPAIR temp_pair;
	while(1)
	{
		temp_pair = (ph -> p)[idx].key;
		if(is_null_entry(temp_pair))
			return -1;
		else if(pair_equal(temp_pair, tp))
			return idx;
		else
		{
			idx = (idx + 1) % (ph -> length);
			if(idx == start_idx)
				return -1;
		}
	}

}

void add_entry(PHASH* ph, HELEM elem)
{
	if(ph -> num_used == ph -> length)
	{
		printf("The hash table is already full.\n");
		exit(1);
	}
	int start_idx = hash_fun(elem.key) % (ph -> length);
	int idx = start_idx; 
	while(!is_null_entry((ph -> p)[idx].key))
		idx = (idx + 1) % (ph -> length);
	elem.pnext = NULL;
	(ph -> p)[idx] = elem;
	(ph -> num_used) += 1;

}

double retrieve_value(PHASH* ph, TPAIR tp)
{
	int start_idx = hash_fun(tp) % (ph -> length);
	int idx = start_idx;
	while(!pair_equal((ph -> p)[idx].key, tp))
		idx = (idx + 1) % (ph -> length);
	return (ph -> p)[idx].val;
}

void update_with(PHASH* ph, int idx, double add_value)
{
	((ph -> p) + idx) -> val += add_value;
}

double retrieve_value_with_idx(PHASH* ph, int idx)
{
	return ((ph -> p) + idx) -> val;
}

void build_same_song_index(PHASH* ph)
{
	int i;
	int fr;
	TPAIR temp_pair;
	HELEM** current_p_array = (HELEM**)malloc((ph -> num_songs) * sizeof(HELEM*));
	HELEM* current_address;
	for(i = 0; i < (ph -> num_songs); i++)
		current_p_array[i] = NULL;
	for(i = 0; i < ph -> length; i++)
	{
		current_address = (ph -> p) + i;
		temp_pair = current_address -> key;
		if(!is_null_entry(temp_pair))
		{
			fr = temp_pair.fr;
			//First transition in the list
			if(current_p_array[fr] == NULL)
				(ph -> p_first_tran)[fr] = current_address;
			//Not the first time. Link it to the back
			else
				current_p_array[fr] -> pnext = current_address;
			current_p_array[fr] = current_address;
		}
	}
	free(current_p_array);
}

void one_more_count(PHASH* ph, TPAIR temp_pair, double val)
{
	if(temp_pair.fr < 0 || temp_pair.to < 0)
		printf("(%d, %d)\n", temp_pair.fr, temp_pair.to);
	assert(temp_pair.fr >= 0 && temp_pair.to >= 0);
	int idx;
	HELEM temp_elem;
	idx = exist_in_hash(ph, temp_pair);
	if(idx < 0)
	{
		temp_elem.key = temp_pair;
		temp_elem.val = val;
		add_entry(ph, temp_elem);
	}
	else
		update_with(ph, idx, val);
}

double get_pair_value(PHASH* ph, TPAIR tp)
{
	int idx = exist_in_hash(ph, tp);
	if(idx < 0)
		return 0.0;
	else
		return retrieve_value_with_idx(ph, idx);
}

double get_undirected_transition_value(PHASH* ph, int fr, int to)
{
	double value = 0.0;
	TPAIR temp_pair;
	temp_pair.fr = fr;
	temp_pair.to = to;
	value += get_pair_value(ph, temp_pair);
	temp_pair.fr = to;
	temp_pair.to = fr;
	value += get_pair_value(ph, temp_pair);
	return value;
}

void naive_print_ph_table(PHASH* ph)
{
	int k = ph -> num_songs;
	int fr, to;
	int idx;
	double val;
	TPAIR temp_pair;
	for(fr = 0; fr < k; fr++)
	{
		for(to = 0; to <k; to++)
		{
			temp_pair.fr = fr;
			temp_pair.to = to;
			idx = exist_in_hash(ph, temp_pair);
			if(idx >= 0)
			{
				val = retrieve_value_with_idx(ph, idx);
				printf("(%d, %d, %f) ", fr, to, val);
			}
		}
		putchar('\n');
	}
}

TPAIR* build_transition_array_for_ps(PHASH* ph, int* num_trans)
{
	int i;
	int j;
	double dcount = 0.0;
	for(i = 0; i < ph -> length; i++)
	{
		if(!is_null_entry((ph -> p)[i].key))
			dcount += (ph -> p)[i].val;
	}
	TPAIR* tarray = (TPAIR*)malloc((int)dcount * sizeof(TPAIR));
	int idx = 0;
	for(i = 0; i < ph -> length; i++)
		if(!is_null_entry((ph -> p)[i].key))
			for(j = 0; j < (int) ((ph -> p)[i].val); j++)
				tarray[idx++] = (ph -> p)[i].key;
	assert(idx == (int)dcount);
	*num_trans = idx;
	return tarray;
}
