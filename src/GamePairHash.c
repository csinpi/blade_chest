#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "GamePairHash.h"

GPHASH create_gp_hash(int l, int num_players)
{
	GPHASH gp_hash;
	gp_hash.length = l;
	gp_hash.num_used = 0;
	gp_hash.num_players = num_players;
	gp_hash.array = (GELE*)malloc(l * sizeof(GELE));
	int i;
	for(i = 0; i < l; i++)
	{
		(gp_hash.array[i]).key.fst = -1;
		(gp_hash.array[i]).key.snd = -1;
		(gp_hash.array[i]).val.fst = -1;
		(gp_hash.array[i]).val.snd = -1;
	}
	return gp_hash;
}

void free_gp_hash(GPHASH gp_hash)
{
	free(gp_hash.array);
}

int gele_is_empty(GELE g)
{
	if((g.key.fst) == -1 && (g.key.snd) == -1)
		return 1;
	else
		return 0;
}

int gp_equal(GPAIR gp1, GPAIR gp2)
{
	return((gp1.fst == gp2.fst) && (gp1.snd == gp2.snd));
}

long gp_hash_fun(GPAIR gp)
{
	return fabs((abs(gp.fst) + 1)) * ((abs(gp.snd) + 1));
}
 
int find_gp_key(GPAIR gp, GPHASH gp_hash, int* ending_point)
{
	int starting_point = gp_hash_fun(gp) % gp_hash.length;
	int i = starting_point;
	*ending_point = i;
	while(1)
	{
		if(gp_equal(gp, gp_hash.array[i].key))
			return i;
		if(gele_is_empty(gp_hash.array[i]))
			return -1;
		else
		{
			i = ((i + 1) % gp_hash.length);
			*ending_point = i;
			if(i == starting_point)
				return -1;
		}
	}
}

void update_with_gele(GELE gele, GPHASH* p_gp_hash)
{
	int exist_key, ending_point;
	exist_key = find_gp_key(gele.key, *p_gp_hash, &ending_point);
	
	if(exist_key >= 0)
	{
		((p_gp_hash -> array)[exist_key]).val.fst += gele.val.fst;
		((p_gp_hash -> array)[exist_key]).val.snd += gele.val.snd;
	}
	else
	{
		if((p_gp_hash -> num_used) == (p_gp_hash -> length))
		{
			printf("The game pair table is full.\n");
			exit(1);
		}
		else
		{
			((p_gp_hash -> array)[ending_point]).key = gele.key;
			((p_gp_hash -> array)[ending_point]).val = gele.val;
			(p_gp_hash -> num_used) += 1;
		}
	}
}

void print_gp_hash(GPHASH gp_hash)
{
	int i;
	for(i = 0; i < gp_hash.length; i++)
	{
		if(!gele_is_empty(gp_hash.array[i]))
			printf("(%d, %d, %d, %d)\n", gp_hash.array[i].key.fst, gp_hash.array[i].key.snd, gp_hash.array[i].val.fst, gp_hash.array[i].val.snd);
	}
}

void copy_gp_hash(GPHASH* src, GPHASH* dest)
{
	dest -> length = src -> length;
	dest -> num_used = src -> num_used;
	dest -> num_players = src -> num_players;
	int i;
	for(i = 0; i < src -> length; i++)
		(dest -> array)[i] = (src -> array)[i];
}
