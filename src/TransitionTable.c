#include <stdio.h>
#include <stdlib.h>
#include "TransitionTable.h"


void init_linked_list(LINKEDLIST* pll)
{
    pll -> head = NULL;
    pll -> tail = NULL;
    pll -> length = 0;
}

void free_linked_list(LINKEDLIST* pll)
{
    LINKEDELEM* p;
    LINKEDELEM* pp;
    if(pll -> length == 0)
    {
	pll -> head = NULL;
	pll -> tail = NULL;
    }
    else
    {
	p = pll -> head;
	while(p != NULL)
	{
	    pp = p;
	    p = p -> pnext;
	    free(pp);
	}
	pll -> head = NULL;
	pll -> tail = NULL;
	pll -> length = 0;
    }
}

int exist_in_linked_list(int q, LINKEDLIST* pll)
{
    LINKEDELEM* p;
    if(pll -> length == 0)
	return 0;
    else
    {
	p = pll -> head;
	while(p != NULL)
	{
	    if(q == p -> idx)
		return 1;
	    else
		p = p -> pnext;
	}
	return 0;
    }

}

void insert_in_linked_list(int q, LINKEDLIST* pll)
{
    if(pll -> length == 0)
    {
	pll -> head = (LINKEDELEM*)malloc(sizeof(LINKEDELEM));
	pll -> head -> idx = q;
	pll -> head -> pnext = NULL;
	pll -> tail = pll -> head;
    }
    else
    {
	pll -> tail -> pnext = (LINKEDELEM*)malloc(sizeof(LINKEDELEM));
	pll -> tail = pll -> tail -> pnext;
	pll -> tail -> idx = q;
	pll -> tail -> pnext = NULL;
    }
    pll -> length += 1;
}

void init_transition_table(TRANSITIONTABLE* ptt, int m)
{
    int i;
    ptt -> n = m;
    ptt -> parray = (LINKEDLIST*)malloc(m * sizeof(LINKEDLIST));
    for(i = 0; i < m; i++)
	init_linked_list((ptt -> parray) + i);
}

void free_transition_table(TRANSITIONTABLE* ptt)
{
    int i;
    for(i = 0; i < ptt -> n; i++)
	free_linked_list((ptt -> parray) + i);
    free(ptt -> parray);
}

void insert_in_transition_table(TRANSITIONTABLE* ptt, int fr, int to)
{
    insert_in_linked_list(to, (ptt -> parray) + fr);
}

int exist_in_transition_table(TRANSITIONTABLE* ptt, int fr, int to)
{
    return exist_in_linked_list(to, (ptt -> parray) + fr);
}

void print_linked_list(LINKEDLIST* pll)
{
    LINKEDELEM* p;
    if(pll -> length == 0)
	return;
    else
    {
	p = pll -> head;
	while(p != NULL)
	{
	    printf("%d ", p -> idx);
	    p = p -> pnext;
	}
    }
}
    
void print_transition_table(TRANSITIONTABLE* ptt)
{
    int i;
    for(i = 0; i < ptt -> n; i++)
    {
	printf("idx %d, length %d: ", i, ptt -> parray[i].length);
	print_linked_list((ptt -> parray) + i);
	putchar('\n');
    }
}

int pop_from_linked_list(LINKEDLIST* pll)
{
    int temp;
    LINKEDELEM* p;
    if(pll -> length == 0)
    {
	printf("Trying to pop from an empty list!\n");
	exit(1);
    }
    else if(pll -> length == 1)
    {
	temp = pll -> head -> idx;
	free(pll -> head);
	pll -> head = NULL;
	pll -> tail = NULL;
	pll -> length = 0;
	return temp;
    }
    else
    {
	temp = pll -> head -> idx;
	p = pll -> head;
	pll -> head = pll -> head -> pnext;
	free(p);
	pll -> length -= 1;
	return temp;
    }
}

void BFS_on_transition_table(TRANSITIONTABLE* ptt, TRANSITIONTABLE* ptt_result, int r)
{
    int i;
    int j;
    int k;
    int current_idx;
    LINKEDELEM* pstart;
    LINKEDELEM* pend;
    LINKEDELEM* p;
    LINKEDELEM* tempp;
    init_transition_table(ptt_result, ptt -> n);
    for(i = 0; i < ptt -> n; i++)
	insert_in_transition_table(ptt_result, i, i);
    //printf("!!!!!!!!!!!!!!!!!\n");
    for(i = 0; i < ptt -> n; i++)
    {
	if((ptt -> parray)[i].length == 0)
	    continue;
	pstart = (ptt_result -> parray)[i].head; 
	pend = (ptt_result -> parray)[i].tail;
	for(j = 0; j < r; j++)
	{
	    p = pstart;
	    while(1)
	    {
		current_idx = p -> idx;
		tempp = (ptt -> parray)[current_idx].head;
		for(k = 0; k < (ptt -> parray)[current_idx].length; k++)
		{
		    if(!exist_in_linked_list(tempp -> idx, (ptt_result -> parray) + i))
			insert_in_linked_list(tempp -> idx, (ptt_result -> parray) + i);
		    tempp = tempp -> pnext;
		}

		if(p == pend)
		    break;
		p = p -> pnext;
	    }
	    if(pend -> pnext == NULL)
		break;
	    pstart = pend -> pnext;
	    pend = (ptt_result -> parray)[i].tail;
	}
    }
}
