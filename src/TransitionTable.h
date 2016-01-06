#ifndef TRANSITION_TABLE
#define TRANSITION_TABLE

typedef struct linkedelem
{
    int idx;
    struct linkedelem* pnext;
}
LINKEDELEM;

typedef struct
{
    LINKEDELEM* head;
    LINKEDELEM* tail;
    int length;
}
LINKEDLIST;

typedef struct
{
    LINKEDLIST* parray;
    int n;
}
TRANSITIONTABLE;


void init_linked_list(LINKEDLIST* pll);
void free_linked_list(LINKEDLIST* pll);
int exist_in_linked_list(int q, LINKEDLIST* pll);
void insert_in_linked_list(int q, LINKEDLIST* pll);
int pop_from_linked_list(LINKEDLIST* pll);
void init_transition_table(TRANSITIONTABLE* ptt, int m);
void free_transition_table(TRANSITIONTABLE* ptt);
void insert_in_transition_table(TRANSITIONTABLE* ptt, int fr, int to);
int exist_in_transition_table(TRANSITIONTABLE* ptt, int fr, int to);
void print_linked_list(LINKEDLIST* pll);
void print_transition_table(TRANSITIONTABLE* ptt);
void BFS_on_transition_table(TRANSITIONTABLE* ptt, TRANSITIONTABLE* ptt_result, int r);

#endif
