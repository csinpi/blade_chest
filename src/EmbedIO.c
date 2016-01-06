#include "HLogisticEmbed.h"
#include "EmbedIO.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define INT_BUF_SZ 2000000
#define DOUBLE_BUF_SZ 2000

void fprint_vec(double* vec, int len, FILE* fp)
{
	int i;
	for(i = 0; i < len; i++)
	{
		fprintf(fp, "%f", vec[i]);
		if(i == len - 1)
			fputc('\n', fp);
		else
			fputc(' ', fp);
	}
}

void fprint_int_vec(int* vec, int len, FILE* fp)
{
	int i;
	for(i = 0; i < len; i++)
	{
		fprintf(fp, "%d", vec[i]);
		if(i == len - 1)
			fputc('\n', fp);
		else
			fputc(' ', fp);
	}
}

void fprint_mat(double** mat, int m, int n, FILE* fp)
{
	int i;
	for(i = 0; i < m; i++)
		fprint_vec(mat[i], n, fp);
}

void fprint_separator(FILE* fp)
{
	int i;
	for(i = 0; i < 20; i++)
		fputc('=', fp);
	fputc('\n', fp);
}

void write_ClusterEmbedding_to_file(CEBD ce, char* filename)
{
	int bias_enabled = (ce.bias_by_cluster != NULL);
	int i;
	FILE*  fp = fopen(filename, "w");
	fprintf(fp, "num_of_clusters %d\n", ce.num_clusters);
	fprintf(fp, "num_of_songs %d\n", ce.num_songs);
	fprintf(fp, "d %d\n", ce.d);
	fprintf(fp, "bias_enabled %d\n", bias_enabled);
	fprintf(fp, "inter_cluster_transition_type %d\n", ce.inter_cluster_transition_type);
	if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5 || ce.inter_cluster_transition_type == 6)
		fprintf(fp, "num_of_portals %d\n", ce.num_portals);
	fprintf(fp, "assignment ");
	fprint_int_vec(ce.assignment, ce.num_songs, fp);
	fprint_separator(fp);
	for(i = 0; i < ce.num_clusters; i++)
	{
		if(ce.inter_cluster_transition_type == 0 || ce.inter_cluster_transition_type == 2)
		{
			if(bias_enabled)
			{
				fprint_vec(ce.bias_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_clusters, fp); 
			}
			fprint_mat(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_clusters, ce.d, fp);
		}
		else if(ce.inter_cluster_transition_type == 1)
		{
			if(bias_enabled)
			{
				fprint_vec(ce.bias_by_cluster[i], ce.num_songs_in_each_cluster[i] + 2 * (ce.num_clusters - 1), fp); 
			}
			fprint_mat(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + 2 * (ce.num_clusters - 1), ce.d, fp);
		}
		else if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 6)
		{
			if(bias_enabled)
			{
				fprint_vec(ce.bias_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_portals, fp); 
			}
			fprint_mat(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_portals, ce.d, fp);
		}
		else if(ce.inter_cluster_transition_type == 5)
		{
			if(bias_enabled)
			{
				fprint_vec(ce.bias_by_cluster[i], ce.num_songs_in_each_cluster[i] + 2 * ce.num_portals, fp); 
			}
			fprint_mat(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + 2 * ce.num_portals, ce.d, fp);
		}
		fprint_separator(fp);
	}
	fclose(fp);
}

void read_int_vec(FILE* fp, int** return_vec, int* length)
{
	char templine[INT_BUF_SZ];
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		(*length) = parse_int_line(templine, return_vec);
}

void read_double_vec(FILE* fp, double** return_vec, int* length)
{
	//fpos_t starting_pos;
	//fgetpos(fp, &starting_pos);
	char templine[INT_BUF_SZ];
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		(*length) = parse_double_line(templine, return_vec);
}

void read_double_mat(FILE* fp, double*** return_mat, int* m, int* n)
{
	fpos_t starting_pos;
	fgetpos(fp, &starting_pos);
	(*n) = 0;
	(*m) = 0;
	int d;
	int i;
	char templine[DOUBLE_BUF_SZ];
	char error_msg[200];
	while(fgets(templine, DOUBLE_BUF_SZ, fp) != NULL)
	{
		//printf("%s", templine);
		if(is_in_float_char_set(templine[0]))
			(*m)++;
		else
			break;
	}
	(*return_mat) = (double**)(malloc((*m) * sizeof(double*)));
	fsetpos(fp, &starting_pos);
	i = 0;
	while(fgets(templine, DOUBLE_BUF_SZ, fp) != NULL)
	{
		//printf("%s", templine);
		if(is_in_float_char_set(templine[0]))
		{
			d = parse_double_line(templine, (*return_mat) + i);
			i++;
			if((*n) == 0)
				((*n) = d);
			else if((*n) != d)
				error_in_read("The lines of the matrix have different lengths.");

		}
		else
			break;
	}
}

CEBD read_ClusterEmbedding_file(char* filename)
{
	int bias_enabled;
	int i;
	FILE *fp;
	fp = fopen(filename, "r");
	char templine[INT_BUF_SZ];
	//char error_msg[200];
	int m;
	int n;
	int temp_length;
	CEBD ce;
	//first line
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		ce.num_clusters = extract_tail_int(templine, "num_of_clusters ");
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		ce.num_songs = extract_tail_int(templine, "num_of_songs ");
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		ce.d = extract_tail_int(templine, "d ");
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		bias_enabled = extract_tail_int(templine, "bias_enabled ");
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		ce.inter_cluster_transition_type = extract_tail_int(templine, "inter_cluster_transition_type ");
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		parse_int_line(find_after_prefix(templine, "assignment "), &(ce.assignment));
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		assert(templine[0] == '=');

	//printf("(%d, %d, %d)\n", ce.num_clusters, ce.num_songs, ce.d);
	//print_int_vec(ce.assignment, ce.num_songs);
	
	ce.num_songs_in_each_cluster = (int*)malloc(ce.num_clusters * sizeof(int));
	ce.cluster_song_relation = (int**)malloc(ce.num_clusters * sizeof(int*));
	ce.X_by_cluster = (double***)malloc(ce.num_clusters * sizeof(double**));
	if(bias_enabled)
		ce.bias_by_cluster = (double**)malloc(ce.num_clusters * sizeof(double*));
	else
		ce.bias_by_cluster = NULL;

	int accu_num = 0;
	for(i = 0; i < ce.num_clusters; i++)
	{
		if(bias_enabled)
			read_double_vec(fp, ce.bias_by_cluster + i, &temp_length);
		read_double_mat(fp, ce.X_by_cluster + i, ce.num_songs_in_each_cluster + i, &n);
		//printf("(%d, %d)\n", m, n);

		if(n != 0 && n != ce.d)
			error_in_read("The dimensionality of an individual cluster of embedding does not match the global definition. File may be corrupted.");
		if(bias_enabled && temp_length != ce.num_songs_in_each_cluster[i])
			error_in_read("The number of bias terms mismatch the number of embedded points.");
		accu_num += ce.num_songs_in_each_cluster[i];
	}
	if((ce.inter_cluster_transition_type == 0 && accu_num != ce.num_songs + ce.num_clusters * ce.num_clusters) \
			|| (ce.inter_cluster_transition_type == 1 && accu_num != ce.num_songs + 2 * (ce.num_clusters - 1) * ce.num_clusters))
		error_in_read("The total number of songs does not match\
			   	the global definition. File may be corrupted.");

	fclose(fp);
	for(i = 0; i < ce.num_clusters; i++)
	{
		m =  find_cluster_song_list(ce.assignment, ce.num_songs, i, ce.cluster_song_relation + i);
		//printf("(%d, %d, %d)\n", m,ce.num_songs_in_each_cluster[i], ce.num_clusters);
		if(ce.inter_cluster_transition_type == 0)
			assert(m == ce.num_songs_in_each_cluster[i] - ce.num_clusters);
		else if(ce.inter_cluster_transition_type == 1)
			assert(m == ce.num_songs_in_each_cluster[i] - 2 * (ce.num_clusters - 1));
		ce.num_songs_in_each_cluster[i] = m;
	}
	//printf("SSone printing..\n");
	//for(i = 0; i < ce.num_clusters; i++)
	//	print_mat(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_clusters, ce.d);
	//printf("DDone printing..\n");

	return ce;
}

void error_in_read(char* str)
{
	printf("Error occured during reading.\n");
	if(str != NULL)
		printf("%s\n", str);
	exit(1);
}

int is_in_float_char_set(char c)
{
	return (c == '-') || ( c == '.') || ((int)c >= 48 && (int)c <= 57);
}

int is_in_int_char_set(char c)
{
	return (c == '-') || ((int)c >= 48 && (int)c <= 57);
}

int parse_int_line(char* str, int** list)
{
	int i;
	int t;
	int idx;
	char current_int[200];
	char error_msg[200];
	int length = 0;
	i = 0;
	while(str[i] != '\0' && str[i] != '\n' && str[i] != EOF)
	{
		if(str[i] == ' ')
			length++;
		i++;
	}
	length++;
	(*list) = (int*)malloc(length * sizeof(int));
	t = 0;
	idx = 0;
	i = 0;
	
	while(str[i] != '\0' && str[i] != '\n' && str[i] != EOF)
	{
		if(is_in_int_char_set(str[i]))
		{
			current_int[t] = str[i];
			t++;
		}
		else if(str[i] == ' ')
		{
			current_int[t] = '\0';
			(*list)[idx] = atoi(current_int);
			t = 0;
			idx++;
		}
		else
		{
			sprintf(error_msg, "Unrecognizeable character", str[i]);
			error_in_read(error_msg);
		}
		i++;
	}
	current_int[t] = '\0';
	(*list)[idx] = atoi(current_int);
	idx++;
	//printf("(%d, %d)\n", idx, length);
	assert(idx == length);
	return length;
}

int parse_double_line(char* str, double** list)
{
	//printf("%s", str);
	int i;
	int t;
	int idx;
	char current_double[200];
	char error_msg[200];
	int length = 0;
	i = 0;
	while(str[i] != '\0' && str[i] != '\n' && str[i] != EOF)
	{
		if(str[i] == ' ')
			length++;
		i++;
	}
	length++;
	(*list) = (double*)malloc(length * sizeof(double));
	t = 0;
	idx = 0;
	i = 0;
	
	while(str[i] != '\0' && str[i] != '\n' && str[i] != EOF)
	{
		if(is_in_float_char_set(str[i]))
		{
			current_double[t] = str[i];
			t++;
		}
		else if(str[i] == ' ')
		{
			current_double[t] = '\0';
			(*list)[idx] = atof(current_double);
			//printf("%f\n", (*list)[idx]);
			t = 0;
			idx++;
		}
		else
		{
			sprintf(error_msg, "Unrecognizeable character", str[i]);
			error_in_read(error_msg);
		}
		i++;
	}
	current_double[t] = '\0';
	(*list)[idx] = atof(current_double);
	//printf("%f\n", (*list)[idx]);
	idx++;
	//printf("(%d, %d)\n", idx, length);
	assert(idx == length);
	//print_vec(*list, length);
	//putchar('\n');
	return length;
}

int extract_tail_int(char* str, char* prefix)
{
	char int_holder[200];
	int l = strlen(prefix);
	if(strlen(str) <= l)
		error_in_read("The line does not contain the number needed.\n");
	if(strncmp(str, prefix, l) == 0)
	{
		strcpy(int_holder, str + l); 
		return atoi(int_holder);
	}
	else
		error_in_read("The format of the file is incorrect.");
}

char* find_after_prefix(char* str, char* prefix)
{
	int l = strlen(prefix);
	if(strlen(str) <= l)
		error_in_read("The line does not contain the number needed.\n");
	if(strncmp(str, prefix, l) == 0)
		return (str + l);
	else
		error_in_read("The format of the file is incorrect.");
}

double extract_ending_double(char* str, char* substr)
{
	char error_msg[1000];
	int l = strlen(substr);
	char* temp_pos = strstr(str, substr);
	if(temp_pos == NULL)
	{
		sprintf(error_msg, "Could not find the substring \"%s\" in the string \"%s\".\n", substr, str);
		error_in_read(error_msg);
	}
	else
		return atof(temp_pos + l);
}

int find_cluster_song_list(int* member_vec, int k, int cluster_idx, int** cluster_idx_list)
{
    int i;
    int cluster_song_count = 0;
    for(i = 0; i < k; i++)
	if(member_vec[i] == cluster_idx)
	    cluster_song_count++;
    (*cluster_idx_list) = (int*)malloc(cluster_song_count * sizeof(int)) ;
    int t = 0;
    for(i = 0; i < k; i++)
	if(member_vec[i] == cluster_idx)
	    (*cluster_idx_list)[t++] = i;

    //sort it
    //print_int_vec((*cluster_idx_list), cluster_song_count);
    sort_int_in_place((*cluster_idx_list), cluster_song_count, 0);
    //print_int_vec((*cluster_idx_list), cluster_song_count);
    return cluster_song_count;
}

int extract_head_int(char* str, int* pos_of_nondigit)
{
	(*pos_of_nondigit) = 0;
	char buffer[256];
	while(str[(*pos_of_nondigit)] >= '0' && str[(*pos_of_nondigit)] <= '9')
		(*pos_of_nondigit)++;
	memcpy(buffer, str, (*pos_of_nondigit) * sizeof(char));
	buffer[(*pos_of_nondigit)] = '\0';
	return atoi(buffer);
}
