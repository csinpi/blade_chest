#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "LogisticEmbed_common.h"
#define MAXLINE 100000
#define BIGNUMBER 100000
#define TEMPARRAYSIZE 1000



PDATA read_playlists_data(char* filename, int verbose)
{
	PDATA pd;
	//Two passes. The first pass decides the total number of lines.
	char c;
	FILE *fp;
	int line_count = 1;


	//printf("Reading data from %s......\n", filename);fflush(stdout);
	fp = fopen(filename, "r");
	if (fp == NULL)
	{
		printf("Error in opening the file %s.\n", filename);fflush(stdout);
		exit(1);
	}

	c = getc(fp);
	while(c != EOF)
	{
		if(c == '\n')
			line_count++;
		c = getc(fp);
	}
	pd.num_playlists = line_count - 3;
	fclose(fp);
	pd.playlists_length = (int*)malloc(pd.num_playlists * sizeof(int));
	pd.playlists = (int**)malloc(pd.num_playlists * sizeof(int*));
	//Second pass
	fp = fopen(filename, "r");
	if (fp == NULL)
	{
		printf("Error in opening the file.\n");fflush(stdout);
		exit(1);
	}

	line_count = 1;
	int array_pos = 0;
	int playlist_pos = 0;
	int temp_num;
	char* temp_array;
	int* temp_playlists_array;
	int i;


	temp_array = (char*)malloc(TEMPARRAYSIZE * sizeof(char));
	temp_playlists_array = (int*)malloc(BIGNUMBER * sizeof(int));
	c = getc(fp);
	while(c != EOF)
	{
		if(c == ' ')
		{
			if(array_pos > 0)
			{
				temp_array[array_pos] = '\0';
				temp_num = atoi(temp_array);
			}
			free(temp_array);
			temp_array = (char*)malloc(TEMPARRAYSIZE * sizeof(char));
			array_pos = 0;
			temp_playlists_array[playlist_pos] = temp_num;
			playlist_pos += 1;
			if(playlist_pos >= BIGNUMBER)
			{
				printf("Playlists array full!!!\n");fflush(stdout);
				exit(1);
			}

			//printf("%d\n", temp_num);
		}
		else if(c == '\n')
		{
			//The first line is all_ids
			if(line_count == 1)
			{
				pd.num_songs = playlist_pos;
				pd.all_ids = (int*)malloc(pd.num_songs * sizeof(int));
				for(i = 0; i < pd.num_songs; i++)
					pd.all_ids[i] = temp_playlists_array[i];
			}
			else if(line_count == 2)
			{
				//sanity check
				if(playlist_pos != pd.num_songs)
				{
					printf("Numbers of unique songs mismatch!!!\n");fflush(stdout);
					exit(1);
				}
				pd.id_counts = (int*)malloc(pd.num_songs * sizeof(int));
				for(i = 0; i < pd.num_songs; i++)
					pd.id_counts[i] = temp_playlists_array[i];
			}
			else
			{
				pd.playlists_length[line_count - 3] = playlist_pos;
				pd.playlists[line_count - 3] = (int*)malloc(playlist_pos * sizeof(int));
				for(i = 0; i < playlist_pos; i++)
					pd.playlists[line_count - 3][i] = temp_playlists_array[i];
			}

			free(temp_playlists_array);
			temp_playlists_array = (int*)malloc(BIGNUMBER * sizeof(int));
			playlist_pos = 0;
			line_count += 1;
		}
		else if((c >= '0' && c <= '9') ||(c == '-'))
		{
			temp_array[array_pos] = c;
			array_pos += 1;
			if(array_pos >= TEMPARRAYSIZE)
			{
				printf("Temp array full!!!\n");fflush(stdout);
				exit(1);
			}
		}
		else
		{
			printf("Found an invalid character %c\n", c);fflush(stdout);
			exit(1);
		}
		c = getc(fp);
	}
	free(temp_array);
	free(temp_playlists_array);
	fclose(fp);


	pd.num_appearance = 0;
	for(i = 0; i < pd.num_playlists; i++)
		pd.num_appearance += pd.playlists_length[i];


	pd.num_transitions = 0;
	for(i = 0; i < pd.num_playlists; i++)
		if(pd.playlists_length[i] > 1)
			pd.num_transitions += (pd.playlists_length[i] - 1);

	//printf("Totally %d transitions.\n", pd.num_transitions);fflush(stdout);
	if(verbose)
	{
		printf("Totally %d playlists.\n", pd.num_playlists);fflush(stdout);
		printf("Totally %d unique songs.\n", pd.num_songs);fflush(stdout);
		printf("Totally %d appearances of song.\n", pd.num_appearance);fflush(stdout);
		printf("Totally %d transitions.\n", pd.num_transitions);fflush(stdout);
	}

	return pd;
}

void print_playlists_data_stats(PDATA pd)
{
	printf("Totally %d playlists.\n", pd.num_playlists);fflush(stdout);
	printf("Totally %d unique songs.\n", pd.num_songs);fflush(stdout);
	printf("Totally %d appearances of song.\n", pd.num_appearance);fflush(stdout);
}



void free_playlists_data(PDATA pd)
{
	int i;
	for(i = 0; i < pd.num_playlists;i++)
		free(pd.playlists[i]);
	free(pd.all_ids);
	free(pd.id_counts);
	free(pd.playlists_length);
	free(pd.playlists);
}

void print_playlists_data(PDATA pd)
{
	int i;
	int j;
	for(i = 0; i < pd.num_songs; i++)
		printf("%d\n", pd.all_ids[i]);fflush(stdout);
	for(i = 0; i < pd.num_songs; i++)
		printf("%d\n", pd.id_counts[i]);fflush(stdout);
	for(i = 0; i < pd.num_playlists;i++)
		for(j = 0; j < pd.playlists_length[i]; j++)
			printf("%d\n", pd.playlists[i][j]);fflush(stdout);
	free(pd.playlists[i]);
}

TDATA read_tag_data(char* filename)
{
	TDATA td;
	FILE* fp = fopen(filename, "r");
	char temp_line[2000]; 
	td.num_songs = 0;
	int i;
	int j;
	int t;
	int pos;
	char num_buf[20];
	int current_tag;

	//first pass, get the total number of songs
	while(fgets(temp_line, 2000, fp) != NULL)
	{
		td.num_songs += 1;
		//printf("%s", temp_line);
	}
	printf("Total number of songs: %d\n", td.num_songs);
	td.num_tags_for_song = (int*)calloc(td.num_songs, sizeof(int));
	td.tags = (int**)malloc(td.num_songs * sizeof(int*));
	for(i = 0; i < td.num_songs; i++)
		td.tags[i] = NULL;

	rewind(fp);

	//second pass, get the number of tags for each song
	for(i = 0; i < td.num_songs; i++)
	{
		if(fgets(temp_line, 2000, fp) == NULL)
		{
			printf("Error in reading file.\n");
			exit(1);
		}
		//printf("%s", temp_line);
		if(temp_line[0] == '#')
			continue;
		else
		{
			td.num_tags_for_song[i] = 1;
			j = 0;
			while(1)
			{
				if(j >= 2000)
				{
					printf("Error: we need a larger line buffer.\n");
					exit(1);
				}
				if(temp_line[j] == '\n')
					break;
				if(temp_line[j] == ' ')
					td.num_tags_for_song[i] += 1;
				j++;
			}
			td.tags[i] = (int*)malloc(td.num_tags_for_song[i] * sizeof(int));
		}
	}

	/*
	   for(i = 0; i < td.num_songs; i++)
	   printf("%d\n", td.num_tags_for_song[i]);
	   */

	rewind(fp);

	td.num_tags = 0;
	//third pass, get the ids for each song
	for(i = 0; i < td.num_songs; i++)
	{
		if(fgets(temp_line, 2000, fp) == NULL)
		{
			printf("Error in reading file.\n");
			exit(1);
		}
		if(temp_line[0] == '#')
			continue;
		else
		{
			j = 0;
			t = 0;
			pos = 0;
			while(1)
			{
				if(temp_line[j] >= '0' && temp_line[j] <= '9')
				{
					num_buf[t] = temp_line[j];
					t++;
					j++;
				}
				else if(temp_line[j] == ' ')
				{
					num_buf[t] = '\0';
					//printf("%s\n", num_buf);
					current_tag = atoi(num_buf);
					td.tags[i][pos] = current_tag;
					td.num_tags = td.num_tags > (current_tag + 1) ? td.num_tags:current_tag + 1;
					//printf("%d\n", td.tags[i][pos]);
					t = 0;
					pos++;
					j++;
				}
				else if(temp_line[j] == '\n')
				{
					num_buf[t] = '\0';
					current_tag = atoi(num_buf);
					td.tags[i][pos] = current_tag;
					td.num_tags = td.num_tags > (current_tag + 1) ? td.num_tags:current_tag + 1;
					//printf("%d\n", td.tags[i][pos]);
					break;
				}
				else
				{
					printf("Error in reading tag file: %s\n", filename);
					exit(1);
				}

			}
		}
	}



	fclose(fp);
	return td;
}
void free_tag_data(TDATA td)
{
	int i;
	for(i = 0; i < td.num_songs; i++)
		if(td.num_tags_for_song[i] > 0)
			free(td.tags[i]);
	free(td.num_tags_for_song);
	free(td.tags);
}


void print_tag_data(TDATA td)
{
	int i;
	int j;
	for(i = 0; i < td.num_songs; i++)
	{
		if(td.num_tags_for_song[i] == 0)
		{
			printf("#\n");
			continue;
		}
		for(j = 0; j < td.num_tags_for_song[i]; j++)
		{
			printf("%d", td.tags[i][j]);
			if(j == td.num_tags_for_song[i] - 1)
				putchar('\n');
			else
				putchar(' ');
		}
	}
}



void Array2Dcopy(double** src, double** dest, int m, int n)
{
	int i;
	int j;
	/*
	   for(i = 0; i < m; i++)
	   for(j = 0; j < n; j++)
	   dest[i][j] = src[i][j];
	   */
	for(i = 0; i < m; i++)
		memcpy(dest[i], src[i], n * sizeof(double));
}

void Array2Dcopy_int(int** src, int** dest, int m, int n)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
		memcpy(dest[i], src[i], n * sizeof(int));
}

void Array2Dexclcopy(double** src, double** dest, int m, int n, int excl_idx)
{
	int i;
	int j;
	int current_dest_line = 0;
	for(i = 0; i < m; i++)
	{
		if(i != excl_idx)
		{
			for(j = 0; j < n; j++)
			{
				dest[current_dest_line][j] = src[i][j];
			}
			current_dest_line++;
		}
	}
}
void Array2Dfree(double** X, int m, int n)
{
	int i;
	for(i = 0; i < m; i++)
		free(X[i]);
	free(X);
}

void Array2Dfree_int(int** X, int m, int n)
{
	int i;
	for(i = 0; i < m; i++)
		free(X[i]);
	free(X);
}
/*
   double** zerosarray(int m, int n)
   {
   double** X;
   int i;
   int j;
   X = (double**)malloc(m * sizeof(double*));
   for(i = 0; i < m; i++)
   X[i] = (double*)malloc(n * sizeof(double));
   for(i = 0; i < m; i++)
   for(j = 0; j < n; j++)
   X[i][j] = 0.0;
   return X;
   }
   */
double** zerosarray(int m, int n)
{
	double** X;
	int i;
	int j;
	X = (double**)malloc(m * sizeof(double*));
	for(i = 0; i < m; i++)
		X[i] = (double*)calloc(n, sizeof(double));
	return X;
}

int** zerosarray_int(int m, int n)
{
	int** X;
	int i;
	int j;
	X = (int**)malloc(m * sizeof(int*));
	for(i = 0; i < m; i++)
		X[i] = (int*)calloc(n, sizeof(int));
	return X;
}

double** randarray(int m, int n, double c)
{
	//srand(time(NULL));
	double** X;
	int i;
	int j;
	X = (double**)malloc(m * sizeof(double*));
	for(i = 0; i < m; i++)
		X[i] = (double*)malloc(n * sizeof(double));
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			X[i][j] = ((double)rand()) / ((double) RAND_MAX) * c;
	return X;
}

void randfill(double** X, int m, int n, double c)
{
	//srand(time(NULL));
	int i;
	int j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			X[i][j] = ((double)rand()) / ((double) RAND_MAX) * c;
}

double* randvec(int m, double c)
{
	//srand(time(NULL));
	double* X;
	int i;
	int j;
	X = (double*)malloc(m * sizeof(double));
	for(i = 0; i < m; i++)
		X[i] = ((double)rand()) / ((double) RAND_MAX) * c;
	return X;
}

double* linspace_vec(double lower, double upper, int l)
{
	double* X;
	int i;
	X = (double*)malloc(l * sizeof(double));
	double interval = (upper - lower) / (double)(l - 1);
	for(i = 0; i < l; i++)
		X[i] = lower + i * interval;
	return X;
}

double innerprod(double* vec1, double* vec2, int length)
{
	double r = 0.0;
	int i;
	for(i = 0; i < length; i++)
	{
		r += vec1[i] * vec2[i];
	}
	return r;

}

void exp_on_vec(double* vec, int length)
{
	int i;
	for(i = 0; i < length; i++)
		vec[i] = exp(vec[i]);
}

void pow_on_vec(double* vec, int length, double the_power)
{
	int i;
	for(i = 0; i < length; i++)
		vec[i] = pow(vec[i], the_power);
}

void log_on_vec(double* vec, int length)
{
	int i;
	for(i = 0; i < length; i++)
		vec[i] = log(vec[i]);
}


void sum_along_direct(double** X, double* vec, int m, int n, int direct)
{
	int i;
	int j;
	if(direct == 0)
	{
		for(i = 0; i < n; i++)
			vec[i] = 0.0;
		for(i = 0; i < n; i++)
			for(j = 0; j < m; j++)
				vec[i] += X[j][i];
	}
	else if(direct ==1)
	{
		for(i = 0; i < m; i++)
			vec[i] = 0.0;
		for(i = 0; i < m; i++)
			for(j = 0; j < n; j++)
				vec[i] += X[i][j];
	}
	else
	{
		printf("Invalid direction!!!\n");fflush(stdout);
		exit(1);
	}
}

double mat_norm_diff(double** X1, double** X2, int m, int n)
{
	int i;
	int j;
	double r = 0.0;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			r += pow(X1[i][j] - X2[i][j], 2);
	return(sqrt(r));

}



void add_vec(double* org_vec, double* update_vec, int length, double coeff)
{
	int i;
	for(i = 0; i < length; i++)
		org_vec[i] += coeff * update_vec[i];
}

void add_mat(double** org_mat, double** update_mat, int m, int n, double coeff)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			org_mat[i][j] += coeff * update_mat[i][j];
}

void tile_vec(double* vec, double** mat, int m, int n, int direct)
{
	int i;
	int j;
	if(direct == 0)
	{
		for(i = 0; i < m; i++)
			for(j = 0; j < n; j++)
				mat[i][j] = vec[j];
	}
	else if(direct == 1)
	{
		for(i = 0; i < m; i++)
			for(j = 0; j < n; j++)
				mat[i][j] = vec[i];
	}
	else
	{
		printf("Invalid direction!!!\n");fflush(stdout);
		exit(1);
	}

}

void mat_mult(double** src1, double** src2, double** dest, int m, int n)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			dest[i][j] = src1[i][j] * src2[i][j];
}

void vec_mult(double* vec1, double* vec2, double* dest, int length)
{
	int i;
	for(i = 0; i < length; i++)
		dest[i] = vec1[i] * vec2[i];
}

void mat_neg_euc(double** src1, double** src2, double** dest, int m, int n)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			dest[i][j] = - pow((src1[i][j] - src2[i][j]), 2);
}

double sum_vec(double* vec, int length)
{
	int i;
	double r = 0.0;
	for(i = 0; i < length; i++)
		r += vec[i];
	return(r);
}

void scale_vec(double* vec, int length, double scale)
{
	int i;
	for(i = 0; i < length; i++)
		vec[i] *= scale;


}

void scale_mat(double** X, int m, int n, double scale)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			X[i][j] = scale * X[i][j];
}

double frob_norm(double** X, int m, int n)
{
	double r = 0.0;
	int i;
	int j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			r += pow(X[i][j], 2);
	return(sqrt(r));
}

double vec_norm(double* vec, int length)
{
	double r = 0.0;
	int i;
	for(i = 0;i < length; i++)
		r += pow(vec[i], 2);
	return(sqrt(r));
}

double avg_norm(double** X, int m, int n)
{
	int i;
	double r = 0.0;
	for(i = 0; i < m; i++)
		r += vec_norm(X[i], n);
	return r / ((double) m);
}


void Veccopy(double* src, double* dest, int length)
{
	memcpy(dest, src, length * sizeof(double));
	/*
	   int i;
	   for(i = 0; i < length; i++)
	   dest[i] = src[i];
	   */
}



double Cal_log_likelihood(double** X, PDATA pd, int d)
{

	double ll = 0.0;
	int i;
	int j;
	int a;
	int b;
	double denom;
	double* temp_vec = (double*)malloc((pd.num_songs - 1) * sizeof(double));
	double** tempX = zerosarray(pd.num_songs - 1, d);
	double** auxX = zerosarray(pd.num_songs - 1, d);

	time_t start_time = time(NULL);
	for(i = 0; i < pd.num_playlists; i++)
	{
		for(j = 0; j < pd.playlists_length[i] - 1; j++)
		{
			a = pd.playlists[i][j];
			b = pd.playlists[i][j + 1];

			if(a != b)
			{
				Array2Dexclcopy(X, tempX, pd.num_songs, d, a);
				tile_vec(X[a], auxX, pd.num_songs - 1, d, 0);
				mat_mult(tempX, auxX, auxX, pd.num_songs - 1, d);
				sum_along_direct(auxX, temp_vec, pd.num_songs - 1, d, 1);
				exp_on_vec(temp_vec, pd.num_songs - 1);
				denom = sum_vec(temp_vec, pd.num_songs - 1);
				ll += (innerprod(X[a], X[b], d) - log(denom));
				/*
				   scale_vec(temp_vec, pd.num_songs - 1, 1.0/denom);
				   tile_vec(temp_vec, auxX, pd.num_songs - 1, d, 1);
				   mat_mult(tempX, auxX, tempX, pd.num_songs - 1, d);
				   sum_along_direct(tempX, dvec, pd.num_songs - 1, d, 0);
				   add_vec(subgrad_a, dvec, d, -1.0);
				   add_vec(subgrad_b, X[a], d, - (exp(innerprod(X[a], X[b])) / denom));

				   add_vec(X[a], subgrad_a, d, ita);
				   add_vec(X[b], subgrad_b, d, ita);
				   scale_vec(X[a], d, 1.0 / vec_norm(X[a], d));
				   scale_vec(X[b], d, 1.0 / vec_norm(X[b], d));
				   */
			}
		}
	}
	printf("Calculating Likelihood took %d seconds.\n", (int)(time(NULL) - start_time));fflush(stdout);
	free(temp_vec);
	Array2Dfree(tempX, pd.num_songs - 1, d);
	Array2Dfree(auxX, pd.num_songs -1, d);
	return ll;
}


void write_embedding_to_file(double** X, int m, int n, char* filename, double* bias_terms)
{
	FILE*  fp = fopen(filename, "w");
	int i;
	int j;
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
		{
			fprintf(fp, "%f", X[i][j]);
			if(j != n - 1)
				fputc(' ', fp);
			else if(j == n - 1 && i != m - 1)
				fputc('\n', fp);
		}
	}

	if (bias_terms != 0) {
		fprintf(fp, "\n## bias terms after this line\n");
		for (i = 0; i < m; i++) {
			fprintf(fp, "%f", bias_terms[i]);
			if (i < m - 1)
				fputc('\n', fp);
		}
	}

	fclose(fp);
}

int split_double_line(char* str, double* parray)
{
	char num_str[1000];
	int i = 0;
	int j = 0;
	int num_count = 0;
	while(1)
	{
		if(str[i] == ' ' || str[i] == '\0')
		{
			if(j == 0)
			{
				printf("Error in reading formatted file.\n");
				exit(1);
			}
			else
			{
				num_str[j]  = '\0';
				//printf("%s\n", num_str);
				if(parray != NULL)
					parray[num_count] = atof(num_str);
				num_count++;
				j = 0;
			}
			if(str[i] == '\0')
				break;
		}
		else if((str[i] >= 48 && str[i] <= 57) || (str[i] == '.') || (str[i] == '-') || (str[i] == 'e') || (str[i] == 'E') || (str[i] == '+'))
		{
			num_str[j] = str[i];
			j++;
		}
    else if (str[i] == 'X') {
      num_str[j] = '0';
      j++;
    }
		else
		{
			printf("Unexpected character found in the file.\n");
			putchar(str[i]);
      printf("\n");
			exit(1);
		}
		i++;
	}
	return num_count;
}

char* remove_eof(char* str)
{
	int i = 0;
	while(str[i] != '\0')
	{
		if(str[i] == '\n')
			str[i] = '\0';
		i++;
	}
	return str;
}

double** read_matrix_file(char* filename, int* m, int* n, double** bias_terms)
{
	int i;
	char mystring[100000];
	FILE* pf = fopen(filename, "r");
	if(pf == NULL)
	{
		printf("Error opening file\n");
		exit(1);
	}

	int line_count = 0;
	int col_count;
	int temp;
	while(fgets(mystring, 100000, pf) != NULL)
	{
		if (remove_eof(mystring)[0] == '#') {
			// marker for begin of bias term
			if (bias_terms == 0) {
				printf("Error: bias_terms zero pointer in read_matrix_file\n");
				exit(1);
			}
			else {
				bias_terms[0] = (double*) calloc(line_count, sizeof(double));
				int line_pos = 0;
				while(fgets(mystring, 100000, pf) != NULL) {
					split_double_line(remove_eof(mystring),
							bias_terms[0] + line_pos);
					line_pos++;
				}
				break;
			}
		}
		line_count++;
		col_count = split_double_line(remove_eof(mystring), NULL);
		//printf("%s\n", remove_eof(mystring));
		//printf("%d\n", split_double_line(mystring, NULL));
	}

	double** X = zerosarray(line_count, col_count);

	rewind(pf);
	line_count = 0;
	while(fgets(mystring, 100000, pf) != NULL)
	{
		if (remove_eof(mystring)[0] == '#')
			break;
		temp = split_double_line(remove_eof(mystring), X[line_count]);
		line_count++;
		if(temp != col_count)
		{
			printf("It is not a matrix file.\n");
			exit(1);
		}
	}
	fclose(pf);
	*m = line_count;
	*n = col_count;
	return X;
}

void Cal_exp_affinity_mat(double** X, double** A, int m, int n)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
	{
		for(j = i; j < m; j++)
		{
			A[i][j] = exp(innerprod(X[i], X[j], n));
			A[j][i] = A[i][j];
		}
	}
}

void Vecexccopy(double* src, double* dest, int length, int exc)
{
	int i;
	int j = 0;
	for(i = 0; i < length; i++)
	{
		if(i != exc)
		{
			dest[j] = src[i];
			j++;
		}
	}

}

void Update_affinity_mat(double**X, double** A, int idx, int m, int n)
{
	int i;
	for(i = 0; i < m; i++)
	{
		A[idx][i] = exp(innerprod(X[idx], X[i], n));
		A[i][idx] = A[idx][i];
	}
}

void insert_in_Nlist(NLIST* nl, int to_insert)
{
	if(nl == NULL)
		return;
	//Empty list
	if(nl -> pheader == NULL)
	{
		nl -> pheader = (NNODE*)malloc(sizeof(NNODE));
		nl -> pheader -> id = to_insert;
		nl -> pheader -> pnext = NULL;
		nl -> length += 1;
	}
	else
	{
		if(nl -> pheader -> id > to_insert)
		{
			NNODE* tempp = (NNODE*)malloc(sizeof(NNODE));
			tempp -> id = to_insert;
			tempp -> pnext = nl -> pheader;
			nl -> pheader = tempp;
			nl -> length += 1;
		}
		else if(nl -> pheader -> id < to_insert)
		{
			NNODE* pcurrent = nl -> pheader;
			while(pcurrent -> pnext != NULL && pcurrent -> pnext -> id < to_insert)
				pcurrent = pcurrent -> pnext;
			//End of the list
			if(pcurrent -> pnext == NULL)
			{
				pcurrent -> pnext = (NNODE*)malloc(sizeof(NNODE));
				pcurrent -> pnext -> id = to_insert;
				pcurrent -> pnext -> pnext = NULL;
				nl -> length += 1;
			}
			else if(pcurrent -> pnext -> id > to_insert)
			{
				NNODE* tempp =(NNODE*)malloc(sizeof(NNODE));
				tempp -> id = to_insert;
				tempp -> pnext = pcurrent -> pnext;
				pcurrent -> pnext = tempp;
				nl -> length += 1;
			}
		}
	}
}

void free_Nlist(NLIST* nl)
{
	if(nl != NULL)
	{
		NNODE* p = nl -> pheader;
		NNODE* tempp;
		while(p != NULL)
		{
			tempp = p;
			p = p -> pnext;
			free(tempp);
		}

	}

	/*
	   free_Nnode(nl -> pheader);
	   free(nl);
	   */
}

/*
   void free_Nnode(NNODE* pnn)
   {
   if(pnn == NULL)
   return;
   else if(pnn -> pnext == NULL)
   free(pnn);
   else
   {
   free_Nnode(pnn -> pnext);
   free(pnn);
   }
   }
   */

NLIST** find_neighbors(PDATA pd)
{
	int i;
	int j;
	int a;
	int b;
	NLIST** p = (NLIST**)malloc(pd.num_songs * sizeof(NLIST*));
	if(p == NULL)
	{
		printf("Cannot malloc 2d NLIST array\n");
		exit(1);
	}
	for(i = 0; i < pd.num_songs; i++)
	{
		//Init empty lists
		p[i] = (NLIST*)malloc(sizeof(NLIST));
		p[i] -> length = 0;
		p[i] -> pheader = NULL;
	}
	for(i = 0; i < pd.num_playlists; i++)
	{
		if(pd.playlists_length[i] > 1)
		{
			for(j = 0; j < pd.playlists_length[i] - 1; j++)
			{
				a = pd.playlists[i][j];
				b = pd.playlists[i][j + 1];
				if (a != b)
					insert_in_Nlist(p[a], b);
			}
		}
	}
	//for(i = 0; i < pd.num_songs; i++)
	//printf("%d\n", p[i] -> length);
	return p;
}

void print_nlist(NLIST* p)
{
	if(p == NULL)
		return;
	else
	{
		NNODE* current_p = p -> pheader;
		while(current_p != NULL)
		{
			printf("%d ", current_p -> id);
			current_p = current_p -> pnext;
		}
		putchar('\n');
	}

}


void Array2Dcopy_nn(double** src, double** dest, int d, NLIST* p)
{
	int i;
	int j;
	if(p -> length <= 0)
	{
		printf("Trying to build a matrix from an empty neighbor list\n");
		exit(1);
	}

	NNODE* pc = p -> pheader;

	for(i = 0; i < p -> length; i++)
	{
		for(j = 0; j < d; j++)
			dest[i][j] = src[pc -> id][j];
		pc = pc -> pnext;
	}
}



double vec_sq_dist(double* x1, double* x2, int length)
{
	int i;
	double result = 0.0;
	for(i = 0; i < length; i++)
		result += pow(x1[i] - x2[i], 2);
	return result;
}

void vec_mat_sum(double* vec, double vec_c, double** mat, double mat_c, double** dest, int m, int n)
{
	int i;
	int j;
	for(i = 0; i < m ; i++)
		for(j = 0; j < n;  j++)
			dest[i][j] = vec[j] * vec_c + mat[i][j] * mat_c;
}

void vec_scalar_sum(double* vec, double sc, int length)
{
	int i;
	for(i = 0; i < length; i++)
		vec[i] += sc;
}

int* merge_two_lists(int* total_length, int* list1, int length1, int* list2, int length2)
{
	int* list_short;
	int length_short;
	int* list_long;
	int length_long;
	length_short = (length1 < length2)? length1 : length2;
	length_long = (length1 < length2)? length2 : length1;
	list_short = (length1 < length2)? list1 : list2;
	list_long = (length1 < length2)? list2 : list1;
	int* total_list = (int*)calloc((length_short + length_long), sizeof(int));
	int i;
	for(i = 0; i < length_short; i++)
		total_list[i] = list_short[i];
	*total_length = length_short;
	for(i = 0; i < length_long; i++)
	{
		if(!exist_in_list(list_long[i], list_short, length_short))
		{
			total_list[*total_length] = list_long[i];
			(*total_length) += 1;
		}
	}
	total_list = (int*)realloc(total_list, (*total_length) * sizeof(int));
	return total_list;
}
int exist_in_list(int target, int* list, int length)
{
	int i;
	for(i = 0; i < length; i++)
		if(target == list[i])
			return 1;
	return 0;
}

int find_in_list(int target, int* list, int length)
{
	int i;
	for(i = 0; i < length; i++)
		if(target == list[i])
			return i;
	return -1;
}

int find_in_sorted_list(int target, int* list, int length, int descend)
//{
//	int i;
//	for(i = 0; i < length; i++)
//	{
//		if(target == list[i])
//			return i;
//		else if((descend && target > list[i]) || ((!descend) && target < list[i])) 
//			return -1;
//	}
//
//	return -1;
//}
{
	int interval_start = 0;
	int interval_length = length;
	int mid_idx;
	while(interval_length > 1)
	{
		mid_idx = interval_start + interval_length / 2;
		if(list[mid_idx] == target)
			return mid_idx;
		if((list[mid_idx] > target && descend) || (list[mid_idx] < target && !descend))
		{
			interval_length = interval_start + interval_length - mid_idx - 1;
			interval_start = mid_idx + 1;
		}
		else
			interval_length = mid_idx - interval_start;
	}
	if(list[interval_start] == target)
		return interval_start;
	else
		return -1;
}

int exist_in_nonneg_list(int target, int* list, int length)
{
	int i;
	for(i = 0; i < length; i++)
	{
		if(target == list[i])
			return 1;
		if(list[i] < 0)
			return 0;
	}
	return 0;
}

void calculate_realX(double** X, double** realX, TDATA td, int k, int m, int d, int num_points)
{
	int i;
	int j;
	int s;

	/*
	   for(s = 0; s < num_points; s++)
	   {
	   Array2Dcopy(X + s * (k + m), realX + s * k, k, d);
	   for(i = 0; i < k; i++)
	   for(j = 0; j < td.num_tags_for_song[i]; j++)
	   add_vec(realX[i + s * k], X[td.tags[i][j] + k + s * (k + m)], d, 1.0 / ((double) td.num_tags_for_song[i]));
	   }
	   */
	Array2Dcopy(X , realX, k * num_points, d);
	for(s = 0; s < num_points; s++)
	{
		for(i = 0; i < k; i++)
			for(j = 0; j < td.num_tags_for_song[i]; j++)
				add_vec(realX[i + s * k], X[td.tags[i][j] + s * m + num_points * k], d, 1.0 / ((double) td.num_tags_for_song[i]));
	}
}



//int* kmeans(double** X, int m, int n, int k, double eps, int verb)
void kmeans(double** X, int m, int n, int k, double eps, int verb, int* assignment, double*** centroids_for_return)
{
	int i;
	int j;

	if(verb)
		printf("Generating random starting points.\n");

	//srand(time(NULL));
	int* centroids_idx = (int*)malloc(k * sizeof(int));
	for(i = 0; i < k; i++)
		centroids_idx[i] = -1;
	j = 0;
	while(j < k)
	{
		int temp_idx = rand() % m;
		if(!exist_in_nonneg_list(temp_idx, centroids_idx, k))
		{
			centroids_idx[j] = temp_idx;
			j++;
		}
	}
	if(verb)
	{
		for(i = 0; i < k; i++)
			printf("%d\n", centroids_idx[i]);
	}

	double** centroids = zerosarray(k, n); 
	for(i = 0; i < k; i++)
		Veccopy(X[centroids_idx[i]], centroids[i], n);
	free(centroids_idx);

	//int* assignment = (int*)malloc(m * sizeof(int));
	int* N = (int*)calloc(k, sizeof(int));
	double* rou = (double*)calloc(k, sizeof(double));
	double J = 0.0;

	//Initial assignment
	if(verb)
		printf("Making initial assignments.\n");
	for(i = 0; i < m; i++)
	{
		double min_dist = 100000000.0;
		for(j = 0; j < k; j++)
		{
			double temp_dist = vec_sq_dist(X[i], centroids[j], n);
			if(temp_dist < min_dist)
			{
				min_dist = temp_dist;
				assignment[i] = j;
			}
		}
		//J += min_dist;
		N[assignment[i]] += 1;
	}

	if(verb)
		printf("Calculating initial centroids and potential function.\n");

	for(i = 0; i < k; i++)
		for(j = 0; j < n; j++)
			centroids[i][j] = 0.0;

	for(i = 0; i < m; i++)
		add_vec(centroids[assignment[i]], X[i], n, 1.0 / ((double)(N[assignment[i]])));
	for(i = 0; i < m; i++)
		J += vec_sq_dist(X[i], centroids[assignment[i]], n);

	if(verb)
	{
		for(i = 0; i < k; i++)
			printf("%d\n", N[i]);
	}


	double lastJ;

	int t = 0;
	while(1)
	{
		t++;
		if(verb)
			printf("Iteration %d...\n", t);
		lastJ = J;
		for(i = 0; i < m; i++)
		{
			if(N[assignment[i]] == 1)
				continue;
			for(j = 0; j < k; j++)
			{
				if(j != assignment[i])
					rou[j] = vec_sq_dist(X[i], centroids[j], n) * (double)N[j] / ((double)(N[j] + 1));
				else
					rou[j] = vec_sq_dist(X[i], centroids[j], n) * (double)N[j] / ((double)(N[j] - 1));
			}
			int min_idx = 0;
			double min_val = 10000000000.0;
			for(j = 0; j < k; j++)
			{
				if(rou[j] < min_val)
				{
					min_idx = j;
					min_val = rou[j];
				}
			}
			if(min_idx != assignment[i])
			{
				scale_vec(centroids[assignment[i]], n, ((double)N[assignment[i]]) / ((double)(N[assignment[i]] - 1)));
				add_vec(centroids[assignment[i]], X[i], n, (-1.0) /((double)(N[assignment[i]] - 1)));
				scale_vec(centroids[min_idx], n, ((double)N[min_idx]) / ((double)(N[min_idx] + 1)));
				add_vec(centroids[min_idx], X[i], n, (1.0) /((double)(N[min_idx] + 1)));
				N[assignment[i]] -= 1;
				N[min_idx] += 1;
				J = J + min_val - rou[assignment[i]];
				assignment[i] = min_idx;
			}
		}

		if(verb)
			printf("Potential function is %f.\n", J);

		if(lastJ - J < eps)
			break;
	}

	//sanity check
	if(verb)
	{
		int count = 0;
		for(i = 0; i < k; i++)
			count += N[i];
		assert(count == m);
	}

	if(centroids_for_return == NULL)
		Array2Dfree(centroids, k, n);
	else
		(*centroids_for_return) = centroids;
	free(N);
	free(rou);
	//return assignment;
}

int* read_hash(char* hash_filename, int* num_idx)
{
	*num_idx = 0;
	char temp_line[2000];
	int* hash = (int*)malloc(BIGNUMBER * sizeof(int));
	FILE* fp = fopen(hash_filename, "r");
	int t;
	while(fgets(temp_line, 2000, fp) != NULL)
	{
		t = 0;
		while(temp_line[t] >= '0' && temp_line[t] <= '9')
			t++;  
		temp_line[t] = '\0';
		hash[*num_idx] = atoi(temp_line);
		(*num_idx)++;
	}
	fclose(fp);
	hash = (int*)realloc(hash, (*num_idx) * sizeof(int));
	return hash;
}

void calculate_realX_with_hash(double** X, double** realX, TDATA td, int k, int m, int d, int num_points, int k_train, int* hash)
{
	int i;
	int j;
	int s;
	int* reverse_hash = (int*)malloc(k * sizeof(int));
	for(i = 0; i < k; i++)
		reverse_hash[i] = -1;
	for(i = 0; i < k_train; i++)
		reverse_hash[hash[i]] = i;

	for(s = 0; s < num_points; s++)
	{
		for(i = 0; i < k; i++)
		{
			//song exists in training set
			if(reverse_hash[i] >= 0)
				Veccopy(X[reverse_hash[i] + s * k_train], realX[i + s * k], d);
			for(j = 0; j < td.num_tags_for_song[i]; j++)
				add_vec(realX[i + s * k], X[td.tags[i][j] + s * m + num_points * k_train], d, 1.0 / ((double) td.num_tags_for_song[i]));
		}
	}
	free(reverse_hash);
}

int* get_test_ids(int k, int k_train, int* train_ids)
{
	int k_test = k - k_train;
	int* test_ids = (int*)malloc(k_test * sizeof(int));
	int i;
	for(i = 0 ; i < k_test; i++)
		test_ids[i] = -1;
	int t = 0;
	for(i = 0; i < k; i++)
	{
		if(!exist_in_list(i, train_ids, k_train))
		{
			test_ids[t] = i;
			t++;
		}
	}
	//sanity check
	for(i = 0 ; i < k_test; i++)
		assert(test_ids[i] >= 0);

	return test_ids;
}

void int_list_copy(int* src, int* dest, int length)
{
	memcpy(dest, src, length * sizeof(int));
}

//after normalization, each p[i] = log(exp(p[i] / sum_j(exp(p[j])))
void norm_wo_underflow(double* p, int k, double* tempk)
{
	int i;
	double max_val = p[0];
	for(i = 0; i < k; i++)
		max_val = p[i] > max_val? p[i] : max_val;
	vec_scalar_sum(p, -max_val, k);
	Veccopy(p, tempk, k);
	exp_on_vec(tempk, k);
	double temp_sum = sum_vec(tempk, k);
	vec_scalar_sum(p, -log(temp_sum), k);
}

void pack_mat(double** X, double* v, int m, int n)
{
	int i;
	for(i = 0; i < m; i++)
		memcpy(v + i * n, X[i], n * sizeof(double));
}
void unpack_mat(double* v, double** X, int m, int n)
{
	int i;
	for(i = 0; i < m; i++)
		memcpy(X[i], v + i * n, n * sizeof(double));
}

int check_ending_info(double* v, int m, int n)
{
	return ((int) v[m * n]);
}

int is_checkpoint(int num_done, int num_total, int num_cps)
{
	assert(num_total >= num_cps);
	if(num_done == num_total)
		return 1;
	else
	{
		int cp_interval = (int)((double)num_total / (double) num_cps);
		if(num_done % cp_interval != 0)
			return 0;
		else
		{
			if(num_total - num_done < cp_interval)
				return 0;
			else
				return 1;
		} 
	}
}

int find_total_num(int k, int size, int rank)
{
	int for_return = k / size;
	int temp = k % size;
	if(rank < temp)
		return for_return + 1;
	else
		return for_return;
}

void sort_idx(double* p, int length, int* idx_array, int descend, int use_org_idx)
{
	int i;
	int j;
	double key;
	int key_idx;
	if(!use_org_idx)
	{
		for(i = 0; i < length; i++)
			idx_array[i] = i;
	}
	double val_array[length];  
	if(length == 1)
		return;
	Veccopy(p, val_array, length);
	if(!descend)
		scale_vec(val_array, length, -1.0);
	for(j = 1; j < length; j++)
	{
		key = val_array[j];
		key_idx = idx_array[j];
		i = j - 1;
		while(i >= 0 && val_array[i] < key)
		{
			val_array[i + 1] = val_array[i];
			idx_array[i + 1] = idx_array[i];
			i--;
		}
		idx_array[i + 1] = key_idx;
		val_array[i + 1] = key;
	}
}

void sort_with_idx(double* val_array, int length, int* idx_array, int descend, int use_org_idx)
{
	int i;
	int j;
	double key;
	int key_idx;
	if(!use_org_idx)
	{
		for(i = 0; i < length; i++)
			idx_array[i] = i;
	}
	if(length == 1)
		return;
	if(!descend)
		scale_vec(val_array, length, -1.0);
	for(j = 1; j < length; j++)
	{
		key = val_array[j];
		key_idx = idx_array[j];
		i = j - 1;
		while(i >= 0 && val_array[i] < key)
		{
			val_array[i + 1] = val_array[i];
			idx_array[i + 1] = idx_array[i];
			i--;
		}
		idx_array[i + 1] = key_idx;
		val_array[i + 1] = key;
	}
	if(!descend)
		scale_vec(val_array, length, -1.0);
}

void sort_int_in_place(int* list, int length, int descend)
{
	int i;
	int j;
	int key;
	if(length == 1)
		return;
	if(!descend)
		for(i = 0; i < length; i++)
			list[i] *= -1;
	for(j = 1; j < length; j++)
	{
		key = list[j];
		i = j - 1;
		while(i >= 0 && list[i] < key)
		{
			list[i + 1] = list[i];
			i--;
		}
		list[i + 1] = key;
	}
	if(!descend)
		for(i = 0; i < length; i++)
			list[i] *= -1;
}

int find_nearest_hub(double* x, double** H, int d, int c)
{
	int i;
	int nearest_idx = -1;
	double nearest_dist = DBL_MAX;
	double temp_dist;
	for(i = 0; i < c; i++)
	{
		temp_dist = vec_sq_dist(x, H[i], d);
		if(temp_dist < nearest_dist)
		{
			nearest_dist = temp_dist;
			nearest_idx = i;
		}
	}
	return nearest_idx;
}

void assign_clusters(double** X, double** H, int k, int d, int c, int* member_vec)
{
	int i;
	for(i = 0; i < k; i++)
		member_vec[i] = find_nearest_hub(X[i], H, d, c);
}

void find_hub(double**X, double* h, int k, int d, double* transition_counts, double eta, double eps, int verbose)
{
	int i;
	int j;
	int t;
	double** delta = zerosarray(k, d);
	double* p = (double*)calloc(k, sizeof(double));
	double** dv = zerosarray(k, d);
	double* tempk = (double*)calloc(k, sizeof(double));
	double** tempkd = zerosarray(k, d);
	double* dufr = (double*)calloc(d, sizeof(double));
	double* dh = (double*)malloc(d * sizeof(double));
	double* last_h = (double*)malloc(d * sizeof(double));
	double transition_sum = sum_vec(transition_counts, k);
	if(transition_sum == 0.0)
	{
		printf("No transition from the hub.\n");
		return;
		//exit(1);
	}
	double iter_diff;
	//Initialize with the weighted mean of all samples
	memset(h, 0, d * sizeof(double));
	for(i = 0; i < k; i++)
		add_vec(h, X[i], d, transition_counts[i]);
	scale_vec(h, d, 1.0 / transition_sum);

	/*
	t = 0;
	while(++t)
	{
		if(verbose)
			printf("Iteration %d for finding the hub...\n", t);

		memcpy(last_h, h, d * sizeof(double));
		for(j = 0; j < k; j++)
			for(i = 0; i < d; i++)
				delta[j][i] = h[i] - X[j][i];

		mat_mult(delta, delta, tempkd, k, d);
		scale_mat(tempkd, k, d, -1.0);
		sum_along_direct(tempkd, p, k, d, 1);
		norm_wo_underflow(p, k, tempk);
		for(j = 0; j < k; j++)
			for(i = 0; i < d; i++)
				dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
		sum_along_direct(dv, dufr, k, d, 0);
		memset(dh, 0, d * sizeof(double));
		for(i = 0; i < k; i++)
			add_vec(dh, delta[i], d, -2.0 * transition_counts[i]);
		add_vec(dh, dufr, d, -transition_sum);
		add_vec(h, dh, d, eta / (double)k);
		iter_diff = vec_diff_norm2(h, last_h, d);
		if(verbose)
			printf("The norm difference with last iteration is %f.\n", iter_diff);
		if( iter_diff< eps)
			break;
	}
	*/
	Array2Dfree(delta, k, d);
	Array2Dfree(dv, k, d);
	Array2Dfree(tempkd, k, d);
	free(tempk);
	free(p);
	free(dufr);
	free(dh);
	free(last_h);
}

double vec_diff_norm2(const double* x, const double* y, int length)
{
	int i;
	double temp = 0.0;
	for(i = 0; i < length; i++)
		temp += pow(x[i] - y[i], 2);
	return sqrt(temp);
}

int hamming_dist(const int* x, const int* y, int length)
{
	int i;
	int dist = 0;
	for(i = 0; i < length; i++)
		if(x[i] != y[i])
			dist++;
	return dist;
}

int hamming_dist_mat(int** X, int** Y, int m, int n)
{
	int dist = 0;
	int i;
	for(i = 0; i < m; i++)
		dist += hamming_dist(X[i], Y[i], n); 
	return dist;
}
void print_vec(double* vec, int len)
{
	int i;
	for(i = 0; i < len; i++)
	{
		printf("%f", vec[i]);
		if(i == len - 1)
			putchar('\n');
		else
			putchar(' ');
	}
}

void print_int_vec(int* vec, int len)
{
	int i;
	for(i = 0; i < len; i++)
	{
		printf("%d", vec[i]);
		if(i == len - 1)
			putchar('\n');
		else
			putchar(' ');
	}
}

void print_mat(double** X, int m, int n)
{
	int i;
	for(i = 0; i < m; i++)
		print_vec(X[i], n);
}

void group_means(double** X, int* member_vec, double** hubs, int k, int d, int num_clusters)
{
	int i;
	int j;
	int sample_count_array[num_clusters];
	double** backup_hubs = zerosarray(num_clusters, d);
	Array2Dcopy(hubs, backup_hubs, num_clusters, d);
	memset(sample_count_array, 0, num_clusters * sizeof(int));
	for(i = 0; i < num_clusters; i++)
		memset(hubs[i], 0, d * sizeof(double));
	for(i = 0; i < k; i++)
	{
		add_vec(hubs[member_vec[i]], X[i], d, 1.0);
		sample_count_array[member_vec[i]]++;
	}
	for(i = 0; i < num_clusters; i++)
		if(sample_count_array[i] > 0)
			scale_vec(hubs[i], d, 1.0 / (double) sample_count_array[i]);
		else
			memcpy(hubs[i], backup_hubs[i], d * sizeof(double));

	Array2Dfree(backup_hubs, num_clusters, d);
}

double compute_percentage(int frac, int total)
{
	return 100.0 * (double) frac / (double) total;
}

int extreme_id(double* list, int length, int do_max)
{
	int i;
	int current_candidate = -1;
	double current_value = do_max? -DBL_MAX : DBL_MAX;
	for(i = 0; i < length; i++)
	{
		if((do_max && list[i] > current_value) || (!do_max && list[i] < current_value))
		{
			current_candidate = i;
			current_value = list[i];
		}
	}
	return current_candidate;
}

void extreme_indices(double* list, int length, int* output_idx_list, int output_length, int do_max)
{
	int i;
	int j;
	int t;
	assert(output_length <= length);
	double top_value_list[output_length];
	for(i = 0; i < output_length; i++)
	{
		output_idx_list[i] = -1;
		if(do_max)
			top_value_list[i] = -DBL_MAX;
		else
			top_value_list[i] = DBL_MAX;
	}
	double value;
	for(i = 0; i < length; i++)
	{
		value = list[i];
		//need to replace a value
		if((do_max && value > top_value_list[output_length - 1]) || (!do_max && value < top_value_list[output_length - 1]))
		{
			j = output_length - 1;
			while(j > 0) 
			{
				if((do_max && top_value_list[j - 1] > value) || (!do_max && top_value_list[j - 1] < value))
					break;
				j--;
			}
			for(t = output_length - 1; t > j ; t--)
			{
				top_value_list[t] = top_value_list[t - 1];
				output_idx_list[t] = output_idx_list[t - 1];
			}
			top_value_list[j] = value;
			output_idx_list[j] = i;
		}
	}
}

int find_idx_with_self_omit(int cluster_idx, int current_idx, int num_clusters, int is_exit_point)
{
	assert(cluster_idx != current_idx);
	int idx = cluster_idx < current_idx? cluster_idx : cluster_idx - 1;
	return idx + (num_clusters - 1) * is_exit_point; 
}

int find_extreme_idx(double* list, int length, int find_max)
{
	int i;
	int candidate_idx = -1;
	double candidate_val = find_max? -DBL_MAX : DBL_MAX;
	for(i = 0; i < length; i++)
	{
		if((find_max && list[i] > candidate_val) || (!find_max && list[i] < candidate_val))
		{
			candidate_idx = i;
			candidate_val = list[i];
		}

	}
	return candidate_idx;
}

void random_permute(int* list, int length)
{
	int i;
	int j;
	int temp;
	//srand(time(NULL));
	//for(i = 0; i < length; i++)
		//list[i] = i;
	for(i = length - 1; i > 0; i--)
	{
		j = rand() % (i + 1);
		temp = list[j];
		list[j] = list[i];
		list[i] = temp;
	}
}

//sample m from n objects
void weighted_random_sampling(int n, int m, int* w, int* output)
{
	//srand(time(NULL));
	int i;
	int j;
	int W = 0;
	for(i = 0; i < n; i++)
		W += w[i];
	int* used = (int*)calloc(n, sizeof(int));
	int* interval_starts = (int*)calloc(n + 1, sizeof(int));
	double r;
	int lower;
	int upper;
	int mid;
	for(i = 1; i < n + 1; i++)
		interval_starts[i] = w[i - 1] + interval_starts[i - 1];
	for(i = 0; i < m; i++)
	{
		while(1)
		{
			r = (double) W * (double)rand() / (double)RAND_MAX;

			//for(lower = 0; lower < n; lower++)
			//	if(interval_starts[lower] <= r && interval_starts[lower + 1] > r)
			//		break;

			lower = 0;
			upper = n;
			while(upper - lower > 1)
			{
				//printf("(%d, %d)\n", lower, upper);
				mid = (lower + upper) / 2;
				if(interval_starts[mid] < r)
					lower = mid;
				else
					upper = mid;
			}

			if(!used[lower])
			{
				used[lower] = 1;
				//printf("[%d]\n", w[lower]);
				break;
			}
		}
		output[i] = lower;
	}
	free(used);
	free(interval_starts);
}

void count_song_frequency(PDATA pd, int* freq_array)
{
	int i;
	int j;
	memset(freq_array, 0, pd.num_songs * sizeof(int));
	for(i = 0; i < pd.num_playlists; i++)
		for(j = 0; j < pd.playlists_length[i]; j++)
			freq_array[pd.playlists[i][j]]++;
}

int sum_int_vec(int* vec, int length)
{
    int i;
    int sum = 0;
    for(i = 0; i < length; i++)
        sum += vec[i];
    return sum;
}

int extreme_int_vec(int* vec, int length, int do_max)
{
    int i;
    int extreme_val = do_max? -INT_MAX : INT_MAX;
    for(i = 0; i < length; i++)
        if((do_max && vec[i] > extreme_val) || (!do_max && vec[i] < extreme_val))
            extreme_val = vec[i];
    return extreme_val;
}

double extreme_double_vec(double* vec, int length, int do_max)
{
    int i;
    double extreme_val = do_max? -DBL_MAX : DBL_MAX;
    for(i = 0; i < length; i++)
        if((do_max && vec[i] > extreme_val) || (!do_max && vec[i] < extreme_val))
            extreme_val = vec[i];
    return extreme_val;
}

double mean(double* list, int length)
{
	double sum =  0.0;
	int i;
	for(i = 0; i < length; i++)
		sum += list[i];
	return sum / (double)length;
}

double std(double* list, int length)
{
	double sum = 0.0;
	int i;
	double temp = mean(list, length);
	for(i = 0; i < length; i++)
		sum += pow(list[i] - temp, 2.0);
	return sqrt(sum / (double) length);
}
