#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "LogisticEmbed_common.h"
#include "PairHashTable.h"
#include "LogisticPred.h"
#include "TransitionTable.h"

char testfile[200];
char embeddingfile[200];
char trainfile[200];
char songdistrfile[200];
char transdistrfile[200];


#ifndef FOROTHER
int main(int argc, char* argv[])
{
    TEST_PARAS myparas = parse_test_paras(argc, argv, testfile, embeddingfile, trainfile);
    printf("Predicting...\n");
    if(!myparas.allow_self_transition)
	printf("Do not allow self-transtion.\n");

    if (!myparas.underflow_correction)
        printf("Underflow correction disabled\n");

    int new_test_song_exp = (myparas.train_test_hash_file[0] != '\0');

    if(myparas.tagfile[0] == '\0' && new_test_song_exp)
    {
	printf("Have to support with a tag file if you want to test on unseen songs.\n");
	exit(1);
    }


    int d;
    int m; 
    int l;
    int i; 
    int j;
    int s;
    int t;
    int fr;
    int to;
    int prev;
    double* bias_terms = 0;
    double** X = read_matrix_file(embeddingfile, &l, &d, &bias_terms);
    double** realX;
    PDATA pd = read_playlists_data(testfile, 1);
    //int k = pd.num_songs;
    int k;
    double llhood = 0.0;
    double uniform_llhood = 0.0;
    double realn = 0.0;
    double not_realn= 0.0;
    int* train_test_hash;
    int k_train;
    int k_test;


    TDATA td;

    if(!new_test_song_exp)
    {
	k = pd.num_songs;
	if(myparas.tagfile[0] != '\0')
	{
	    td = read_tag_data(myparas.tagfile);
	    m = td.num_tags;
	    myparas.num_points = l / (k + m); 
	    realX = zerosarray(k * myparas.num_points, d);
	    calculate_realX(X, realX, td, k, m, d, myparas.num_points);
	    free_tag_data(td);

	    if(myparas.tag_ebd_filename[0] != '\0')
		write_embedding_to_file(X + k * myparas.num_points, m * myparas.num_points, d, myparas.tag_ebd_filename, 0);
	}
	else
	{
	    myparas.num_points = l / k;
	    realX = zerosarray(k * myparas.num_points, d);
	    Array2Dcopy(X, realX, l, d);
	}
	Array2Dfree(X, l, d);
    }
    else
    {
	printf("Prediction on unseen songs.\n");
	td = read_tag_data(myparas.tagfile);
	m = td.num_tags;
	k = td.num_songs;
	train_test_hash = read_hash(myparas.train_test_hash_file, &k_train);
	k_test = k - k_train;
	printf("Number of new songs %d.\n", k_test);
	myparas.num_points = l / (k_train + m); 
	realX = zerosarray(k * myparas.num_points, d);
	calculate_realX_with_hash(X, realX, td, k, m, d, myparas.num_points, k_train, train_test_hash);
	free_tag_data(td);
	Array2Dfree(X, l, d);
    }

    if(myparas.song_ebd_filename[0] != '\0')
	write_embedding_to_file(realX, k * myparas.num_points, d, myparas.song_ebd_filename, 0);
    if(myparas.bias_ebd_filename[0] != '\0')
    {
	FILE* fp = fopen(myparas.bias_ebd_filename, "w");
    
	for( i = 0; i < k ;i++)
	{
	    fprintf(fp, "%f", bias_terms[i]);
	    if ( i != k - 1)
		fputc('\n', fp);
	}

	fclose(fp);
    }

    double** square_dist;
    if(myparas.square_dist_filename[0] != '\0')
	square_dist = zerosarray(k, k);


    int n = 0;
    for(i = 0; i < pd.num_playlists; i ++)
	if(pd.playlists_length[i] > 0)
	    n += pd.playlists_length[i] - 1;
    printf("Altogether %d transitions.\n", n);fflush(stdout);

    PHASH* tcount;
    PHASH* tcount_train;
    double** tcount_full;
    double** tcount_full_train;

    if(myparas.use_hash_TTable)
	{
		//printf("(%d, %d)\n", 2 * n, k);
        tcount = create_empty_hash(2 * n, k);
	}
    else
        tcount_full = zerosarray(k, k);
    HELEM temp_elem;
    TPAIR temp_pair;
    int idx;
    double temp_val;
    for(i = 0; i < pd.num_playlists; i ++)
    {
	if(pd.playlists_length[i] > myparas.range)
	{
	    for(j = 0; j < pd.playlists_length[i] - 1; j++)
	    {
                temp_pair.fr = pd.playlists[i][j];
                temp_pair.to = pd.playlists[i][j + myparas.range];
                //printf("(%d, %d)\n", temp_pair.fr, temp_pair.to);
                if(temp_pair.fr >= 0 && temp_pair.to >= 0)
                {
                    if(myparas.use_hash_TTable)
                    {
                        idx = exist_in_hash(tcount, temp_pair);
                        if(idx < 0)
                        {
                            temp_elem.key = temp_pair;
                            temp_elem.val = 1.0;
                            add_entry(tcount, temp_elem);
                        }
                        else
                            update_with(tcount, idx, 1.0);
                    }
                    else
                        tcount_full[temp_pair.fr][temp_pair.to] += 1.0;
                }
	    }
	}
    }

    TRANSITIONTABLE ttable;
    TRANSITIONTABLE BFStable;


    //Need to use the training file
    if(myparas.output_distr)
    {
	PDATA pd_train = read_playlists_data(trainfile, 1);
        if(myparas.use_hash_TTable)
            tcount_train = create_empty_hash(2 * n, k);
        else
            tcount_full_train = zerosarray(k, k);
        for(i = 0; i < pd_train.num_playlists; i ++)
        {
            if(pd_train.playlists_length[i] > 1)
            {
                for(j = 0; j < pd_train.playlists_length[i] - 1; j++)
                {
                    temp_pair.fr = pd_train.playlists[i][j];
                    temp_pair.to = pd_train.playlists[i][j + 1];
                    if(myparas.use_hash_TTable)
                    {
                        idx = exist_in_hash(tcount_train, temp_pair);
                        if(idx < 0)
                        {
                            temp_elem.key = temp_pair;
                            temp_elem.val = 1.0;
                            add_entry(tcount_train, temp_elem);
                        }
                        else
                            update_with(tcount_train, idx, 1.0);
                    }
                    else
                        tcount_full_train[temp_pair.fr][temp_pair.to] += 1.0;
                }
            }
        }
    }

    FILE* song_distr_file;
    FILE* trans_distr_file;
    double* song_sep_ll;

    if(myparas.output_distr)
    {
        printf("Output likelihood distribution file turned on.\n");
        if(myparas.output_distr)
        {
            song_distr_file = fopen(songdistrfile, "w");
            trans_distr_file = fopen(transdistrfile, "w");
            song_sep_ll = (double*)calloc(k, sizeof(double));
        }

    }

    int* test_ids_for_new_songs;
    if(new_test_song_exp)
	test_ids_for_new_songs = get_test_ids(k, k_train, train_test_hash);


    if(!myparas.test_with_order_two)
    {
	for(fr = 0; fr < k; fr++)
	{
	    int collection_size;
	    int* collection_idx;
	    if(myparas.fast_collection)
	    {
		collection_size = (BFStable.parray)[fr].length;
		if (collection_size == 0)
		    continue;

		collection_idx = (int*)malloc(collection_size * sizeof(int));
		LINKEDELEM* tempp = (BFStable.parray)[fr].head;
		for(i = 0; i < collection_size; i++)
		{
		    collection_idx[i] = tempp -> idx; 
		    tempp = tempp -> pnext;
		}
	    }
	    else if(new_test_song_exp)
	    {
		collection_size = k_test;
		collection_idx = (int*)malloc(collection_size * sizeof(int));
		int_list_copy(test_ids_for_new_songs, collection_idx, k_test);
	    }
	    else
		collection_size = k;

	    double** delta = zerosarray(collection_size, d);
	    double* p = (double*)calloc(collection_size, sizeof(double));
	    double** tempkd = zerosarray(collection_size, d);
	    double* tempk = (double*)calloc(collection_size, sizeof(double));
	    double** mid_delta = 0;
	    double* mid_p = 0;
	    double** mid_tempkd = 0;

	    // I get a seg fault when these get freed. Don't understand.
	    if (myparas.num_points == 3) {
		mid_delta = zerosarray(collection_size, d);
		mid_p = (double*)calloc(collection_size, sizeof(double));
		mid_tempkd = zerosarray(collection_size, d);
	    }

	    for(j = 0; j < collection_size; j++)
	    {
		for(i = 0; i < d; i++)
		{
		    if(myparas.fast_collection || new_test_song_exp)
			delta[j][i] = realX[fr][i] - realX[(myparas.num_points - 1) * k + collection_idx[j]][i];
		    else
			delta[j][i] = realX[fr][i] - realX[(myparas.num_points - 1) * k + j][i];
		}
		if(myparas.num_points == 3) {
		    if(myparas.fast_collection || new_test_song_exp)
			mid_delta[j][i] =
			    realX[k + fr][i] - realX[k + collection_idx[j]][i];
		    else
			mid_delta[j][i] = realX[k + fr][i] - realX[k + j][i];
		}
	    }

	    mat_mult(delta, delta, tempkd, collection_size, d);
	    scale_mat(tempkd, collection_size, d, -1.0);
	    sum_along_direct(tempkd, p, collection_size, d, 1);

	    if(myparas.square_dist_filename[0] != '\0')
		for(i = 0; i < k; i++)
		    square_dist[fr][i] = -p[i];

	    if (bias_terms != 0)
		add_vec(p, bias_terms, collection_size, 1.0);

	    if (myparas.num_points == 3) {
		// Just use the mid_deltas (midpoint differences): square them,
		// then sum and add to the p vector directly, then the midpoint
		// probability is incorporated
		mat_mult(mid_delta, mid_delta, mid_tempkd, collection_size, d);
		scale_mat(mid_tempkd, collection_size, d, -1.0);
		sum_along_direct(mid_tempkd, mid_p, collection_size, d, 1);
		add_vec(p, mid_p, collection_size, 1.0); 
	    }

	    if (myparas.underflow_correction == 1) {
		double max_val = p[0];
		for(i = 0; i < collection_size; i++)
		    max_val = p[i] > max_val? p[i] : max_val;
		vec_scalar_sum(p, -max_val, collection_size);
	    }

	    Veccopy(p, tempk, collection_size);
	    exp_on_vec(tempk, collection_size);

	    //exp_on_vec(p, collection_size);

	    // underflow checking:

	    //    for (i = 0; i < collection_size; i++)
	    //        if (p[i] < 0.000001)
	    //            p[i] = 0.000001;

	    double temp_sum;
	    if(myparas.allow_self_transition)
		temp_sum = sum_vec(tempk, collection_size);
	    else
	    {
		temp_sum = 0.0;
		for(i = 0; i < collection_size; i++)
		    if(!myparas.fast_collection || new_test_song_exp)
			temp_sum += (i != fr)? tempk[i] : 0.0;
		    else
			temp_sum += (collection_idx[i] != fr)? tempk[i] : 0.0;
	    }
	    vec_scalar_sum(p, -log(temp_sum), collection_size);
	    //scale_vec(p, collection_size, 1.0 / temp_sum);

	    //printf("done...\n");
	    for(to = 0; to < k; to++)
	    {
		if(myparas.allow_self_transition || (!myparas.allow_self_transition && fr != to))
		{
		    temp_pair.fr = fr;
		    temp_pair.to = to;
		    //printf("(%d, %d)\n", fr, to);
		    if(myparas.use_hash_TTable)
			idx = exist_in_hash(tcount, temp_pair); 
		    else
			idx = tcount_full[fr][to] > 0.0? 1 : -1;
		    //printf("%d\n", idx);fflush(stdout);
		    int idx_train;
		    //printf("done...\n");fflush(stdout);

		    if(myparas.output_distr)
		    {
			if(myparas.use_hash_TTable)
			    idx_train = exist_in_hash(tcount_train, temp_pair);
			else
			    idx_train = tcount_full_train[fr][to] > 0.0? 1 : -1;
		    }



		    if(idx >= 0)
		    {
			if(myparas.fast_collection || new_test_song_exp)
			{
			    s = -1;
			    for(i = 0; i < collection_size; i++)
			    {
				if(collection_idx[i] == to)
				{
				    s = i;
				    break;
				}
			    }
			}
			else
			    s = to;

			//printf("%d\n", idx);fflush(stdout);
			if(myparas.use_hash_TTable)
			    temp_val = retrieve_value_with_idx(tcount, idx);
			else
			    temp_val = tcount_full[fr][to];

			if(s < 0)
			    not_realn += temp_val;
			else
			{
			    //printf("s = %d\n", s);
			    llhood += temp_val * p[s];
			    if(new_test_song_exp)
				uniform_llhood += temp_val * log(1.0 / (double) k_test);
			    realn += temp_val;

			    if(myparas.output_distr)
			    {
				//double temp_val_train =  idx_train >= 0? retrieve_value_with_idx(tcount_train, idx_train): 0.0;
				double temp_val_train;
				if(idx_train < 0)
				    temp_val_train = 0.0;
				else
				    temp_val_train = myparas.use_hash_TTable ? retrieve_value_with_idx(tcount_train, idx_train) : tcount_full_train[fr][to];

				song_sep_ll[fr] += temp_val * p[s];
				song_sep_ll[to] += temp_val * p[s];
				fprintf(trans_distr_file, "%d %d %f\n", (int)temp_val_train, (int)temp_val, temp_val * p[s]);
			    }
			}
		    }
		}
	    }




	    Array2Dfree(delta, collection_size, d);
	    free(p);
	    Array2Dfree(tempkd, collection_size, d);
	    free(tempk);
	    if (myparas.num_points == 3) {
		Array2Dfree(mid_delta, collection_size, d);
		free(mid_p);
		Array2Dfree(mid_tempkd, collection_size, d);
	    }
	    if(myparas.fast_collection || new_test_song_exp)
		free(collection_idx);
	}
    }
    //Test whether higher order transtion would help us
    else
    {
	//print_mat(realX, k, d);
	printf("Experimental test with smoothed order-2 predecessors.\n");
	double** delta = zerosarray(k, d);
	double* p = (double*)calloc(k, sizeof(double));
	double** tempkd = zerosarray(k, d);
	double* tempk = (double*)calloc(k, sizeof(double));
	double* smooth_predecessors = (double*)calloc(d, sizeof(double));
        for(i = 0; i < pd.num_playlists; i ++)
        {
            if(pd.playlists_length[i] > 2)
            {
                for(j = 0; j < pd.playlists_length[i] - 2; j++)
                {
                    prev = pd.playlists[i][j];
                    fr = pd.playlists[i][j + 1];
                    to = pd.playlists[i][j + 2];
		    //printf("(%d, %d, %d)\n", prev, fr, to);

		    //print_vec(realX[prev], d);
		    Veccopy(realX[prev], smooth_predecessors, d);
		    //print_vec(smooth_predecessors, d);
		    add_vec(smooth_predecessors, realX[fr], d, 1.0);
		    //printf("OK till here.\n");fflush(stdout);
		    scale_vec(smooth_predecessors, d, 0.5);



		    for(s = 0; s < k; s++)
			for(t = 0; t < d; t++)
			    delta[s][t] = smooth_predecessors[t]- realX[s][t];

		    mat_mult(delta, delta, tempkd, k, d);
		    scale_mat(tempkd, k, d, -1.0);
		    sum_along_direct(tempkd, p, k, d, 1);
		    norm_wo_underflow(p, k, tempk);
		    realn += 1.0;
		    llhood += p[to];
                }
            }
        }
	Array2Dfree(delta, k, d);
	free(p);
	Array2Dfree(tempkd, k, d);
	free(smooth_predecessors);
	free(tempk);
    }

    if(myparas.output_distr)
    {
	printf("Writing song distr.\n");
	for(i = 0; i < k; i++)
	    fprintf(song_distr_file, "%d %f\n", (int)(pd.id_counts[i]), song_sep_ll[i]);
	fclose(song_distr_file);
	fclose(trans_distr_file);
	free(song_sep_ll);
    }

    llhood /= realn;
    printf("Avg log-likelihood on test: %f\n", llhood);
    if(myparas.fast_collection)
	printf("Ratio of transitions that do not appear in the training set: %f\n", not_realn / (realn + not_realn));
    if(new_test_song_exp)
    {
	uniform_llhood /= realn;
	printf("Avg log-likelihood for uniform baseline: %f\n", uniform_llhood);
    }

    if(myparas.use_hash_TTable)
        free_hash(tcount);
    else
        Array2Dfree(tcount_full, k, k);
    free_playlists_data(pd);
    if(myparas.output_distr)
    {
        if(myparas.use_hash_TTable)
            free_hash(tcount_train);
        else
            Array2Dfree(tcount_full_train, k, k);
    }
    Array2Dfree(realX, k * myparas.num_points, d);

    if(new_test_song_exp)
    {
	free(train_test_hash);
	free(test_ids_for_new_songs);
    }

    if(myparas.square_dist_filename[0] != '\0')
    {
	write_embedding_to_file(square_dist, k, k, myparas.square_dist_filename, 0); 
	Array2Dfree(square_dist, k, k);
    }
}

#endif


TEST_PARAS parse_test_paras(int argc, char* argv[], char* testfile, char* embedfile, char* trainfile)
{
    //Default options
    TEST_PARAS myparas;
    myparas.method = 1;
    myparas.allow_self_transition = 1;
    myparas.fast_collection = 0;
    myparas.num_points = 2;
    myparas.output_distr = 0;
    myparas.underflow_correction = 1;
    myparas.tagfile[0] = '\0';
    myparas.train_test_hash_file[0] = '\0';
    myparas.range = 1;

    myparas.bias_ebd_filename[0] = '\0';
    myparas.square_dist_filename[0] = '\0';
    myparas.song_ebd_filename[0] = '\0';
    myparas.tag_ebd_filename[0] = '\0';
    myparas.use_hash_TTable = 1;
    myparas.test_with_order_two = 0;
	myparas.transition_range = 1;

    int i;
    for(i = 1; (i < argc) && argv[i][0] == '-'; i++)
    {
	switch(argv[i][1])
	{
	    case 't': i++; myparas.method = atoi(argv[i]); break;
	    case 'k': i++; myparas.use_hash_TTable = atoi(argv[i]); break;
	    case 'a': i++; myparas.allow_self_transition = atoi(argv[i]); break;
	    case 'f': i++; myparas.fast_collection = atoi(argv[i]); break;
	    case 's': i++; myparas.radius = atoi(argv[i]); break;
	    case 'u': i++; myparas.underflow_correction = atoi(argv[i]); break;
	    case 'x': i++; strcpy(myparas.tagfile, argv[i]); break;
	    case 'h': i++; strcpy(myparas.train_test_hash_file, argv[i]); break;
	    case 'n': i++; myparas.range = atoi(argv[i]); break;
		      
	    case 'b': i++; strcpy(myparas.bias_ebd_filename, argv[i]); break;
	    case 'd': i++; strcpy(myparas.square_dist_filename, argv[i]); break;
	    case 'o': i++; strcpy(myparas.song_ebd_filename, argv[i]); break;
	    case 'g': i++; strcpy(myparas.tag_ebd_filename, argv[i]); break;
	    case 'N': i++; myparas.transition_range = atoi(argv[i]); break;
	    case 'T': i++; myparas.test_with_order_two = atoi(argv[i]); break;
	    default: printf("Unrecognizable option -%c\n", argv[i][1]); exit(1);
	}

    }

    if((i + 1) < argc)
    {
	strcpy(testfile, argv[i]);
	strcpy(embedfile, argv[i + 1]);
	if((i + 2) < argc)
	    strcpy(trainfile, argv[i + 2]);
    }
    else
    {
	printf("Not enough parameters.\n");
	exit(1);
    }

    if((i + 4) < argc)
    {
	strcpy(songdistrfile, argv[i + 3]);
	strcpy(transdistrfile, argv[i + 4]);
        myparas.output_distr = 1;
    }

    return myparas;
}



