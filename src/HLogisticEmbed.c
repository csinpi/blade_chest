//Hierarchical Logistic Markov Embedding
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "LogisticEmbed_common.h"
#include "LogisticPred.h"
#include "HLogisticEmbed.h"
#include "PairHashTable.h"
#include "EmbedIO.h"
#include "IdxQueue.h"
#include "CNMap.h"
#define REBOOT_THRESHOLD -DBL_MAX

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

extern char trainfile[200];
extern char testfile[200];
extern char embeddingfile[200];
char clusterfile[200];
int global_num_clusters;
int global_num_portals;
int burnin_max_iter = 20;
double initial_percentage;

#ifdef TEST
int main(int argc, char* argv[])
{
    TEST_PARAS myparas = parse_test_paras(argc, argv, testfile, embeddingfile, trainfile);
	PDATA pd = read_playlists_data(testfile, 1);
	CEBD ce =  read_ClusterEmbedding_file(embeddingfile);
	int i;
	lme_by_cluster_test(pd, ce, myparas);
	free_playlists_data(pd);
	free_ClusterEmbedding(ce);
	return 0;
}

#else
#ifdef MPI
int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int i;
	int l;
	PARAS myparas = parse_paras(argc, argv, trainfile, embeddingfile);
	PDATA pd = read_playlists_data(trainfile, 0);
	CEBD ce = lme_by_cluster_MPI(pd, myparas, embeddingfile, &l, global_num_clusters);
	free_playlists_data(pd);
	free_ClusterEmbedding(ce);
	MPI_Finalize();
	return 0;
}

#else
int main(int argc, char* argv[])
{
	int i;
	int l;
	PARAS myparas = parse_paras(argc, argv, trainfile, embeddingfile);
	PDATA pd = read_playlists_data(trainfile, 1);
	CEBD ce;
	if(myparas.inter_cluster_transition_type != 3 && myparas.inter_cluster_transition_type != 4 && myparas.inter_cluster_transition_type != 5 && myparas.inter_cluster_transition_type != 6)
		ce = lme_by_cluster(pd, myparas, embeddingfile, &l, global_num_clusters);
	else
		ce = lme_by_cluster_uniportal(pd, myparas, embeddingfile, &l, global_num_clusters, global_num_portals);
	free_playlists_data(pd);
	free_ClusterEmbedding(ce);
	return 0;
}
#endif
#endif

double logistic_embed_component(PARAS myparas, double** X, double* bias, HPARAS hparas, int* seconds_used)
{
	time_t total_time_counter = time(NULL);
	if(hparas.tcount -> num_used == 0)
		return 0.0;
	int n; //Number of training transitions
	int i;
	int j;
	int s;
	//int q;
	int t;
	int fr;
	int to;
	int d = myparas.d;
	int k = hparas.num_subset_songs;
	//int kk = hparas.num_subset_songs + hparas.num_clusters;
	int kk = (hparas.tcount -> num_songs);
	//printf("kk is %d.\n", kk);
	double eta;
	int idx;
	if (myparas.ita > 0)
		eta = myparas.ita;
	else
		eta = k;
	n = 0;
	for(i = 0; i < ((hparas.tcount) -> length); i++)
		if(!is_null_entry(((hparas.tcount) -> p)[i].key))
			n += (int)((hparas.tcount) -> p)[i].val;
	/*
	   for(i = 0; i < pd.num_playlists; i ++)
	   if(pd.playlists_length[i] > 0)
	   n += pd.playlists_length[i] - 1;
	   */
	if(hparas.verbose)
		printf("Altogether %d transitions.\n", n);fflush(stdout);

	if(hparas.use_random_init)
	{
		randfill(X, kk, d, 1.0);
		if(myparas.bias_enabled)
			memset(bias, 0, kk * sizeof(double));
	}

	//print_mat(X, kk, d);

	double llhood = 0.0;
	double llhoodprev;
	double realn;

	double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
	int llhood_track_index = 0;

	for (i = 0; i < myparas.num_llhood_track; i++)
		last_n_llhoods[i] = 1.0;

	time_t start_time;
	clock_t start_clock, temp_clock;

	double** cdx = zerosarray(kk, d);
	double** delta = zerosarray(kk, d);
	double** dv = zerosarray(kk, d);
	double* p = (double*)calloc(kk, sizeof(double));
	double* tempk = (double*)calloc(kk, sizeof(double));
	double** tempkd = zerosarray(kk, d);
	double* dufr = (double*)calloc(d, sizeof(double));
	double temp_val;
	HELEM* p_valid_entry;

	double* bias_grad;
	if(myparas.bias_enabled)
	{
		bias_grad = (double*)calloc(kk, sizeof(double));
	}

	//printf("Before. Norm of the hubs: %f.\n", frob_norm(X, hparas.num_clusters, d));
	



	int order_list[kk];
	for(i = 0; i < kk; i++)
		order_list[i] = i;

	for(t = 0; t < myparas.max_iter; t++)
	{
		start_time = time(NULL);
		start_clock = clock();
		if(hparas.verbose)
			printf("Little Iteration %d\n", t + 1);fflush(stdout);
		llhoodprev = llhood;
		llhood = 0.0;
		realn = 0.0;
		//for(s = 0; s < hparas.num_subset_songs; s++)

		if(myparas.do_rperm)
			random_permute(order_list, kk);
		//for(fr = 0; fr < kk; fr++)
		for(s = 0; s < kk; s++)
		{
			//printf("%d\n", s);
			fr = order_list[s];
			//fr = hparas.idx_vec[s];
			for(i = 0; i < kk; i++)
				memset(cdx[i], 0, d * sizeof(double));
			if(myparas.bias_enabled)
				memset(bias_grad, 0, kk * sizeof(double));

			for(j = 0; j < kk; j++)
				for(i = 0; i < d; i++)
					delta[j][i] = X[fr][i] - X[j][i];

			mat_mult(delta, delta, tempkd, kk, d);
			scale_mat(tempkd, kk, d, -1.0);
			sum_along_direct(tempkd, p, kk, d, 1);
			if(myparas.bias_enabled)
                add_vec(p, bias, kk, 1.0);
			norm_wo_underflow(p, kk, tempk);
			for(j = 0; j < kk; j++)
				for(i = 0; i < d; i++)
					dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
			sum_along_direct(dv, dufr, kk, d, 0);

			double accu_temp_vals = 0.0;
			p_valid_entry = (hparas.tcount) -> p_first_tran[fr];
			while(p_valid_entry != NULL)
			{
				to = (p_valid_entry -> key.to);
				//q = find_in_list(to, hparas.idx_vec, hparas.num_subset_songs); 
				temp_val = p_valid_entry -> val;
				//printf("%f\n", temp_val);
				accu_temp_vals += temp_val;
                if (myparas.bias_enabled)
					bias_grad[to] += temp_val;
				add_vec(cdx[to], delta[to], d, 2.0 * temp_val);
				/*
				   Veccopy(dufr, tempd, d);
				   add_vec(tempd, delta[to], d, 2.0);
				   scale_vec(tempd, d, -1.0);
				   add_vec(cdx[fr], tempd, d, temp_val);
				   */
				add_vec(cdx[fr], delta[to], d, -2.0 * temp_val);
				llhood += temp_val * p[to];
				realn += temp_val;
				p_valid_entry = (p_valid_entry -> pnext);
			}
			add_mat(cdx, dv, kk, d, accu_temp_vals);
			add_vec(cdx[fr], dufr, d, -accu_temp_vals);
            if (myparas.bias_enabled)
                for (i = 0; i < kk; i++)
                    bias_grad[i] += -1.0 * accu_temp_vals * exp(p[i]);
			add_mat(X, cdx, kk, d, eta / (double)n);
			if(myparas.bias_enabled)
				add_vec(bias, bias_grad, kk, eta / (double)n);

			//write_embedding_to_file(X, kk, d, embeddingfile, 0);
		}


		llhood /= realn;


		if(hparas.verbose)
		{
			printf("The iteration took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
			printf("The iteration took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
			printf("Norms of coordinates: %f\n", frob_norm(X, kk, d));fflush(stdout);
			printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
		}
		if(t > 0)
		{
			if(llhoodprev > llhood)
			{
				eta /= myparas.beta;
				//eta = 0.0;
				if(hparas.verbose)
					printf("Reducing eta to: %f\n", eta);
			}
			else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
			{
				eta *= myparas.alpha;
				if(hparas.verbose)
					printf("Increasing eta to: %f\n", eta);
			}
		}
		last_n_llhoods[llhood_track_index] = llhood;
		double min_llhood = 1.0, max_llhood = -1000.0, total_llhood = 0.0;
		for (i = 0; i < myparas.num_llhood_track; i++) {
			total_llhood += last_n_llhoods[i];
			if (last_n_llhoods[i] < min_llhood)
				min_llhood = last_n_llhoods[i];
			if (last_n_llhoods[i] > max_llhood)
				max_llhood = last_n_llhoods[i];
		}
		if(hparas.verbose)
		{
			printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
			//printf("magnitude of gradient %f\n", frob_norm(cdx + hparas.num_clusters, k, d));
			printf("magnitude of gradient %f\n", frob_norm(cdx, kk, d));
		}

		/*
		   if(embedding_file != NULL)
		   write_embedding_to_file(X, k, d, embeddingfile, 0);
		   */

		if (llhood < REBOOT_THRESHOLD)
			break;
		if (myparas.num_llhood_track > 0
				&& fabs(min_llhood - max_llhood) < myparas.eps 
				&& total_llhood < myparas.num_llhood_track) {
			break;
		}
		else if (myparas.num_llhood_track <= 0
				&& llhood >= llhoodprev
				&& llhood - llhoodprev < myparas.eps 
				&& t > myparas.least_iter)
			break;


		// update llhood tracker index

		llhood_track_index++;
		llhood_track_index %= myparas.num_llhood_track;
	}



	//printf("Done. Norm of the hubs: %f.\n", frob_norm(X, hparas.num_clusters, d));
	Array2Dfree(cdx, kk , d);
	Array2Dfree(delta, kk , d);
	Array2Dfree(dv, kk , d);
	free(p);
	free(tempk);
	Array2Dfree(tempkd, kk, d);
	free(last_n_llhoods);
	free(dufr);
	if(myparas.bias_enabled)
		free(bias_grad);

	if(llhood < REBOOT_THRESHOLD)
	{
		PARAS newparas = myparas;
		newparas.ita /= 2.0;
		printf("Need to use smaller initial learning rate %f.\n", newparas.ita);
		return logistic_embed_component(newparas, X, bias, hparas, seconds_used);
	}
	else
	{
        *seconds_used = (int)time(NULL) - (int)total_time_counter; 
		printf("Avg log-likelihood on train: %f, with %d songs. %d seconds elapsed.\n", llhood, hparas.num_subset_songs, *seconds_used);
		//printf("Realn is %f.\n", realn);
		return llhood * realn;
	}
}

//pure stochastic algorithm
double logistic_embed_component_ps(PARAS myparas, double** X, double* bias, HPARAS hparas, int* seconds_used)
{
	time_t total_time_counter = time(NULL);
	if(hparas.tcount -> num_used == 0)
		return 0.0;


	int n; //Number of training transitions
	int i;
	int j;
	int s;
	//int q;
	int t;
	int fr;
	int to;
	int d = myparas.d;
	int k = hparas.num_subset_songs;
	//int kk = hparas.num_subset_songs + hparas.num_clusters;
	int kk = (hparas.tcount -> num_songs);
	//printf("kk is %d.\n", kk);
	double eta;
	int idx;
	if (myparas.ita > 0)
		eta = myparas.ita;
	else
		eta = k;
	n = 0;
	for(i = 0; i < ((hparas.tcount) -> length); i++)
		if(!is_null_entry(((hparas.tcount) -> p)[i].key))
			n += (int)((hparas.tcount) -> p)[i].val;
	/*
	   for(i = 0; i < pd.num_playlists; i ++)
	   if(pd.playlists_length[i] > 0)
	   n += pd.playlists_length[i] - 1;
	   */
	if(hparas.verbose)
		printf("Altogether %d transitions.\n", n);fflush(stdout);

	if(hparas.use_random_init)
		randfill(X, kk, d, 1.0);

	if(myparas.ps_interval <= 0)
		myparas.ps_interval = 10 * k;


	double llhood = 0.0;
	double llhoodprev;
	double realn;

	double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
	//int llhood_track_index = 0;

	for (i = 0; i < myparas.num_llhood_track; i++)
		last_n_llhoods[i] = 1.0;

	time_t start_time;
	clock_t start_clock, temp_clock;

	double** cdx = zerosarray(kk, d);
	double** delta = zerosarray(kk, d);
	double* p = (double*)calloc(kk, sizeof(double));
	double* tempk = (double*)calloc(kk, sizeof(double));
	double** tempkd = zerosarray(kk, d);
	double* dufr = (double*)calloc(d, sizeof(double));
	double temp_val;
	HELEM* p_valid_entry;

	double** max_X = zerosarray(kk, d);
	double* max_bias;
	if(myparas.bias_enabled)
		max_bias = (double*)calloc(kk, sizeof(double));
	double max_llhood = -DBL_MAX;

	double* bias_grad;
	if(myparas.bias_enabled)
	{
		bias_grad = (double*)calloc(kk, sizeof(double));
		memset(bias, 0, kk * sizeof(double));
	}

	//printf("Before. Norm of the hubs: %f.\n", frob_norm(X, hparas.num_clusters, d));
	



	int num_trans;
	TPAIR* tarray =  build_transition_array_for_ps(hparas.tcount, &num_trans);
	//printf("%d~~~~~~~~~~\n", num_trans);
	//printf("haha\n");
	//for(i = 0; i < pd.num_transitions; i++)
		//printf("(%d, %d)\n", tarray[i].fr, tarray[i].to);

	srand(time(NULL));
	//for(t = 0; t < myparas.max_iter; t++)
	for(t = 0; t < 10000000; t++)
	{
		if(t % myparas.ps_interval == 0)
		{
			start_time = time(NULL);
			start_clock = clock();
			llhoodprev = llhood;
			llhood = 0.0;
			realn = 0.0;
		}
		//for(s = 0; s < hparas.num_subset_songs; s++)

		int rand_idx = rand() % num_trans;
		fr = tarray[rand_idx].fr;
		to = tarray[rand_idx].to;

		//for(i = 0; i < kk; i++)
		//	memset(cdx[i], 0, d * sizeof(double));
		//if(myparas.bias_enabled)
		//	memset(bias_grad, 0, kk * sizeof(double));

		for(j = 0; j < kk; j++)
			for(i = 0; i < d; i++)
				delta[j][i] = X[fr][i] - X[j][i];

		mat_mult(delta, delta, tempkd, kk, d);
		scale_mat(tempkd, kk, d, -1.0);
		sum_along_direct(tempkd, p, kk, d, 1);
		if(myparas.bias_enabled)
			add_vec(p, bias, kk, 1.0);
		norm_wo_underflow(p, kk, tempk);
		llhood += p[to];
		realn += 1.0;

		//gradient step
		for(j = 0; j < kk; j++)
			for(i = 0; i < d; i++)
				cdx[j][i] = -2.0 * exp(p[j]) * delta[j][i];
		sum_along_direct(cdx, dufr, kk, d, 0);
		add_vec(cdx[to], delta[to], d, 2.0);
		add_vec(cdx[fr], delta[to], d, -2.0);
		add_vec(cdx[fr], dufr, d, -1.0);
		if(myparas.bias_enabled)
		{
			for (i = 0; i < kk; i++)
				bias_grad[i] = -1.0 * exp(p[i]);
			bias_grad[to] += 1.0;
		}

		add_mat(X, cdx, kk, d, eta / (double)n);
		if(myparas.bias_enabled)
			add_vec(bias, bias_grad, kk, eta / (double)n);


		if(t % myparas.ps_interval == (myparas.ps_interval - 1))
		{

			llhood /= realn;
			assert((int)realn == myparas.ps_interval);
			if(hparas.verbose)
			{
				printf("The stochastic interval took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
				printf("The stochastic interval took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
				printf("Norms of coordinates: %f\n", frob_norm(X, kk, d));fflush(stdout);
				printf("Sampled avg log-likelihood on train: %f\n", llhood);fflush(stdout);
			}
			if(t > myparas.ps_interval)
			{
				if(llhoodprev > llhood)
				{
					eta /= myparas.beta;
					//eta = 0.0;
					if(hparas.verbose)
						printf("Reducing eta to: %f\n", eta);
				}
				else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
				{
					eta *= myparas.alpha;
					if(hparas.verbose)
						printf("Increasing eta to: %f\n", eta);
				}
			}
			
			for(i = myparas.num_llhood_track - 1; i > 0; i--)
				last_n_llhoods[i] = last_n_llhoods[i - 1];
			last_n_llhoods[0] = llhood;

			if(llhood > max_llhood)
			{
				max_llhood = llhood;
				Array2Dcopy(X, max_X, kk, d);
				if(myparas.bias_enabled)
					Veccopy(bias, max_bias, kk);
			}

			double min_llhood = 1.0, max_llhood = -1000.0, total_llhood = 0.0;
			for (i = 0; i < myparas.num_llhood_track; i++) {
				total_llhood += last_n_llhoods[i];
				if (last_n_llhoods[i] < min_llhood)
					min_llhood = last_n_llhoods[i];
				if (last_n_llhoods[i] > max_llhood)
					max_llhood = last_n_llhoods[i];
			}

			int num_going_down = 0;
			for(i = 0; i < myparas.num_llhood_track - 1; i++)
			{
				if(last_n_llhoods[i] < last_n_llhoods[i + 1] && last_n_llhoods[i + 1] < 0)
					num_going_down++;
				else
					break;
				//double a = last_n_llhoods[(llhood_track_index - i - 1 ) % myparas.num_llhood_track];
				//double b = last_n_llhoods[(llhood_track_index - i) % myparas.num_llhood_track];
				//if(a < 0 && b < 0 && a > b)
				//	num_going_down++;
				//if(last_n_llhoods[(llhood_track_index - i - 1 ) % myparas.num_llhood_track] \
				> last_n_llhoods[(llhood_track_index - i) % myparas.num_llhood_track])
			}

			if(hparas.verbose)
			{
				printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
				printf("Ratio of degrade %d/%d.\n", num_going_down, myparas.num_llhood_track - 1);
				//printf("magnitude of gradient %f\n", frob_norm(cdx + hparas.num_clusters, k, d));
				printf("magnitude of gradient %f\n", frob_norm(cdx, kk, d));
			}

			/*
			   if(embedding_file != NULL)
			   write_embedding_to_file(X, k, d, embeddingfile, 0);
			   */

			if (llhood < REBOOT_THRESHOLD)
				break;
			if(eta < 1e-4 * myparas.ita)
				break;
			if (myparas.num_llhood_track > 0
					&& (fabs(min_llhood - max_llhood) < myparas.eps || num_going_down > 2) 
					&& total_llhood < myparas.num_llhood_track) {
				break;
			}
			else if (myparas.num_llhood_track <= 0
					&& llhood >= llhoodprev
					&& llhood - llhoodprev < myparas.eps 
					&& t > myparas.least_iter)
				break;


			// update llhood tracker index

			//llhood_track_index++;
			//llhood_track_index %= myparas.num_llhood_track;
		}
	}

	llhood = max_llhood;
	Array2Dcopy(max_X, X, kk, d);
	if(myparas.bias_enabled)
		Veccopy(max_bias, bias, kk);


	//printf("Done. Norm of the hubs: %f.\n", frob_norm(X, hparas.num_clusters, d));
	Array2Dfree(cdx, kk , d);
	Array2Dfree(delta, kk , d);
	free(p);
	free(tempk);
	Array2Dfree(tempkd, kk, d);
	free(last_n_llhoods);
	free(dufr);
	free(tarray);
	Array2Dfree(max_X, kk, d);
	if(myparas.bias_enabled)
		free(max_bias);
	if(myparas.bias_enabled)
		free(bias_grad);

	if(llhood < REBOOT_THRESHOLD)
	{
		PARAS newparas = myparas;
		newparas.ita /= 2.0;
		printf("Need to use smaller initial learning rate %f.\n", newparas.ita);
		return logistic_embed_component_ps(newparas, X, bias, hparas, seconds_used);
	}
	else
	{
        *seconds_used = (int)time(NULL) - (int)total_time_counter;
		printf("Avg log-likelihood on train: %f, with %d songs. %d seconds elapsed.\n", llhood, hparas.num_subset_songs, *seconds_used);
		//printf("Realn is %f.\n", realn);
		return llhood * num_trans;
	}
}


double logistic_test_component(double** X, double* bias, PHASH* tcount, int d)
{
	int bias_enabled = (bias != NULL);
	int i;
	int j;
	int k = tcount -> num_songs;
	int fr;
	int to;
	double temp_val;
	double llhood = 0.0;
	double** delta = zerosarray(k, d);
	double* p = (double*)calloc(k, sizeof(double));
	double* tempk = (double*)calloc(k, sizeof(double));
	double** tempkd = zerosarray(k, d);
	HELEM* p_valid_entry;
	//print_mat(X, k, d);
	//printf("done printing.\n");
	for(fr = 0; fr < k; fr++)
	{
		for(j = 0; j < k; j++)
			for(i = 0; i < d; i++)
				delta[j][i] = X[fr][i] - X[j][i];
		mat_mult(delta, delta, tempkd, k, d);
		scale_mat(tempkd, k, d, -1.0);
		sum_along_direct(tempkd, p, k, d, 1);
		if(bias_enabled)
			add_vec(p, bias, k, 1.0);
		norm_wo_underflow(p, k, tempk);

		p_valid_entry = tcount -> p_first_tran[fr];
		while(p_valid_entry != NULL)
		{
			to = (p_valid_entry -> key.to);
			//printf("(%d, %d)\n", fr, to);
			temp_val = p_valid_entry -> val;
			llhood += temp_val * p[to];
			p_valid_entry = (p_valid_entry -> pnext);
		}
	}
	free(p);
	free(tempk);
	Array2Dfree(tempkd, k, d);
	Array2Dfree(delta, k, d);
	return llhood;
}



PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile)
{
	//Default options
	PARAS myparas;
	myparas.do_normalization = 0;
	myparas.method = 2;
	myparas.d = 2;
	//myparas.ita = 300;
	myparas.ita = 10;
	myparas.eps = 0.00001;
	myparas.lambda = 1.0;
	myparas.nu_multiplier = 1.0;
	myparas.random_init = 1;
	myparas.fast_collection = 0;
	myparas.alpha = 1.1;
	myparas.beta = 2.0;
	myparas.least_iter = 250;
	myparas.radius = 2;
	myparas.regularization_type = 0; // default no regularization
	myparas.stoc_grad = 1;
	myparas.num_points = 1;
	myparas.allow_self_transition = 1;
	myparas.output_distr = 0;
	myparas.grid_heuristic = 0;
	myparas.regeneration_interval = 10;
	myparas.bias_enabled = 1;
	myparas.num_llhood_track = 10;
	myparas.tagfile[0] = '\0';
	myparas.tag_regularizer = 0;
	myparas.hessian = 0;
	myparas.landmark_heuristic = 0;
	myparas.num_landmark = 0;
	myparas.lowerbound_ratio = 0.3;
	myparas.reboot_enabled = 0;
	myparas.landmark_burnin_iter = 100;
	myparas.use_hash_TTable = 1;
	myparas.triple_dependency = 0;
	myparas.angle_lambda = 1.0;
	myparas.transition_range = 1;
	myparas.num_threads = 0;
	myparas.mpi_num_checkpoints = 50;
	myparas.inter_cluster_transition_type = 1;
	global_num_clusters = 1;
	global_num_portals = 5;
	initial_percentage = 0.08;
	clusterfile[0] = '\0';
	myparas.verbose = 0;
	myparas.max_iter = 500;
	myparas.do_rperm = 0;
	myparas.do_ps = 0;
	myparas.ps_interval = 10000;


	// don't use r, e, or n for anything which you plan to actually change.
	// (reserved in sge_unit.sh script)

	int i;
	for(i = 1; (i < argc) && argv[i][0] == '-'; i++)
	{
		switch(argv[i][1])
		{
			case 'n': i++; myparas.do_normalization = atoi(argv[i]); break;
			case 't': i++; myparas.method = atoi(argv[i]); break;
			case 'r': i++; myparas.random_init = atoi(argv[i]); break;
			case 'd': i++; myparas.d = atoi(argv[i]); break;
			case 'i': i++; myparas.ita = atof(argv[i]); break;
			case 'e': i++; myparas.eps = atof(argv[i]); break;
			case 'l': i++; myparas.lambda = atof(argv[i]); break;
			case 'f': i++; myparas.fast_collection= atoi(argv[i]); break;
			case 's': i++; myparas.radius= atoi(argv[i]); break;
			case 'a': i++; myparas.alpha = atof(argv[i]); break;
			case 'b': i++; myparas.beta = atof(argv[i]); break;
			case 'g':
					  i++;
					  if (argv[i][1] == '\0') {
						  myparas.regularization_type = atoi(argv[i]);
						  myparas.tag_regularizer = atoi(argv[i]);
						  printf("Both regularizers set to %d\n", myparas.regularization_type);
					  }
					  else {
						  char first_reg[2] = "\0\0";
						  char second_reg[2] = "\0\0";
						  first_reg[0] = argv[i][0];
						  second_reg[0] = argv[i][1];

						  myparas.regularization_type = atoi(first_reg);
						  myparas.tag_regularizer = atoi(second_reg);
						  printf("Song regularizer set to %d\n", myparas.regularization_type);
						  printf("Tag regularizer set to %d\n", myparas.tag_regularizer);
					  }                
					  break;
			case 'h': i++; myparas.grid_heuristic = atoi(argv[i]); break;
			case 'm': i++; myparas.landmark_heuristic = atoi(argv[i]); break;
			case 'p': i++; myparas.bias_enabled = atoi(argv[i]); break;
			case 'w': i++; myparas.num_llhood_track = atoi(argv[i]); break;
			case 'c': i++; myparas.hessian = atoi(argv[i]); break;
			case 'x': i++; strcpy(myparas.tagfile, argv[i]); break;
			case 'q': i++; myparas.num_landmark = atoi(argv[i]); break;
			case 'y': i++; myparas.lowerbound_ratio = atof(argv[i]); break;
			case 'o': i++; myparas.reboot_enabled = atoi(argv[i]); break;
			case '0': i++; myparas.landmark_burnin_iter = atoi(argv[i]); break;
			case 'u': i++; myparas.nu_multiplier = atof(argv[i]); break;
			case 'k': i++; myparas.use_hash_TTable = atoi(argv[i]); break;
			case 'T': i++; myparas.triple_dependency = atoi(argv[i]); break;
			case 'L': i++; myparas.angle_lambda = atof(argv[i]); break;
			case 'N': i++; myparas.transition_range = atoi(argv[i]); break;
			case 'D': i++; myparas.num_threads = atoi(argv[i]); break;
			case 'K': i++; global_num_clusters = atoi(argv[i]); break;
			case 'O': i++; global_num_portals = atoi(argv[i]); break;
			case 'C': i++; strcpy(clusterfile, argv[i]); break;
            case 'M': i++; myparas.mpi_num_checkpoints = atoi(argv[i]); break;
            case 'P': i++; myparas.inter_cluster_transition_type = atoi(argv[i]); break;
            case 'V': i++; myparas.verbose = atoi(argv[i]); break;
            case 'I': i++; myparas.max_iter = atoi(argv[i]); break;
            case 'R': i++; myparas.do_rperm = atoi(argv[i]); break;
            case 'S': i++; myparas.do_ps = atoi(argv[i]); break;
            case 'W': i++; myparas.ps_interval = atoi(argv[i]); break;
			case 'G': i++; initial_percentage = atof(argv[i]); break;
			default: printf("Unrecognizable option -%c\n", argv[i][1]); exit(1);

		}

	}

	if((i + 1) < argc)
	{
		strcpy(trainfile, argv[i]);
		strcpy(embedfile, argv[i + 1]);
	}
	else
	{
		printf("Not enough parameters.\n");
		exit(1);
	}
	return myparas;
}


//int find_cluster_song_list(int* member_vec, int k, int cluster_idx, int** cluster_idx_list)
//{
//	int i;
//	int cluster_song_count = 0;
//	for(i = 0; i < k; i++)
//		if(member_vec[i] == cluster_idx)
//			cluster_song_count++;
//	(*cluster_idx_list) = (int*)malloc(cluster_song_count * sizeof(int)) ;
//	int t = 0;
//	for(i = 0; i < k; i++)
//		if(member_vec[i] == cluster_idx)
//			(*cluster_idx_list)[t++] = i;
//
//	//sort it
//	//print_int_vec((*cluster_idx_list), cluster_song_count);
//	sort_int_in_place((*cluster_idx_list), cluster_song_count, 0);
//	//print_int_vec((*cluster_idx_list), cluster_song_count);
//	return cluster_song_count;
//}

void free_ClusterEmbedding(CEBD ce)
{
	int i;
	for(i = 0; i < ce.num_clusters; i++)
	{
		if(ce.inter_cluster_transition_type == 0 || ce.inter_cluster_transition_type == 2)
			Array2Dfree(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_clusters, ce.d);
		else if(ce.inter_cluster_transition_type == 1)
			Array2Dfree(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + 2 *(ce.num_clusters - 1), ce.d);
		else if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 6)
			Array2Dfree(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_portals, ce.d);
		else if(ce.inter_cluster_transition_type == 5)
			Array2Dfree(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + 2 * ce.num_portals, ce.d);

		free(ce.cluster_song_relation[i]);
		if(ce.bias_by_cluster != NULL)
		   free(ce.bias_by_cluster[i]);	
	}
	free(ce.num_songs_in_each_cluster);
	free(ce.cluster_song_relation);
	free(ce.X_by_cluster);
	free(ce.assignment);
	if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5 || ce.inter_cluster_transition_type == 6)
	{
		//for(i = 0; i < ce.num_clusters; i++)
		//	free(ce.preferred_portal_array[i]);
		//free(ce.preferred_portal_array);
		Array2Dfree_int(ce.preferred_portal_array, ce.num_clusters, ce.num_songs); 
	}
	if(ce.bias_by_cluster != NULL)
		free(ce.bias_by_cluster);
}

CEBD lme_by_cluster(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters)
{
	time_t start_time = time(NULL);
	CEBD ce;
	ce.num_songs = pd.num_songs;
	ce.d = myparas.d;
	ce.inter_cluster_transition_type = myparas.inter_cluster_transition_type;
	int k = pd.num_songs;
	int i;
	int j;
	int d = myparas.d;
	*l = k;
	char error_msg[200];
	double precomputation_time = 0.0;
    int precluster_time = 0;
    time_t precluster_start;
	char temp_str[1000];
	//temp_str[0] = '\0';
	if(clusterfile[0] != '\0')
	{
		printf("Using existing clustering file: %s\n", clusterfile);
		//fp = fopen(clusterfile, "r");
		//while(!feof(fp))
		//{
		//	fgets(temp_str, 100000, fp);
		//	printf("%s", temp_str);
		//}
		//	printf("=============\n");
		//fclose(fp);
		FILE* fp = fopen(clusterfile, "r");
		int temp_length;
		read_int_vec(fp, &(ce.assignment), &temp_length);
		//print_int_vec(ce.assignment, temp_length);
		if(fgets(temp_str, 1000, fp) != NULL)
		{
			precomputation_time = extract_ending_double(temp_str, "total: ");
			printf("Precomputation time is %f seconds.\n", precomputation_time);
		}

		fclose(fp);
		if(temp_length != k)
		{
			sprintf(error_msg, "The number of songs in clustering file does not match that in the playlist file (%d vs %d).\n", temp_length, k); 
			error_in_read(error_msg);
		}
		num_clusters = 0;
		for(i = 0; i < k; i++)
		{
			assert(ce.assignment[i] >= 0);
			num_clusters = max(ce.assignment[i], num_clusters);
		}
		num_clusters++;
		ce.num_clusters = num_clusters;
		printf("The number of clusters is decided by the file, which is %d.\n", num_clusters);

		//pre computation time is available
	}
	else
	{
        precluster_start = time(NULL);
		ce.num_clusters = num_clusters;
		ce.assignment = (int*)calloc(k, sizeof(int));
		//preclustering_by_lme(pd, myparas, num_clusters, num_clusters, ce.assignment);
		//preclustering_by_lme(pd, myparas, 10, num_clusters, ce.assignment);
		//greedy_assign_cluster_with_constraints(ce.assignment, k, num_clusters, pd, 4.0 * (double)k / num_clusters, 0);
		//random_assign_cluster(ce.assignment, k, num_clusters);
		if(num_clusters > 1)
			preclustering_with_aggregated_song(pd, myparas, num_clusters, initial_percentage, ce.assignment);
        precluster_time = (int)time(NULL) - (int)precluster_start;
	}
    
    printf("Done, preclustering took %d seconds\n", precluster_time);

	ce.num_songs_in_each_cluster = (int*)malloc(num_clusters * sizeof(int));
	ce.cluster_song_relation = (int**)malloc(num_clusters * sizeof(int*));
	ce.X_by_cluster = (double***)malloc(num_clusters * sizeof(double**));
	if(myparas.bias_enabled)
		ce.bias_by_cluster = (double**)malloc(num_clusters * sizeof(double*));
	else
		ce.bias_by_cluster = NULL;

	int* last_assignment = (int*)malloc(k * sizeof(int));

	//random_assign_cluster(current_assignment, k, num_clusters);

	HPARAS lower_hparas_array[num_clusters];
	for(i = 0; i < num_clusters; i++)
	{
		lower_hparas_array[i].use_random_init = 1;
		lower_hparas_array[i].num_clusters = num_clusters;
		lower_hparas_array[i].verbose = myparas.verbose;
	}

	int t = 0;
	double llhood_by_cluster[num_clusters];
	double** transitions_to_each_cluster = zerosarray(k, num_clusters);
	//double** p_cluster_square = zerosarray(num_clusters, num_clusters);
	//double* cluster_rating_vec = (double*)malloc(num_clusters * sizeof(double));

    int component_time_array[num_clusters];
	while(++t)
	{
		printf("Big iteration %d...\n", t);
		memcpy(last_assignment, ce.assignment, k * sizeof(int));
		int temp_sum = 0;
		for(i = 0; i < num_clusters; i++)
		{
			ce.num_songs_in_each_cluster[i] =  find_cluster_song_list(ce.assignment, k, i, ce.cluster_song_relation + i);
			lower_hparas_array[i].num_subset_songs = ce.num_songs_in_each_cluster[i];
			temp_sum += ce.num_songs_in_each_cluster[i];
			printf("Cluster %d has %d songs.\n", i, ce.num_songs_in_each_cluster[i]);
			lower_hparas_array[i].tcount = create_cluster_hash(pd, myparas.transition_range, ce.cluster_song_relation[i], ce.num_songs_in_each_cluster[i], ce.assignment, num_clusters, i, myparas.inter_cluster_transition_type);

			if(myparas.inter_cluster_transition_type == 0 || myparas.inter_cluster_transition_type == 2)
				ce.X_by_cluster[i] = zerosarray(ce.num_songs_in_each_cluster[i] + num_clusters, d);
			else if(myparas.inter_cluster_transition_type == 1)
				ce.X_by_cluster[i] = zerosarray(ce.num_songs_in_each_cluster[i] + 2 * (num_clusters - 1), d);
			if(myparas.bias_enabled)
			{
				if(myparas.inter_cluster_transition_type == 0 || myparas.inter_cluster_transition_type == 2)
					ce.bias_by_cluster[i] = (double*)calloc(ce.num_songs_in_each_cluster[i] + num_clusters, sizeof(double));
				else if(myparas.inter_cluster_transition_type == 1)
					ce.bias_by_cluster[i] = (double*)calloc(ce.num_songs_in_each_cluster[i] + 2 * (num_clusters - 1), sizeof(double));
			}
		}
		//printf("Total number of songs that have been assigned: %d.\n", temp_sum);
		assert(temp_sum == pd.num_songs);


		//Build Transition to cluster matrix
		printf("Buidling song-cluster transition table....\n");
		for(i = 0; i < num_clusters; i++)
		{
			for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
			{
				find_transitions_to_each_cluster(j, lower_hparas_array[i].tcount, transitions_to_each_cluster[ce.cluster_song_relation[i][j]], num_clusters, i, myparas.inter_cluster_transition_type);
				//void find_transitions_to_each_cluster(int local_song_idx, PHASH* tcount, double* tvec, int num_clusters, int current_cluster_idx)

			}
		}

		/*
		double assert_trans = 0.0;
		for(i = 0; i < k; i++)
			for(j = 0; j < num_clusters; j++)
				assert_trans += transitions_to_each_cluster[i][j];
		printf("%d, %d\n", (int)assert_trans, pd.num_transitions);
		assert((int)assert_trans == pd.num_transitions);
		*/

		printf("Done.\n");
		//print_mat(transitions_to_each_cluster, k, num_clusters);

		double intra_count = 0.0;
		//printf("Norm %f\n", frob_norm(transitions_to_each_cluster, k, num_clusters));
		for(i = 0; i < k; i++)
			intra_count += transitions_to_each_cluster[i][ce.assignment[i]];
		printf("Intra-cluster transition percentage: %f%%.\n", compute_percentage((int)intra_count, pd.num_transitions));

		//CNMap cnm = create_CNmap(num_clusters, 50, ce.num_songs_in_each_cluster);
		//free_CNMap(cnm);

		for(i = 0; i < num_clusters; i++)
		{
			if(!myparas.do_ps)
			{
				if(myparas.bias_enabled)
					llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i], component_time_array + i);
				else
					llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i], component_time_array + i);
			}
			else
			{
				if(myparas.bias_enabled)
					llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i], component_time_array + i);
				else
					llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i], component_time_array + i);
			}
			printf("Lower-level embedding for cluster %d is done.\n", i);
			free_hash(lower_hparas_array[i].tcount);
		}
		//print_vec(llhood_by_cluster, num_clusters);
		//printf("Num of transitions: %d.\n", pd.num_transitions);
		printf("=====================================================\n");
		printf("Avg. training log-likelihood: %f.\n", sum_vec(llhood_by_cluster, num_clusters) / (double) pd.num_transitions);
		printf("=====================================================\n");

		/*
		printf("Reassigning songs...\n");
		for(i = 0; i < num_clusters; i++)
			find_trans_cluster_log_prob(ce.X_by_cluster[i], num_clusters, d, i, p_cluster_square[i]);
		for(i = 0; i < k; i++)
		{
			//for(j = 0; j < num_clusters; j++)
			//	cluster_rating_vec[j] =  innerprod(transitions_to_each_cluster[i], p_cluster_square[j], num_clusters);
			memcpy(cluster_rating_vec, transitions_to_each_cluster[i], num_clusters * sizeof(double));
			int new_cluster = -1;
			double highest_rating = -DBL_MAX; 
			for(j = 0; j < num_clusters; j++)
			{
				if(cluster_rating_vec[j] > highest_rating)
				{
				   highest_rating = cluster_rating_vec[j];
				   new_cluster = j;
				}
			}
			ce.assignment[i] = new_cluster;
		}
		printf("Done.\n");
		*/



		//for(i = 0; i < num_clusters; i++)
		//{
		//	for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
		//	{
		//		current_assignment[ce.cluster_song_relation[i][j]] = find_nearest_hub(ce.X_by_cluster[i][j + num_clusters], ce.X_by_cluster[i], d, num_clusters);
		//	}
		//}

		int temp_hamming_dist = hamming_dist(ce.assignment, last_assignment, k);
		printf("Hamming distance with assignment vector of last iteration is %d.\n", temp_hamming_dist);

		break;
		if(temp_hamming_dist == 0)
			break;
		for(i = 0; i < num_clusters; i++)
		{
			Array2Dfree(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + num_clusters, d);
			free(ce.cluster_song_relation[i]);
		}
	}

	int lme_time = (int)time(NULL) - (int)start_time;
	if(precomputation_time > 0.0)
		printf("Finished. It took %d seconds (%d seconds if precomputation included).\n", lme_time, lme_time + (int) precomputation_time);
	else
		printf("Finished. It took %d seconds.\n", lme_time);
    printf("Time statistics:\n");
    printf("Preclustering: %d seconds\n", precluster_time);
    printf("Sum of time spent on all LME components: %d\n", sum_int_vec(component_time_array, num_clusters));
    printf("Mean of time spent on all LME components: %d\n", (int)((double) sum_int_vec(component_time_array, num_clusters) /(double) num_clusters));
    printf("Max of time spent on all LME components: %d\n", extreme_int_vec(component_time_array, num_clusters, 1));
	//putchar('\n');
	printf("Writing to file %s\n", embedding_file);
	write_ClusterEmbedding_to_file(ce, embedding_file);
	printf("Done.\n");

	//free(ce.assignment);
	free(last_assignment);
	Array2Dfree(transitions_to_each_cluster, k, num_clusters);
	//Array2Dfree(p_cluster_square, num_clusters, num_clusters);
	//free(cluster_rating_vec);
	return ce;
}


CEBD lme_by_cluster_uniportal(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters, int num_portals)
{
	PHASH* total_tcount = create_hash_from_pd(pd);
	CEBD ce;
	ce.num_songs = pd.num_songs;
	ce.d = myparas.d;
	ce.inter_cluster_transition_type = myparas.inter_cluster_transition_type;
	//We shouldn't be here if not for a particular type
	assert(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5 || ce.inter_cluster_transition_type == 6);
	int k = pd.num_songs;
	int i;
	int j;
	int d = myparas.d;
	*l = k;
	ce.num_clusters = num_clusters;
	ce.num_portals = num_portals;
	ce.assignment = (int*)malloc(k * sizeof(int));
	//greedy_assign_cluster_with_constraints(ce.assignment, k, num_clusters, pd, 4.0 * (double)k / num_clusters, 0);
	random_assign_cluster(ce.assignment, k, num_clusters);

	ce.num_songs_in_each_cluster = (int*)malloc(num_clusters * sizeof(int));
	ce.cluster_song_relation = (int**)malloc(num_clusters * sizeof(int*));
	ce.X_by_cluster = (double***)malloc(num_clusters * sizeof(double**));
	if(myparas.bias_enabled)
		ce.bias_by_cluster = (double**)malloc(num_clusters * sizeof(double*));
	else
		ce.bias_by_cluster = NULL;


	HPARAS lower_hparas_array[num_clusters];
	for(i = 0; i < num_clusters; i++)
	{
		lower_hparas_array[i].use_random_init = 0;
		lower_hparas_array[i].num_clusters = num_clusters;
		lower_hparas_array[i].verbose = myparas.verbose;
	}

	int t = 0;
	double llhood_by_cluster[num_clusters];
	double avg_tr_llhood; 
	//double** transitions_to_each_cluster = zerosarray(k, num_clusters);

	time_t start_time = time(NULL);

	//printf("Big iteration %d...\n", t);
	//memcpy(last_assignment, ce.assignment, k * sizeof(int));
	int temp_sum = 0;
	for(i = 0; i < num_clusters; i++)
	{
		ce.num_songs_in_each_cluster[i] =  find_cluster_song_list(ce.assignment, k, i, ce.cluster_song_relation + i);
		lower_hparas_array[i].num_subset_songs = ce.num_songs_in_each_cluster[i];
		temp_sum += ce.num_songs_in_each_cluster[i];
		printf("Cluster %d has %d songs.\n", i, ce.num_songs_in_each_cluster[i]);


		//lower_hparas_array[i].tcount = create_cluster_hash(pd, myparas.transition_range, ce.cluster_song_relation[i], ce.num_songs_in_each_cluster[i], ce.assignment, num_clusters, i, myparas.inter_cluster_transition_type);

		if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 6)
		{
			ce.X_by_cluster[i] = randarray(ce.num_songs_in_each_cluster[i] + num_portals, d, 1.0);
			if(myparas.bias_enabled)
			{
				ce.bias_by_cluster[i] = (double*)calloc(ce.num_songs_in_each_cluster[i] + num_portals, sizeof(double));
			}
		}
		else if(ce.inter_cluster_transition_type == 5)
		{
			ce.X_by_cluster[i] = randarray(ce.num_songs_in_each_cluster[i] + 2 * num_portals, d, 1.0);
			if(myparas.bias_enabled)
			{
				ce.bias_by_cluster[i] = (double*)calloc(ce.num_songs_in_each_cluster[i] + 2 * num_portals, sizeof(double));
			}
		}
	}
	//printf("Total number of songs that have been assigned: %d.\n", temp_sum);
	assert(temp_sum == pd.num_songs);
	//printf("OK till here.\n");
	
	int inter_transition_count = count_inter_transition(ce, total_tcount);
	printf("Inter transition count/percentage: %d/%%%f.\n", inter_transition_count, 100 * (double)inter_transition_count / (double)pd.num_transitions);

	int** last_preferred;
	if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 6)
	{
		ce.preferred_portal_array = zerosarray_int(num_clusters, k);
		last_preferred = zerosarray_int(num_clusters, k); 
	}
	else if(ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5)
	{
		ce.preferred_portal_array = (int**)malloc(num_clusters * sizeof(int*));
		for(i = 0; i < num_clusters; i++)
			ce.preferred_portal_array[i] = (int*)calloc(ce.num_songs_in_each_cluster[i], sizeof(int));

		last_preferred = (int**)malloc(num_clusters * sizeof(int*));
		for(i = 0; i < num_clusters; i++)
			last_preferred[i] = (int*)calloc(ce.num_songs_in_each_cluster[i], sizeof(int));
	}
	//ce.preferred_portal_array = (int**)malloc(num_clusters * sizeof(int*));
	//for(i = 0; i < num_clusters; i++)
		//ce.preferred_portal_array[i] = (int*)malloc(k * sizeof(int));
	
	int temp_max_iter = myparas.max_iter;
	myparas.max_iter = burnin_max_iter;
	while(1)
	{
		if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 6)
			Array2Dcopy_int(ce.preferred_portal_array, last_preferred, num_clusters, k);
		else if(ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5)
			for(i = 0; i < num_clusters; i++)
				memcpy(last_preferred[i], ce.preferred_portal_array[i], ce.num_songs_in_each_cluster[i] * sizeof(int));
		find_preferred_portals(ce, total_tcount);
		build_tcount_array(ce, lower_hparas_array, total_tcount);

        int component_time_array[num_clusters];
		for(i = 0; i < num_clusters; i++)
		{
			if(!myparas.do_ps)
			{
				if(myparas.bias_enabled)
					llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i], component_time_array + i);
				else
					llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i], component_time_array + i);
			}
			else
			{
				if(myparas.bias_enabled)
					llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i], component_time_array + i);
				else
					llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i], component_time_array + i);
			}
			printf("Lower-level embedding for cluster %d is done.\n", i);
			//naive_print_ph_table(lower_hparas_array[i].tcount);
		}
		//print_vec(llhood_by_cluster, num_clusters);

		if(ce.inter_cluster_transition_type == 3)
			avg_tr_llhood = sum_vec(llhood_by_cluster, num_clusters) / (double) pd.num_transitions;
		else if(ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5 || ce.inter_cluster_transition_type == 6)
		{
			avg_tr_llhood = sum_vec(llhood_by_cluster, num_clusters);
			if(inter_transition_count > 0)
				avg_tr_llhood += (double)inter_transition_count * log( 1.0 / (double)(ce.num_clusters - 1));
			avg_tr_llhood /= (double) pd.num_transitions;
		}
		printf("Overall avg. training log-likelihood: %f.\n", avg_tr_llhood);

		int hdist;
		if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 6)
			hdist = hamming_dist_mat(ce.preferred_portal_array, last_preferred, num_clusters, k);
		else if(ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5)
		{
			hdist = 0;
			for(i = 0; i < ce.num_clusters; i++)
				for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
					hdist += (ce.preferred_portal_array[i][j] != last_preferred[i][j]);
		}

		printf("Hamming distance between consecutive prefer-matrx is %d.\n", hdist);

		if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 6)
			if(hdist < 0.01 * (double)(num_clusters * k))
				break;
		if(ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5)
			if(hdist < 0.01 * (double)k)
				break;
		//if(hdist == 0)
			//break;
		for(i = 0; i < num_clusters; i++)
			free_hash(lower_hparas_array[i].tcount);
	}

	printf("\n---------- All the portal configuration is fixed from now on ----------\n\n");
	myparas.max_iter = temp_max_iter;
    int component_time_array[num_clusters];
	for(i = 0; i < num_clusters; i++)
	{
		if(!myparas.do_ps)
		{
			if(myparas.bias_enabled)
				llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i], component_time_array + i);
			else
				llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i], component_time_array + i);
		}
		else
		{
			if(myparas.bias_enabled)
				llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i], component_time_array + i);
			else
				llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i], component_time_array + i);
		}
		printf("Lower-level embedding for cluster %d is done.\n", i);
		//naive_print_ph_table(lower_hparas_array[i].tcount);
	}
	printf("=====================================================\n");
	if(ce.inter_cluster_transition_type == 3)
		avg_tr_llhood = sum_vec(llhood_by_cluster, num_clusters) / (double) pd.num_transitions;
	else if(ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 5 || ce.inter_cluster_transition_type == 6)
	{
		avg_tr_llhood = sum_vec(llhood_by_cluster, num_clusters);
		if(inter_transition_count > 0)
			avg_tr_llhood += (double)inter_transition_count * log( 1.0 / (double)(ce.num_clusters - 1));
		avg_tr_llhood /= (double) pd.num_transitions;
	}
	printf("Avg. training log-likelihood: %f.\n", avg_tr_llhood);
	printf("=====================================================\n");
	for(i = 0; i < num_clusters; i++)
		free_hash(lower_hparas_array[i].tcount);


	//Build Transition to cluster matrix
	//printf("Buidling song-cluster transition table....\n");
	//for(i = 0; i < num_clusters; i++)
	//{
	//	for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
	//	{
	//		find_transitions_to_each_cluster(j, lower_hparas_array[i].tcount, transitions_to_each_cluster[ce.cluster_song_relation[i][j]], num_clusters, i, myparas.inter_cluster_transition_type);

	//	}
	//}


	printf("Done.\n");
	if(embedding_file != NULL)
		write_ClusterEmbedding_to_file(ce, embedding_file);

	//if(ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 6)
	//{
	//	int* cluster_song_counters = (int*)calloc(num_portals, sizeof(int));
	//	for(i = 0; i < num_clusters; i++)
	//		for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
	//			cluster_song_counters[ce.preferred_portal_array[i][j]]++;
	//	print_int_vec(cluster_song_counters, num_portals);
	//	free(cluster_song_counters);
	//}
	//print_mat(transitions_to_each_cluster, k, num_clusters);

	//double intra_count = 0.0;
	//printf("Norm %f\n", frob_norm(transitions_to_each_cluster, k, num_clusters));
	//for(i = 0; i < k; i++)
	//	intra_count += transitions_to_each_cluster[i][ce.assignment[i]];
	//printf("Intra-cluster transition percentage: %f%%.\n", compute_percentage((int)intra_count, pd.num_transitions));

	//CNMap cnm = create_CNmap(num_clusters, 50, ce.num_songs_in_each_cluster);
	//free_CNMap(cnm);

	//for(i = 0; i < num_clusters; i++)
	//{
	//	if(!myparas.do_ps)
	//	{
	//		if(myparas.bias_enabled)
	//			llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i]);
	//		else
	//			llhood_by_cluster[i] = logistic_embed_component(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i]);
	//	}
	//	else
	//	{
	//		if(myparas.bias_enabled)
	//			llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], ce.bias_by_cluster[i], lower_hparas_array[i]);
	//		else
	//			llhood_by_cluster[i] = logistic_embed_component_ps(myparas, ce.X_by_cluster[i], NULL, lower_hparas_array[i]);
	//	}
	//	printf("Lower-level embedding for cluster %d is done.\n", i);
	//	free_hash(lower_hparas_array[i].tcount);
	//}
	//printf("=====================================================\n");
	//printf("Avg. training log-likelihood: %f.\n", sum_vec(llhood_by_cluster, num_clusters) / (double) pd.num_transitions);
	//printf("=====================================================\n");


	////int temp_hamming_dist = hamming_dist(ce.assignment, last_assignment, k);
	////printf("Hamming distance with assignment vector of last iteration is %d.\n", temp_hamming_dist);


	//int lme_time = (int)time(NULL) - (int)start_time;
	//if(precomputation_time > 0.0)
	//	printf("Finished. It took %d seconds (%d seconds if precomputation included).\n", lme_time, lme_time + (int) precomputation_time);
	//else
	//	printf("Finished. It took %d seconds.\n", lme_time);
	//printf("Writing to file %s\n", embedding_file);
	//write_ClusterEmbedding_to_file(ce, embedding_file);
	//printf("Done.\n");



	//free(last_assignment);
	//Array2Dfree(transitions_to_each_cluster, k, num_clusters);
	free_hash(total_tcount);
	Array2Dfree_int(last_preferred, num_clusters, k);
	return ce;
}

void preclustering_by_lme(PDATA pd, PARAS myparas, int num_clusters, int num_portals, int* output_assignment)
{
	int dummy;
	int i;
	int j;
	myparas.max_iter = 0; 
	myparas.inter_cluster_transition_type = 6;
	CEBD ce = lme_by_cluster_uniportal(pd, myparas, NULL, &dummy, num_clusters, num_portals);

	//int* cluster_song_counters = (int*)calloc(num_portals, sizeof(int));
	//memset(output_assignment, 0, num_portals * sizeof(int));
	for(i = 0; i < num_clusters; i++)
		for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
			//output_assignment[ce.cluster_song_relation[i][j]] = ce.preferred_portal_array[i][j];
			output_assignment[ce.cluster_song_relation[i][j]] = find_nearest_hub(ce.X_by_cluster[i][j + ce.num_portals], ce.X_by_cluster[i], ce.d, ce.num_portals);
;
	//print_int_vec(cluster_song_counters, num_portals);
	//free(cluster_song_counters);
	//print_int_vec(output_assignment, ce.num_songs);
	free_ClusterEmbedding(ce);
}

void preclustering_with_aggregated_song(PDATA pd, PARAS myparas, int num_as, double internal_percentage, int* output_assignment)
{
	myparas.bias_enabled = 0;
	myparas.d = 2;
	int i;
	int weights[pd.num_songs];
	count_song_frequency(pd, weights);
	//random_init_aggregated_song(pd.num_songs, num_as, internal_percentage, output_assignment);
	//weighted_random_init_aggregated_song(pd.num_songs, num_as, internal_percentage, output_assignment, weights);
	max_init_aggregated_song(pd.num_songs, num_as, internal_percentage, output_assignment, weights);
	int num_internal_songs = 0;
	int num_external_songs = 0;
	for(i = 0; i < pd.num_songs; i++)
	{
		if(output_assignment[i] == -1)
			num_internal_songs++;
		else
			num_external_songs++;
	}

	printf("\nStarting built-in preclustering...\n");

	printf("Total number of songs: %d\n", pd.num_songs);
	printf("Number of internal songs: %d\n", num_internal_songs);
	printf("Number of external songs: %d\n", num_external_songs);
	printf("Number of aggregated songs: %d\n", num_as);
	assert(num_internal_songs + num_external_songs == pd.num_songs);
	int* internal_song_list = (int*)malloc(num_internal_songs * sizeof(int));
	int* external_song_list = (int*)malloc(num_external_songs * sizeof(int));
	int internal_tag = 0;
	int external_tag = 0;
	for(i = 0; i < pd.num_songs; i++)
	{
		if(output_assignment[i] == -1)
			internal_song_list[internal_tag++] = i;
		else
			external_song_list[external_tag++] = i;
	}
	assert(internal_tag == num_internal_songs);
	assert(external_tag == num_external_songs);
	//print_int_vec(output_assignment, pd.num_songs);
	PHASH* tcount_partial;
	//tcount_total = create_hash_from_pd(pd);
	tcount_partial = create_inter_hash_from_pd(pd, output_assignment);

	//PHASH* tcount;
	double** X = randarray(num_internal_songs + num_as, myparas.d, 1.0);
	double* bias = NULL;
	if(myparas.bias_enabled)
		bias = (double*)calloc(num_internal_songs + num_as, sizeof(double));
	HPARAS hparas;
	hparas.use_random_init = 0;
	hparas.num_clusters = num_as;
	hparas.num_subset_songs = num_internal_songs;
	hparas.verbose = 0;
	//hparas.tcount = tcount;
	double** as_to_internal_table = zerosarray(num_as, num_internal_songs);
	double** internal_to_as_table = zerosarray(num_internal_songs, num_as);
	//int difference;
	double diff_ratio;

	//compute_two_tables(num_as, num_internal_songs, myparas.d, as_to_internal_table, internal_to_as_table, X, bias);

	printf("Initialization done.\n");
	while(1)
	{
		hparas.tcount =  build_tcount_with_aggregate_songs(tcount_partial, output_assignment, num_as, num_internal_songs, internal_song_list);
		printf("tcount built.\n");
        int dummy_time;
		logistic_embed_component(myparas, X, bias, hparas, &dummy_time);
		printf("embedding done.\n");
		//if(bias != NULL)
			//print_vec(bias, num_internal_songs + num_as);
		compute_two_tables(num_as, num_internal_songs, myparas.d, as_to_internal_table, internal_to_as_table, X, bias);
		diff_ratio = reassign_external_songs(tcount_partial, num_as, num_internal_songs, internal_song_list, num_external_songs, external_song_list,  as_to_internal_table, internal_to_as_table, output_assignment);
		//printf("difference is %d\n", difference);
		free_hash(hparas.tcount);
		printf("diff-ratio: %f\n", diff_ratio);
		if(diff_ratio < 0.005)
			break;
	}
	free_hash(tcount_partial);
	PHASH* tcount_total = create_hash_from_pd(pd);
	settle_down_internal_songs(X, num_internal_songs, internal_song_list, myparas.d, pd.num_songs, num_as, output_assignment);
    settle_down_unconnected_external_songs(num_as, pd.num_songs, output_assignment, tcount_total);
	//print_int_vec(output_assignment, pd.num_songs);
	//printf("haha\n");
	Array2Dfree(as_to_internal_table, num_as, num_internal_songs);
	Array2Dfree(internal_to_as_table, num_internal_songs, num_as);
	Array2Dfree(X, num_internal_songs + num_as, myparas.d);
	if(myparas.bias_enabled)
		free(bias);
	free(internal_song_list);
	free(external_song_list);
	free_hash(tcount_total);
	//exit(0);
	printf("\nPreclustering finished.\n");
}

void settle_down_internal_songs(double** X, int num_internal_songs, int* internal_song_list, int d, int num_songs, int num_as, int* assignment)
{
	int i;
	int internal_idx;
	for(i = 0; i < num_songs; i++)
	{
		if(assignment[i] == -1)
		{
			internal_idx = find_in_sorted_list(i, internal_song_list, num_internal_songs, 0);
			assignment[i] = find_nearest_hub(X[num_as + internal_idx], X, d, num_as);
		}
	}
}

void settle_down_unconnected_external_songs(int num_as, int num_songs, int* assignment, PHASH* tcount_total)
{
    int i;
    double** song_cluster_table = zerosarray(num_songs, num_as);
    build_song_cluster_ttable(song_cluster_table, assignment, num_songs, num_as, tcount_total);

    int current_as = 0;
    double bound;
    int candidate_id; 
    while(1)
    {
        bound = -1.0;
        candidate_id = -1;
        for(i = 0; i < num_songs; i++)
        {
            if(assignment[i] == -2)
            {
                if(song_cluster_table[i][current_as] > bound)
                {
                    bound = song_cluster_table[i][current_as];
                    candidate_id = i;
                }
            }
        } 
        if(candidate_id < 0)
            break;
        assignment[candidate_id] = current_as++;
        current_as %= num_as;
    }

    for(i = 0; i < num_songs; i++)
        assert(assignment[i] >= 0);
    Array2Dfree(song_cluster_table, num_songs, num_as);
}

void build_song_cluster_ttable(double** ttable, int* assignment, int num_songs, int num_clusters, PHASH* tcount)
{
    int i;
    for(i = 0; i < num_songs; i++)
       memset(ttable[i], 0, num_clusters * sizeof(double)); 
	HELEM* p_valid_entry;
    int fr;
    int to;
    int fr_cluster;
    int to_cluster;
    double temp_val;
	for(fr = 0; fr < tcount -> num_songs; fr++) 
	{
		p_valid_entry = tcount-> p_first_tran[fr];
		while(p_valid_entry != NULL)
		{
			to = (p_valid_entry -> key.to);
			temp_val = p_valid_entry -> val;
            fr_cluster = assignment[fr];
            to_cluster = assignment[to];
            if(fr_cluster >= 0)
                ttable[to][fr_cluster] += temp_val;
            if(to_cluster >= 0)
                ttable[fr][to_cluster] += temp_val;
			p_valid_entry = (p_valid_entry -> pnext);
		}
	}
}

void random_init_aggregated_song(int num_songs, int num_as, double internal_percentage, int* assignment)
{
	int i;
	srand(time(NULL));
	double r;
	for(i = 0; i < num_songs; i++)
	{
		r = (double)rand()/(double)RAND_MAX;
		//internal song
		if(r < internal_percentage)
			assignment[i] = -1;
		//external song
		else
			assignment[i] = rand() % num_as; 
	}
}
void weighted_random_init_aggregated_song(int num_songs, int num_as, double internal_percentage, int* assignment, int* weights)
{
	//print_int_vec(weights, num_songs);
	int m = (int)((double)num_songs * internal_percentage);
	int candidates[m];
	//printf("haha\n");
	weighted_random_sampling(num_songs, m, weights, candidates);
	//printf("haha\n");
	int i;
	srand(time(NULL));
	for(i = 0; i < num_songs; i++)
		assignment[i] = rand() % num_as; 
	for(i = 0; i < m; i++)
		assignment[candidates[i]] = -1;
}

void max_init_aggregated_song(int num_songs, int num_as, double internal_percentage, int* assignment, int* weights)
{
	int i;
	int m = (int)((double)num_songs * internal_percentage);
	//printf("==========m is %d==========\n", m);
	int candidates[m];
	double double_weights[num_songs];
	for(i = 0; i < num_songs; i++)
		double_weights[i] = (double)(weights[i]);
	//print_int_vec(weights, num_songs);
	//print_vec(double_weights, num_songs);
	extreme_indices(double_weights, num_songs, candidates, m, 1);
	//print_int_vec(candidates, m);
	srand(time(NULL));
	for(i = 0; i < num_songs; i++)
		assignment[i] = rand() % num_as; 
	for(i = 0; i < m; i++)
		assignment[candidates[i]] = -1;
}

PHASH* build_tcount_with_aggregate_songs(PHASH* tcount_total, int* assignment, int num_as, int num_internal_songs, int* internal_song_list)
{
	int fr;
	int to;

    //HELEM temp_elem;
    TPAIR temp_pair;
	PHASH* tcount = create_empty_hash(5 * tcount_total -> num_used, num_internal_songs + num_as);
	double temp_val;

	//for(i = 0; i < pd.num_playlists; i++)
	//{
	//	if(pd.playlists_length[i] > 1)
	//	{
	//		for(j = 0; j < pd.playlists_length[i] - 1; j++)
	//		{
	//			for(t = 1; t <= 1; t++)
	//			{
	//				if(j + t < pd.playlists_length[i])
	//				{
	//					fr = pd.playlists[i][j];
	//					to = pd.playlists[i][j + t];
	//					//We need at least one internal song
	//					if(assignment[fr] == -1 || assignment[to] == -1)
	//					{
	//						if(assignment[fr] == -1)
	//							temp_pair.fr = num_as + find_in_sorted_list(fr, internal_song_list, num_internal_songs, 0);
	//						else
	//							temp_pair.fr = assignment[fr];

	//						if(assignment[to] == -1)
	//							temp_pair.to = num_as + find_in_sorted_list(to, internal_song_list, num_internal_songs, 0);
	//						else
	//							temp_pair.to = assignment[to];

	//						//temp_pair.fr = pd.playlists[i][j];
	//						//temp_pair.to = pd.playlists[i][j + t];

	//						if(temp_pair.fr >= 0 && temp_pair.to >= 0)
	//						{
	//							idx = exist_in_hash(tcount, temp_pair);
	//							if(idx < 0)
	//							{
	//								temp_elem.key = temp_pair;
	//								temp_elem.val = 1.0 / (double) t;
	//								add_entry(tcount, temp_elem);
	//							}
	//							else
	//								update_with(tcount, idx, 1.0 / (double) t);
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//}

	HELEM* p_valid_entry;
	int intra_count = 0;
	int inter_count = 0;
	for(fr = 0; fr < tcount_total -> num_songs; fr++) 
	{
		p_valid_entry = tcount_total -> p_first_tran[fr];
		while(p_valid_entry != NULL)
		{
			to = (p_valid_entry -> key.to);
			temp_val = p_valid_entry -> val;
			if(assignment[fr] == -1 || assignment[to] == -1)
			{
				if(assignment[fr] == -1)
					temp_pair.fr = num_as + find_in_sorted_list(fr, internal_song_list, num_internal_songs, 0);
				else
					temp_pair.fr = assignment[fr];

				if(assignment[to] == -1)
					temp_pair.to = num_as + find_in_sorted_list(to, internal_song_list, num_internal_songs, 0);
				else
					temp_pair.to = assignment[to];
				one_more_count(tcount, temp_pair, temp_val);
				if(assignment[fr] * assignment[to] == 1)
					intra_count += (int)temp_val;
				else
					inter_count += (int)temp_val;
			}
			p_valid_entry = (p_valid_entry -> pnext);
		}
	}
    build_same_song_index(tcount);
	printf("tcount num used: %d\n", tcount -> num_used);
	printf("inter/intra:%d, %d\n", inter_count, intra_count);fflush(stdout);
	return tcount;
}

void compute_two_tables(int num_as, int num_internal_songs, int d, double** as_to_internal_table, double** internal_to_as_table, double** X, double* bias)
{
	int i;
	int j;
	int fr;
	int kk = num_as + num_internal_songs;
	double** delta = zerosarray(kk, d);
	double** tempkd = zerosarray(kk, d);
	double* p = (double*)malloc(kk * sizeof(double));
	double* tempk = (double*)malloc(kk * sizeof(double));

	for(fr = 0; fr < kk; fr++)
	{
		for(j = 0; j < kk; j++)
			for(i = 0; i < d; i++)
				delta[j][i] = X[fr][i] - X[j][i];

		mat_mult(delta, delta, tempkd, kk, d);
		scale_mat(tempkd, kk, d, -1.0);
		sum_along_direct(tempkd, p, kk, d, 1);
		if(bias != NULL)
			add_vec(p, bias, kk, 1.0);
		norm_wo_underflow(p, kk, tempk);
		//print_vec(p, kk);
		//Aggregated songs
		if(fr < num_as)
			memcpy(as_to_internal_table[fr], p + num_as, num_internal_songs * sizeof(double));
		//Internal songs
		else
			memcpy(internal_to_as_table[fr - num_as], p, num_as * sizeof(double));
	}

	//print_mat(as_to_internal_table, num_as, num_internal_songs);
	//printf("====================\n");
	//print_mat(internal_to_as_table, num_internal_songs, num_as);
	Array2Dfree(delta, kk, d);
	Array2Dfree(tempkd, kk, d);
	free(p);
	free(tempk);
	//printf("haha\n");
}

double reassign_external_songs(PHASH* tcount_total, int num_as, int num_internal_songs, int* internal_song_list, int num_external_songs, int* external_song_list, double** as_to_internal_table, double** internal_to_as_table, int* assignment)
{
	int i;
	int fr;
	int to;
	int external_idx;
	int internal_idx;
	HELEM* p_valid_entry;
	double temp_val;
	int num_songs = tcount_total -> num_songs;
	double** vote_array = zerosarray(num_external_songs, num_as);
	int* max_vote_idx_array = (int*)calloc(num_external_songs, sizeof(int));

	for(fr = 0; fr < num_songs; fr++) 
	{
		p_valid_entry = tcount_total -> p_first_tran[fr];
		while(p_valid_entry != NULL)
		{
			to = (p_valid_entry -> key.to);
			temp_val = p_valid_entry -> val;
			if(assignment[fr] == -1 && assignment[to] != -1)
			{
				external_idx = find_in_sorted_list(to, external_song_list, num_external_songs, 0);
				internal_idx = find_in_sorted_list(fr, internal_song_list, num_internal_songs, 0);
				for(i = 0; i < num_as; i++)
					vote_array[external_idx][i] += (temp_val * internal_to_as_table[internal_idx][i]); 
			}
			if(assignment[fr] != -1 && assignment[to] == -1)
			{
				external_idx = find_in_sorted_list(fr, external_song_list, num_external_songs, 0);
				internal_idx = find_in_sorted_list(to, internal_song_list, num_internal_songs, 0);
				for(i = 0; i < num_as; i++)
					vote_array[external_idx][i] += (temp_val * as_to_internal_table[i][internal_idx]); 
			}
			p_valid_entry = (p_valid_entry -> pnext);
		}
	}
	//print_mat(vote_array, num_external_songs, num_as);
	int num_relevant_external_songs = 0;
	for(i = 0; i < num_external_songs; i++)
	{
		//external songs that have connection with internal songs
		if(vote_array[i][0] != 0.0)
		{
			max_vote_idx_array[i] = find_extreme_idx(vote_array[i], num_as, 1);
			num_relevant_external_songs++;
		}
		//otherwise, random assignment
		else
			max_vote_idx_array[i] = rand() % num_as;
	}
	//print_int_vec(max_vote_idx_array, num_external_songs);
	int t = 0;
	int difference = 0;
	for(i = 0; i < num_songs; i++)
	{
		if(assignment[i] != -1)
		{
			if(assignment[i] != max_vote_idx_array[t])
			{
				//printf("(%d, %d)\n", assignment[i], max_vote_idx_array[t]);
				if(vote_array[t][0] != 0.0)
                {
					difference++;
                    assignment[i] = max_vote_idx_array[t];
                }
                else
                    assignment[i] = -2;
			}
			t++;
		}
	}
	printf("difference: %d\n", difference);
	//print_int_vec(assignment, num_songs);
	assert(t == num_external_songs);
	Array2Dfree(vote_array, num_external_songs, num_as);
	free(max_vote_idx_array);
	return (double)difference / (double)num_relevant_external_songs;
}

void lme_by_cluster_test(PDATA pd, CEBD ce, TEST_PARAS myparas)
{
	int bias_enabled = (ce.bias_by_cluster != NULL);
	if(bias_enabled)
		printf("Bias term is enabled.\n");
	else
		printf("Bias term is disabled.\n");
	if(pd.num_songs != ce.num_songs)
	{
		printf("Numbers of songs in the playlist data and the embedding data do not match.\n");
		exit(1);
	}
	int i;
	//for(i = 0; i < ce.num_clusters; i++)
	//	print_mat(ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i] + ce.num_clusters, ce.d);
	//printf("Done printing..\n");
	//for(i = 0; i < ce.num_clusters; i++)
	//	print_int_vec(ce.cluster_song_relation[i], ce.num_songs_in_each_cluster[i]);
	//print_int_vec(ce.assignment, ce.num_songs);
	double llhood = 0.0;
	PHASH* tcount_array[ce.num_clusters];
	for(i = 0; i < ce.num_clusters; i++)
	{
		tcount_array[i] = create_cluster_hash(pd, myparas.transition_range, ce.cluster_song_relation[i], ce.num_songs_in_each_cluster[i], ce.assignment, ce.num_clusters, i, ce.inter_cluster_transition_type);
		//printf("Done %d ce.\n", i);
		if(bias_enabled)
			llhood += logistic_test_component(ce.X_by_cluster[i], ce.bias_by_cluster[i], tcount_array[i], ce.d);
		else
			llhood += logistic_test_component(ce.X_by_cluster[i], NULL, tcount_array[i], ce.d);
		free_hash(tcount_array[i]);
		//printf("LDone %d ce.\n", i);
	}
    printf("Avg log-likelihood on test: %f\n", llhood / (double) pd.num_transitions);
}

void random_assign_cluster(int* assignment, int k, int num_clusters)
{
	srand(time(NULL));
	int i;
	int temp;
	for(i = 0; i < k; i++)
	{
		temp = (int) (((double)rand()) / ((double) RAND_MAX) * (double) num_clusters);
		assignment[i] = temp == num_clusters? (temp - 1) : temp;
	}
}

void altenate_assign_cluster(int* assignment, int k, int num_clusters)
{
	int i;
	for(i = 0; i < k; i++)
		assignment[i] = i % num_clusters;
}

void greedy_assign_cluster(int* assignment, int k, int num_clusters, PDATA pd)
{
	int i;
	int j;
	int t = 1;
	int shift = 0;
	random_assign_cluster(assignment, k, num_clusters);
	int* last_assignment = (int*)malloc(k * sizeof(int));
	int temp_hamming_dist;
	int fr;
	int to;
	double** song_cluster_transitions = zerosarray(k, num_clusters);
	int counter = 0;
	while(++counter)
	{
		memcpy(last_assignment, assignment, k * sizeof(int));
		for(i = 0; i < k; i++)
			memset(song_cluster_transitions[i], 0, num_clusters * sizeof(double));

		for(i = 0; i < pd.num_playlists; i ++)
		{
			if(pd.playlists_length[i] > 1)
			{
				for(j = 0; j < pd.playlists_length[i] - 1; j++)
				{
					if(j + t < pd.playlists_length[i])
					{
						fr = pd.playlists[i][j] + shift;
						to = pd.playlists[i][j + t] + shift;
						if(fr >= shift && to >= shift)
							song_cluster_transitions[fr][assignment[to]]++;
					}
				}
			}
		}

		for(i = 0; i < k; i++)
			assignment[i] = extreme_id(song_cluster_transitions[i], num_clusters, 1);

		temp_hamming_dist = hamming_dist(assignment, last_assignment, k);
		printf("Hamming distance with assignment vector of last iteration is %d.\n", temp_hamming_dist);
		if(temp_hamming_dist == 0 || counter > 50)
			break;
	}
	Array2Dfree(song_cluster_transitions, k, num_clusters);
	free(last_assignment);
}

void greedy_assign_cluster_with_constraints(int* assignment, int k, int num_clusters, PDATA pd, int upper, int lower)
{
	int i;
	int j;
	int fr;
	int to;
	int s;
	int t;
	int change_tag;
	int cluster_to_switch;
	double gain;
	double temp_val;
	double temp_val1;
	double temp_val2;
	TPAIR temp_pair;
	QELEM* p;
	//altenate_assign_cluster(assignment, k, num_clusters);
	random_assign_cluster(assignment, k, num_clusters);
	IDXQUEUE cs_relation[num_clusters];
	double** sc_transition = zerosarray(k, num_clusters);
	double** cs_transition = zerosarray(num_clusters, k);
	//double** cc_transition = zerosarray(num_clusters, num_clusters);
	int* last_assignment = (int*)malloc(k * sizeof(int));
	for(i = 0; i < num_clusters; i++)
		init_idx_queue(cs_relation + i);
	for(i = 0; i < k; i++)
		push_in_idx_queue(i, cs_relation + assignment[i]);

	/*
	for(i = 0; i < num_clusters; i++)
		printf("Length: %d\n", (cs_relation + i) -> length);
		*/

	PHASH* tcount = create_empty_hash(2 * pd.num_transitions, k);

	for(i = 0; i < pd.num_playlists; i ++)
	{
		if(pd.playlists_length[i] > 1)
		{
			for(j = 0; j < pd.playlists_length[i] - 1; j++)
			{
				fr = pd.playlists[i][j];
				to = pd.playlists[i][j + 1];
				sc_transition[fr][assignment[to]]++;
				cs_transition[assignment[fr]][to]++;
				//cc_transition[assignment[fr]][assignment[to]]++;
				temp_pair.fr = fr;
				temp_pair.to = to;
				one_more_count(tcount, temp_pair, 1.0);
			}
		}
	}
	build_same_song_index(tcount);
	
	double temp_sum = 0.0;
	for(i = 0; i < k; i++)
		temp_sum += sum_vec(sc_transition[i], num_clusters);
	assert((int) temp_sum == pd.num_transitions);
	

	while(1)
	{
		change_tag = 0;
		memcpy(last_assignment, assignment, k * sizeof(int));
		for(i = 0; i < k && (cs_relation + assignment[i]) -> length > lower; i++)
		{
			//printf("OK till here.\n");
			cluster_to_switch = -1;
			gain = -DBL_MAX;
			for(j = 0; j < num_clusters && j != assignment[i] && (cs_relation + j) -> length < upper ; j++) 
			{
				temp_val = sc_transition[i][j] + cs_transition[j][i] - sc_transition[i][assignment[i]] - cs_transition[assignment[i]][i];
				if(temp_val > gain)
				{
					gain = temp_val;
					cluster_to_switch = j;
				}
			}
			//printf("gain %f\n", gain);
			if(gain > 0)
			{
				//printf("(%d, %d)\n", assignment[i], cluster_to_switch);
				//Do switch i from assignment[i] to cluster_to_switch
				change_tag++;

				//cc_transition[assignment[i]][assignment[i]] -= sc_transition[i][assignment[i]];
				//cc_transition[assignment[i]][cluster_to_switch] += sc_transition[i][cluster_to_switch];
				//cc_transition[cluster_to_switch][assignment[i]] -= cs_transition[assignment[i]][i];
				//cc_transition[cluster_to_switch][cluster_to_switch] += cs_transition[cluster_to_switch][i];

				for(t = 0; t < k; t++)
				{
					temp_pair.fr = t;
					temp_pair.to = i;
					temp_val1 = get_pair_value(tcount, temp_pair);
					temp_pair.fr = i;
					temp_pair.to = t;
					temp_val2 =  get_pair_value(tcount, temp_pair);
					sc_transition[t][assignment[i]] -= temp_val1;
					sc_transition[t][cluster_to_switch] += temp_val1;
					cs_transition[assignment[i]][t] -= temp_val2;
					cs_transition[cluster_to_switch][t] += temp_val2;
				}

				//p = (cs_relation + assignment[i]) -> head;
				//for(s = 0; s < (cs_relation + assignment[i]) -> length; s++)
				//{
				//	t = p -> idx;
				//	if(t != i)
				//	{
				//		temp_pair.fr = t;
				//		temp_pair.to = i;
				//		temp_val1 = get_pair_value(tcount, temp_pair);
				//		temp_pair.fr = i;
				//		temp_pair.to = t;
				//		temp_val2 =  get_pair_value(tcount, temp_pair);
				//		//temp_val = get_undirected_transition_value(tcount, i, t);
				//		sc_transition[t][assignment[i]] -= temp_val1;
				//		sc_transition[t][cluster_to_switch] += temp_val1;
				//		cs_transition[assignment[i]][t] -= temp_val2;
				//		cs_transition[cluster_to_switch][t] += temp_val2;
				//	}
				//	p = p -> pnext;
				//}

				//p = (cs_relation + cluster_to_switch) -> head;
				//for(s = 0; s < (cs_relation + cluster_to_switch) -> length; s++)
				//{
				//	t = p -> idx;

				//	temp_pair.fr = t;
				//	temp_pair.to = i;
				//	temp_val1 =  get_pair_value(tcount, temp_pair);
				//	temp_pair.fr = i;
				//	temp_pair.to = t;
				//	temp_val2 =  get_pair_value(tcount, temp_pair);

				//	//temp_val = get_undirected_transition_value(tcount, i, t);
				//	sc_transition[t][cluster_to_switch] += temp_val1;
				//	sc_transition[t][assignment[i]] -= temp_val1;
				//	cs_transition[cluster_to_switch][t] += temp_val2;
				//	cs_transition[assignment[i]][t] -= temp_val2;
				//	p = p -> pnext;
				//}

				delete_from_idx_queue(cs_relation + assignment[i], i);
				push_in_idx_queue(i, cs_relation + cluster_to_switch);
				assignment[i] = cluster_to_switch;
			}
		}
		//printf("%d songs changed memebership in last iteration.\n", change_tag);
		//for(i = 0; i < num_clusters; i++)
		//	printf("Length: %d\n", (cs_relation + i) -> length);
		//int temp_hamming_dist = hamming_dist(assignment, last_assignment, k);
		//printf("Hamming distance with assignment vector of last iteration is %d.\n", temp_hamming_dist);
		double temp_sum = 0.0;
		for(i = 0; i < k; i++)
			temp_sum += sum_vec(sc_transition[i], num_clusters);
		assert((int) temp_sum == pd.num_transitions);
		
		temp_sum = 0.0;
		for(i = 0; i < num_clusters; i++)
			temp_sum += sum_vec(cs_transition[i], k);
		assert((int) temp_sum == pd.num_transitions);

		/*
		temp_sum = 0.0;
		for(i = 0; i < num_clusters; i++)
			temp_sum += sum_vec(cc_transition[i], num_clusters);
		assert((int) temp_sum == pd.num_transitions);
		*/

		temp_sum = 0.0;
		for(i = 0; i < k; i++)
			temp_sum += sc_transition[i][assignment[i]];
		//printf("Norm %f\n", frob_norm(sc_transition, k, num_clusters));
		//printf("Intra-cluster transition percentage: \%f%%.\n", compute_percentage((int)temp_sum, pd.num_transitions));
		if(!change_tag)
			break;
	}

	//for(i = 0; i < num_clusters; i++)
	//	printf("Length: %d\n", (cs_relation + i) -> length);

	//printf("Time to free.\n");
	for(i = 0; i < num_clusters; i++)
		free_idx_queue(cs_relation + i);
	Array2Dfree(sc_transition, k, num_clusters);
	Array2Dfree(cs_transition, num_clusters, k);
	//Array2Dfree(cc_transition, num_clusters, num_clusters);
	free_hash(tcount);
}

int count_intra_cluster_transitions(int* assignment, int k, int num_clusters, PDATA pd)
{
	int i;
	int j;
	int fr;
	int to;
	int counter = 0;

	for(i = 0; i < pd.num_playlists; i++)
	{
		if(pd.playlists_length[i] > 1)
		{
			for(j = 0; j < pd.playlists_length[i] - 1; j++)
			{
					fr = pd.playlists[i][j];
					to = pd.playlists[i][j + 1];
					if(assignment[fr] == assignment[to])
						counter++;
			}
		}
	}
	return counter;
}

void find_transitions_to_each_cluster(int local_song_idx, PHASH* tcount, double* tvec, int num_clusters, int current_cluster_idx, int type)
{
	int to;
	memset(tvec, 0, num_clusters * sizeof(double));
	HELEM* p_valid_entry;
	if(type == 0 || type == 2)
		p_valid_entry = tcount -> p_first_tran[local_song_idx + num_clusters];
	else if(type == 1)
		p_valid_entry = tcount -> p_first_tran[local_song_idx + 2 * (num_clusters - 1)];
	while(p_valid_entry != NULL)
	{
		to = (p_valid_entry -> key.to);
		if(type == 0 || type == 2)
		{
			if(to < num_clusters)
			{
				assert(to != current_cluster_idx);
				tvec[to] += (p_valid_entry -> val);
			}
			else
				tvec[current_cluster_idx] += (p_valid_entry -> val);
		}
		else if(type == 1)
		{
			if(to < 2 * (num_clusters - 1))
			{
				assert(to >= num_clusters - 1);
				int temp_int = to - (num_clusters - 1);
				temp_int = temp_int < current_cluster_idx ? temp_int : temp_int + 1; 
				tvec[temp_int] += (p_valid_entry -> val);
			}
			else
				tvec[current_cluster_idx] += (p_valid_entry -> val);
		}
		p_valid_entry = (p_valid_entry -> pnext);
	}
}

PHASH* create_cluster_hash(PDATA pd, int transition_range, int* cluster_idx_list, int cluster_song_count, int* member_vec, int num_clusters, int current_cluster_id, int type)
{
	int i;
	int j;
	int t;
	int idx;
	int fr;
	int to;
	PHASH* tcount;
	//double** tcount_full;

	HELEM temp_elem;
	TPAIR temp_pair;

	assert(type == 0 || type == 1 || type == 2);


	if(type == 0 || type == 2)
		tcount = create_empty_hash(2 * transition_range * pd.num_transitions, cluster_song_count + num_clusters);
	else if(type == 1)
		tcount = create_empty_hash(2 * transition_range * pd.num_transitions, cluster_song_count + 2 * (num_clusters - 1));

	for(i = 0; i < pd.num_playlists; i ++)
	{
		if(pd.playlists_length[i] > 1)
		{
			for(j = 0; j < pd.playlists_length[i] - 1; j++)
			{
				for(t = 1; t <= transition_range; t++)
				{
					if(j + t < pd.playlists_length[i])
					{
						fr = pd.playlists[i][j];
						to = pd.playlists[i][j + t];
						if(fr >= 0 && to >= 0)
						{
							if(type == 0)
							{
								temp_pair.fr = member_vec[fr] == current_cluster_id? find_in_sorted_list(fr, cluster_idx_list, cluster_song_count, 0) + num_clusters : current_cluster_id;
								temp_pair.to = member_vec[to] == current_cluster_id? find_in_sorted_list(to, cluster_idx_list, cluster_song_count, 0) + num_clusters : member_vec[to];
								if(temp_pair.fr >= num_clusters || temp_pair.to >= num_clusters)
									one_more_count(tcount, temp_pair, 1.0 / (double) t);
							}
							else if(type == 2)
							{
								temp_pair.fr = member_vec[fr] == current_cluster_id? find_in_sorted_list(fr, cluster_idx_list, cluster_song_count, 0) + num_clusters : member_vec[fr];
								temp_pair.to = member_vec[to] == current_cluster_id? find_in_sorted_list(to, cluster_idx_list, cluster_song_count, 0) + num_clusters : member_vec[to];
								if(temp_pair.fr >= num_clusters || temp_pair.to >= num_clusters)
									one_more_count(tcount, temp_pair, 1.0 / (double) t);
							}
							else if(type == 1)
							{
								temp_pair.fr = member_vec[fr] == current_cluster_id? find_in_sorted_list(fr, cluster_idx_list, cluster_song_count, 0) + 2 * (num_clusters - 1) : find_idx_with_self_omit(member_vec[fr], current_cluster_id, num_clusters, 0) ;
								temp_pair.to = member_vec[to] == current_cluster_id? find_in_sorted_list(to, cluster_idx_list, cluster_song_count, 0) + 2 * (num_clusters - 1) : find_idx_with_self_omit(member_vec[to], current_cluster_id, num_clusters, 1) ;
								if(temp_pair.fr >= 2 * (num_clusters - 1) || temp_pair.to >= 2 * (num_clusters - 1))
									one_more_count(tcount, temp_pair, 1.0 / (double) t);
							}
						}
					}
				}
			}
		}
	}
	build_same_song_index(tcount);
	//naive_print_ph_table(tcount);
	return tcount;
}

PHASH* create_hash_from_pd(PDATA pd)
{
	int t;
	int i;
	int j;
	int idx;

    HELEM temp_elem;
    TPAIR temp_pair;
	PHASH* tcount = create_empty_hash(2 * pd.num_transitions, pd.num_songs);
	//printf("Transition matrix created.\n");
	for(i = 0; i < pd.num_playlists; i++)
	{
		if(pd.playlists_length[i] > 1)
		{
			for(j = 0; j < pd.playlists_length[i] - 1; j++)
			{
				for(t = 1; t <= 1; t++)
				{
					if(j + t < pd.playlists_length[i])
					{
						temp_pair.fr = pd.playlists[i][j];
						temp_pair.to = pd.playlists[i][j + t];

						if(temp_pair.fr >= 0 && temp_pair.to >= 0)
						{
							idx = exist_in_hash(tcount, temp_pair);
							if(idx < 0)
							{
								temp_elem.key = temp_pair;
								temp_elem.val = 1.0 / (double) t;
								add_entry(tcount, temp_elem);
							}
							else
								update_with(tcount, idx, 1.0 / (double) t);
						}
					}
				}
			}
		}
	}
    build_same_song_index(tcount);

	//Sanity check
	//int temp_int = 0;
	//for(i = 0; i < tcount -> length; i++)
	//{
	//	if(!is_null_entry((tcount -> p)[i].key))
	//		temp_int++;
	//} 
	//printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(k * k)));
	//printf("Transition matrix initialized.\n");
	return tcount;
}

PHASH* create_inter_hash_from_pd(PDATA pd, int* assignment)
{
	int t;
	int i;
	int j;
	int idx;

    HELEM temp_elem;
    TPAIR temp_pair;
	PHASH* tcount = create_empty_hash(2 * pd.num_transitions, pd.num_songs);
	//printf("Transition matrix created.\n");
	for(i = 0; i < pd.num_playlists; i++)
	{
		if(pd.playlists_length[i] > 1)
		{
			for(j = 0; j < pd.playlists_length[i] - 1; j++)
			{
				for(t = 1; t <= 1; t++)
				{
					if(j + t < pd.playlists_length[i])
					{
						temp_pair.fr = pd.playlists[i][j];
						temp_pair.to = pd.playlists[i][j + t];

						if(temp_pair.fr >= 0 && temp_pair.to >= 0 && (assignment[temp_pair.fr] == -1 || assignment[temp_pair.to] == -1))
						{
							idx = exist_in_hash(tcount, temp_pair);
							if(idx < 0)
							{
								temp_elem.key = temp_pair;
								temp_elem.val = 1.0 / (double) t;
								add_entry(tcount, temp_elem);
							}
							else
								update_with(tcount, idx, 1.0 / (double) t);
						}
					}
				}
			}
		}
	}
    build_same_song_index(tcount);
	return tcount;
}

void find_preferred_portals(CEBD ce, PHASH* total_tcount)
{
	if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 6)
	{
		int i;
		int j;

		for(i = 0; i < ce.num_clusters; i++)
			for(j = 0; j < ce.num_songs; j++)
				ce.preferred_portal_array[i][j] = -2;

		for(i = 0; i < ce.num_clusters; i++)
			for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
				ce.preferred_portal_array[i][ce.cluster_song_relation[i][j]] = -1; //no portal required for local transition
		int** every_songs_nearest_portal = (int**)malloc(ce.num_clusters * sizeof(int*));
		for(i = 0; i < ce.num_clusters; i++)
		{
			every_songs_nearest_portal[i] = (int*)malloc(ce.num_songs_in_each_cluster[i] * sizeof(int));
			assign_clusters(ce.X_by_cluster[i] + ce.num_portals, ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i], ce.d, ce.num_portals, every_songs_nearest_portal[i]);
		}
		double*** vote_counting_array = (double***)malloc(ce.num_clusters * sizeof(double**));
		for(i = 0; i < ce.num_clusters; i++)
			vote_counting_array[i] = zerosarray(ce.num_songs, ce.num_portals);

		HELEM* p_valid_entry;
		int fr;
		int to;
		int fr_cluster;
		int to_cluster;
		int local_fr;
		int local_to;
		double temp_val;
		for(fr = 0; fr < ce.num_songs; fr++)
		{
			p_valid_entry = (total_tcount) -> p_first_tran[fr];
			while(p_valid_entry != NULL)
			{
				to = (p_valid_entry -> key.to);
				temp_val = p_valid_entry -> val;
				fr_cluster = ce.assignment[fr];
				to_cluster = ce.assignment[to];
				if(fr_cluster != to_cluster)
				{
					local_fr = find_in_sorted_list(fr, ce.cluster_song_relation[fr_cluster], ce.num_songs_in_each_cluster[fr_cluster], 0);
					local_to = find_in_sorted_list(to, ce.cluster_song_relation[to_cluster], ce.num_songs_in_each_cluster[to_cluster], 0);
					if(ce.inter_cluster_transition_type == 3)
					{
						vote_counting_array[fr_cluster][to][every_songs_nearest_portal[fr_cluster][local_fr]] += temp_val;
						vote_counting_array[to_cluster][fr][every_songs_nearest_portal[to_cluster][local_to]] += temp_val;
					}
					else if(ce.inter_cluster_transition_type == 6)
					{
						int temp_portal = every_songs_nearest_portal[fr_cluster][local_fr];
						vote_counting_array[fr_cluster][to][temp_portal] += temp_val;
						vote_counting_array[to_cluster][fr][temp_portal] += temp_val;
					}
				}
				p_valid_entry = (p_valid_entry -> pnext);
			}
		}
		//printf("OK till here.\n");

		for(i = 0; i < ce.num_clusters; i++)
			for(j = 0; j < ce.num_songs; j++)
				if(ce.preferred_portal_array[i][j] != -1)
					ce.preferred_portal_array[i][j] = extreme_id(vote_counting_array[i][j], ce.num_portals, 1);
		//for(i = 0; i < ce.num_clusters; i++)
		//print_int_vec(ce.preferred_portal_array[i], ce.num_songs);

		for(i = 0; i < ce.num_clusters; i++)
			free(every_songs_nearest_portal[i]);
		free(every_songs_nearest_portal);

		for(i = 0; i < ce.num_clusters; i++)
			Array2Dfree(vote_counting_array[i], ce.num_songs, ce.num_portals);
		free(vote_counting_array);
	}
	else if(ce.inter_cluster_transition_type == 4)
	{
		int i;
		for(i = 0; i < ce.num_clusters; i++)
			assign_clusters(ce.X_by_cluster[i] + ce.num_portals, ce.X_by_cluster[i], ce.num_songs_in_each_cluster[i], ce.d, ce.num_portals, ce.preferred_portal_array[i]);
	}
	else if(ce.inter_cluster_transition_type == 5)
	{
		int i;
		for(i = 0; i < ce.num_clusters; i++)
			assign_clusters(ce.X_by_cluster[i] + 2 * ce.num_portals, ce.X_by_cluster[i] + ce.num_portals, ce.num_songs_in_each_cluster[i], ce.d, ce.num_portals, ce.preferred_portal_array[i]);
	}
}

void build_tcount_array(CEBD ce, HPARAS* hparas_array, PHASH* total_tcount)
{
	HELEM* p_valid_entry;
	int i;
	int j;

	for(i = 0; i < ce.num_clusters; i++)
		if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 6)
			hparas_array[i].tcount = create_empty_hash(2 * (total_tcount -> num_used), ce.num_songs_in_each_cluster[i] + ce.num_portals);
		else if(ce.inter_cluster_transition_type == 5)
			hparas_array[i].tcount = create_empty_hash(2 * (total_tcount -> num_used), ce.num_songs_in_each_cluster[i] + 2 * ce.num_portals);
  
	int fr;
	int to;
	int fr_cluster;
	int to_cluster;
	int local_fr;
	int local_to;
	double temp_val;
	TPAIR temp_pair;
	for(fr = 0; fr < ce.num_songs; fr++)
	{
		p_valid_entry = (total_tcount) -> p_first_tran[fr];
		while(p_valid_entry != NULL)
		{
			to = (p_valid_entry -> key.to);
			temp_val = p_valid_entry -> val;
			fr_cluster = ce.assignment[fr];
			to_cluster = ce.assignment[to];
			local_fr = find_in_sorted_list(fr, ce.cluster_song_relation[fr_cluster], ce.num_songs_in_each_cluster[fr_cluster], 0);
			local_to = find_in_sorted_list(to, ce.cluster_song_relation[to_cluster], ce.num_songs_in_each_cluster[to_cluster], 0);
			if(fr_cluster != to_cluster)
			{
				if(ce.inter_cluster_transition_type == 3)
				{
					temp_pair.fr = local_fr + ce.num_portals;
					temp_pair.to = ce.preferred_portal_array[fr_cluster][to];
					one_more_count(hparas_array[fr_cluster].tcount, temp_pair, temp_val);

					temp_pair.fr = ce.preferred_portal_array[to_cluster][fr];
					temp_pair.to = local_to + ce.num_portals;
					one_more_count(hparas_array[to_cluster].tcount, temp_pair, temp_val);
				}
				else if(ce.inter_cluster_transition_type == 4)
				{
					temp_pair.fr = local_fr + ce.num_portals;
					temp_pair.to = ce.preferred_portal_array[fr_cluster][local_fr];
					one_more_count(hparas_array[fr_cluster].tcount, temp_pair, temp_val);
					temp_pair.fr = temp_pair.to; 
					temp_pair.to = local_to + ce.num_portals;
					one_more_count(hparas_array[to_cluster].tcount, temp_pair, temp_val);
				}
				else if(ce.inter_cluster_transition_type == 5)
				{
					temp_pair.fr = local_fr + 2 * ce.num_portals;
					temp_pair.to = ce.preferred_portal_array[fr_cluster][local_fr] + ce.num_portals;
					one_more_count(hparas_array[fr_cluster].tcount, temp_pair, temp_val);
					temp_pair.fr = temp_pair.to - ce.num_portals; 
					temp_pair.to = local_to + 2 * ce.num_portals;
					one_more_count(hparas_array[to_cluster].tcount, temp_pair, temp_val);
				}
				else if(ce.inter_cluster_transition_type == 6)
				{
					temp_pair.fr = local_fr + ce.num_portals;
					temp_pair.to = ce.preferred_portal_array[fr_cluster][to];
					one_more_count(hparas_array[fr_cluster].tcount, temp_pair, temp_val);

					temp_pair.fr = temp_pair.to;
					temp_pair.to = local_to + ce.num_portals;
					one_more_count(hparas_array[to_cluster].tcount, temp_pair, temp_val);
				}
			}
			else
			{
				if(ce.inter_cluster_transition_type == 3 || ce.inter_cluster_transition_type == 4 || ce.inter_cluster_transition_type == 6)
				{
					temp_pair.fr = local_fr + ce.num_portals;
					temp_pair.to = local_to + ce.num_portals;
					one_more_count(hparas_array[fr_cluster].tcount, temp_pair, temp_val);
				}
				else if(ce.inter_cluster_transition_type == 5)
				{
					temp_pair.fr = local_fr + 2 * ce.num_portals;
					temp_pair.to = local_to + 2 * ce.num_portals;
					one_more_count(hparas_array[fr_cluster].tcount, temp_pair, temp_val);
				}
			}
			p_valid_entry = (p_valid_entry -> pnext);
		}
	}
	for(i = 0; i < ce.num_clusters; i++)
		build_same_song_index(hparas_array[i].tcount);
}

int count_inter_transition(CEBD ce, PHASH* total_tcount)
{
	int fr;
	int to;
	double tr_count = 0.0;
	HELEM* p_valid_entry;
	for(fr = 0; fr < ce.num_songs; fr++)
	{
		p_valid_entry = (total_tcount) -> p_first_tran[fr];
		while(p_valid_entry != NULL)
		{
			to = (p_valid_entry -> key.to);
			if(ce.assignment[fr] != ce.assignment[to])
				tr_count += (p_valid_entry -> val);
			p_valid_entry = (p_valid_entry -> pnext);
		}
	}
	return (int)tr_count;
}

#ifdef MPI
#include <mpi.h>
double logistic_embed_component_with_comm(PARAS myparas, double** X, double* bias, HPARAS hparas, MPI_Comm comm, int* seconds_used)
{
	int myrank;
	int comm_sz;
	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &comm_sz);
	int n; //Number of training transitions
	int i;
	int j;
	//int s;
	//int q;
	int t;
	int fr;
	int to;
	int d = myparas.d;
	int k = hparas.num_subset_songs;
	//int kk = hparas.num_subset_songs + hparas.num_clusters;
	int kk = (hparas.tcount) -> num_songs;
	double eta;
	int idx;
	if (myparas.ita > 0)
		eta = myparas.ita;
	else
		eta = k;
	n = 0;
	for(i = 0; i < ((hparas.tcount) -> length); i++)
		if(!is_null_entry(((hparas.tcount) -> p)[i].key))
			n += (int)((hparas.tcount) -> p)[i].val;
	/*
	   for(i = 0; i < pd.num_playlists; i ++)
	   if(pd.playlists_length[i] > 0)
	   n += pd.playlists_length[i] - 1;
	   */
	if(hparas.verbose && myrank == 0)
		printf("Altogether %d transitions.\n", n);fflush(stdout);

	if(hparas.use_random_init && myrank == 0)
		randfill(X, kk, d, 1.0);

	double llhood = 0.0;
	double llhoodprev;
	double realn;

	double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
	int llhood_track_index = 0;

	for (i = 0; i < myparas.num_llhood_track; i++)
		last_n_llhoods[i] = 1.0;

	time_t start_time;
	clock_t start_clock, temp_clock;

	double** cdx = zerosarray(kk, d);
	double** delta = zerosarray(kk, d);
	double** dv = zerosarray(kk, d);
	double* p = (double*)calloc(kk, sizeof(double));
	double* tempk = (double*)calloc(kk, sizeof(double));
	double** tempkd = zerosarray(kk, d);
	double* dufr = (double*)calloc(d, sizeof(double));
	double temp_val;
	HELEM* p_valid_entry;

	double* bias_grad;
	double* bias_grad_accum;
	if(myparas.bias_enabled)
	{
		bias_grad = (double*)calloc(kk, sizeof(double));
		bias_grad_accum = (double*)calloc(kk, sizeof(double));
		memset(bias, 0, kk * sizeof(double));
	}

	//printf("Before. Norm of the hubs: %f.\n", frob_norm(X, hparas.num_clusters, d));
	

	//Communication arrays
    double local_llhood_realn[2];
    double global_llhood_realn[2];
	int break_signal = 0;
    double* comm_vec = (double*)malloc(kk * d * sizeof(double));
    double* sum_of_reduce = (double*)malloc(kk * d * sizeof(double));

    if(myrank == 0)
        pack_mat(X, comm_vec, kk, d);
    MPI_Bcast(comm_vec, kk * d, MPI_DOUBLE, 0, comm);
    if(myrank != 0)
        unpack_mat(comm_vec, X, kk, d);

    if(myparas.bias_enabled)
        MPI_Bcast(bias, kk, MPI_DOUBLE, 0, comm);

	time_t total_time_counter = time(NULL);

	for(t = 0; t < myparas.max_iter; t++)
	{
        if(break_signal)
            break;
		llhoodprev = llhood;
		llhood = 0.0;
		realn = 0.0;
        if(myrank == 0)
        {
            start_time = time(NULL);
            start_clock = clock();
			if(hparas.verbose)
				printf("Little Iteration %d\n", t + 1);fflush(stdout);
            memset(sum_of_reduce, 0, kk * d * sizeof(double));
            global_llhood_realn[0] = 0.0;
            global_llhood_realn[1] = 0.0;
        }

		//Begin parallelization
        local_llhood_realn[0] = 0.0;
        local_llhood_realn[1] = 0.0;
		//for(s = 0; s < hparas.num_subset_songs; s++)
		//for(fr = 0; fr < hparas.num_subset_songs; fr++)
		int check_point_counter;
		for(fr = myrank, check_point_counter = 1; fr < kk; fr += comm_sz, check_point_counter++)
		{
			for(j = 0; j < kk; j++)
				for(i = 0; i < d; i++)
					delta[j][i] = X[fr][i] - X[j][i];

			mat_mult(delta, delta, tempkd, kk, d);
			scale_mat(tempkd, kk, d, -1.0);
			sum_along_direct(tempkd, p, kk, d, 1);
			if(myparas.bias_enabled)
                add_vec(p, bias, kk, 1.0);
			norm_wo_underflow(p, kk, tempk);
			for(j = 0; j < kk; j++)
				for(i = 0; i < d; i++)
					dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
			sum_along_direct(dv, dufr, kk, d, 0);

			double accu_temp_vals = 0.0;
			p_valid_entry = (hparas.tcount) -> p_first_tran[fr];
			while(p_valid_entry != NULL)
			{
				to = (p_valid_entry -> key.to);
				//q = find_in_list(to, hparas.idx_vec, hparas.num_subset_songs); 
				temp_val = p_valid_entry -> val;
				accu_temp_vals += temp_val;
                if (myparas.bias_enabled)
					bias_grad[to] += temp_val;
				add_vec(cdx[to], delta[to], d, 2.0 * temp_val);
				/*
				   Veccopy(dufr, tempd, d);
				   add_vec(tempd, delta[to], d, 2.0);
				   scale_vec(tempd, d, -1.0);
				   add_vec(cdx[fr], tempd, d, temp_val);
				   */
				add_vec(cdx[fr], delta[to], d, -2.0 * temp_val);
                local_llhood_realn[0] += temp_val * p[to];
                local_llhood_realn[1] += temp_val;
				p_valid_entry = (p_valid_entry -> pnext);
			}
			add_mat(cdx, dv, kk, d, accu_temp_vals);
			add_vec(cdx[fr], dufr, d, -accu_temp_vals);
            if (myparas.bias_enabled)
                for (i = 0; i < kk; i++)
                    bias_grad[i] += -1.0 * accu_temp_vals * exp(p[i]);
			add_mat(X, cdx, kk, d, eta / (double)n);
			if(myparas.bias_enabled)
				add_vec(bias, bias_grad, kk, eta / (double)n);

			//if(is_checkpoint(check_point_counter, find_total_num(kk, comm_sz, myrank), myparas.mpi_num_checkpoints))
			
			//if(is_checkpoint(check_point_counter, find_total_num(kk, comm_sz, myrank), (int)(ceil((double)kk / (double) 20))))
			if(is_checkpoint(check_point_counter, find_total_num(kk, comm_sz, myrank), myparas.mpi_num_checkpoints))
			{
                pack_mat(cdx, comm_vec, kk, d);
                MPI_Reduce(comm_vec, sum_of_reduce, kk * d, MPI_DOUBLE, MPI_SUM, 0, comm);
                //MPI_Reduce(local_llhood_realn, global_llhood_realn, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
                MPI_Allreduce(local_llhood_realn, global_llhood_realn, 2, MPI_DOUBLE, MPI_SUM, comm);
				llhood += global_llhood_realn[0];
				realn += global_llhood_realn[1];
				global_llhood_realn[0] = 0.0;
				global_llhood_realn[1] = 0.0;
                if(myparas.bias_enabled)
                    MPI_Reduce(bias_grad, bias_grad_accum, kk, MPI_DOUBLE, MPI_SUM, 0, comm);
                if(myrank == 0)
                {
                    unpack_mat(sum_of_reduce, cdx, kk, d);
                    add_mat(X, cdx, kk, d, eta / (double)n);
                    memset(sum_of_reduce, 0, kk * d * sizeof(double));
                    if(myparas.bias_enabled)
                    {
                        add_vec(bias, bias_grad_accum, kk, eta / (double)n);
                        memset(bias_grad_accum, 0, kk * sizeof(double));
                    }
                }
                local_llhood_realn[0] = 0.0;
                local_llhood_realn[1] = 0.0;
                for(i = 0; i < kk; i++)
                    memset(cdx[i], 0, d * sizeof(double));
                if(myparas.bias_enabled)
                    memset(bias_grad, 0, kk * sizeof(double));
                //Dispatch the current embedding
                if(myrank == 0)
                    pack_mat(X, comm_vec, kk, d);
                MPI_Bcast(comm_vec, kk * d, MPI_DOUBLE, 0, comm);
                if(myrank != 0)
                    unpack_mat(comm_vec, X, kk, d);
                if(myparas.bias_enabled)
                    MPI_Bcast(bias, kk, MPI_DOUBLE, 0, comm);
			}
		}

		if(realn == 0.0)
			return 0.0;

		if(myrank == 0)
		{
			llhood /= realn;


			if(hparas.verbose)
			{
				printf("The iteration took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
				printf("The iteration took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
				printf("Norms of coordinates: %f\n", frob_norm(X, kk, d));fflush(stdout);
				printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
			}
			if(t > 0)
			{
				if(llhoodprev > llhood)
				{
					eta /= myparas.beta;
					//eta = 0.0;
					if(hparas.verbose)
						printf("Reducing eta to: %f\n", eta);
				}
				else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
				{
					eta *= myparas.alpha;
					if(hparas.verbose)
						printf("Increasing eta to: %f\n", eta);
				}
			}
			last_n_llhoods[llhood_track_index] = llhood;
			double min_llhood = 1.0, max_llhood = -1000.0, total_llhood = 0.0;
			for (i = 0; i < myparas.num_llhood_track; i++) {
				total_llhood += last_n_llhoods[i];
				if (last_n_llhoods[i] < min_llhood)
					min_llhood = last_n_llhoods[i];
				if (last_n_llhoods[i] > max_llhood)
					max_llhood = last_n_llhoods[i];
			}
			if(hparas.verbose)
			{
				printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
				//printf("magnitude of gradient %f\n", frob_norm(cdx + hparas.num_clusters, k, d));
				printf("magnitude of gradient %f\n", frob_norm(cdx, kk, d));
			}

			/*
			   if(embedding_file != NULL)
			   write_embedding_to_file(X, k, d, embeddingfile, 0);
			   */

			if(llhood < REBOOT_THRESHOLD)
				break_signal = 1;
			if (myparas.num_llhood_track > 0
					&& fabs(min_llhood - max_llhood) < myparas.eps 
					&& total_llhood < myparas.num_llhood_track) {
				break_signal = 1;
			}
			else if (myparas.num_llhood_track <= 0
					&& llhood >= llhoodprev
					&& llhood - llhoodprev < myparas.eps 
					&& t > myparas.least_iter)
				break_signal = 1;


			// update llhood tracker index

			llhood_track_index++;
			llhood_track_index %= myparas.num_llhood_track;
		}
        MPI_Bcast(&break_signal, 1, MPI_INT, 0, comm);
	}

	MPI_Bcast(&llhood, 1, MPI_DOUBLE, 0, comm);
	MPI_Bcast(&realn, 1, MPI_DOUBLE, 0, comm);

	Array2Dfree(cdx, kk , d);
	Array2Dfree(delta, kk , d);
	Array2Dfree(dv, kk , d);
	free(p);
	free(tempk);
	Array2Dfree(tempkd, kk, d);
	free(last_n_llhoods);
	free(dufr);
	if(myparas.bias_enabled)
	{
		free(bias_grad);
		free(bias_grad_accum);
	}
	free(comm_vec);
	free(sum_of_reduce);

	if(llhood < REBOOT_THRESHOLD)
	{
		PARAS newparas = myparas;
		newparas.ita /= 2.0;
		if(myrank == 0)
			printf("Need to use smaller initial learning rate %f.\n", newparas.ita);
		return logistic_embed_component_with_comm(newparas, X, bias, hparas, comm, seconds_used);
	}
	else
	{
        *seconds_used = (int)time(NULL) - (int)total_time_counter;
		if(myrank == 0)
			printf("Avg log-likelihood on train: %f, with %d songs. %d seconds elapsed.\n", llhood, hparas.num_subset_songs, *seconds_used);
		return llhood * realn;
	}
}

CEBD lme_by_cluster_MPI(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters)
{
	//MPI related variables
	int myrank;
	int mynewrank;
	int comm_sz;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	//printf("My rank is %d.\n", myrank);
	MPI_Comm lower_comm;
	MPI_Group upper_group;
	MPI_Group lower_group;
	MPI_Comm_group(MPI_COMM_WORLD, &upper_group);
	time_t start_time = time(NULL);
	
	int type = myparas.inter_cluster_transition_type;
	if(type != 0 && type != 1 && type != 2)
	{
		if(myrank == 0)
			printf("MPI code can only handle type 0, 1 or 2.\n");
		exit(1);
	}	

	if(myrank == 0)
		print_playlists_data_stats(pd);


	CEBD ce;
	ce.num_songs = pd.num_songs;
	ce.d = myparas.d;
	ce.inter_cluster_transition_type = myparas.inter_cluster_transition_type;
	int k = pd.num_songs;
	int i;
	int j;
	int d = myparas.d;
	*l = k;
	char error_msg[200];
	char temp_str[1000];
	double precomputation_time = 0.0;
    int precluster_time = 0;
    time_t precluster_start;
	if(clusterfile[0] != '\0')
	{
		if(myrank == 0)
			printf("Using existing clustering file: %s\n", clusterfile);
		FILE* fp = fopen(clusterfile, "r");
		int temp_length;
		read_int_vec(fp, &(ce.assignment), &temp_length);
		if(fgets(temp_str, 1000, fp) != NULL)
		{
			precomputation_time = extract_ending_double(temp_str, "total: ");
			if(myrank == 0)
				printf("Precomputation time is %f seconds.\n", precomputation_time);
		}
		fclose(fp);
		if(temp_length != k)
		{
			if(myrank == 0)
			{
				sprintf(error_msg, "The number of songs in clustering file does not match that in the playlist file (%d vs %d).\n", temp_length, k); 
				error_in_read(error_msg);
			}
		}
		num_clusters = 0;
		for(i = 0; i < k; i++)
		{
			assert(ce.assignment[i] >= 0);
			num_clusters = max(ce.assignment[i], num_clusters);
		}
		num_clusters++;
		ce.num_clusters = num_clusters;
		if(myrank == 0)
			printf("The number of clusters is decided by the file, which is %d.\n", num_clusters);

	}
	else
	{
        precluster_start = time(NULL);
		ce.num_clusters = num_clusters;
		ce.assignment = (int*)calloc(k, sizeof(int));
		//greedy_assign_cluster_with_constraints(ce.assignment, k, num_clusters, pd, 5.0 * (double)k / num_clusters, 0);
		if(myrank == 0 && num_clusters > 1)
			preclustering_with_aggregated_song(pd, myparas, num_clusters, initial_percentage, ce.assignment);
        MPI_Barrier(MPI_COMM_WORLD);
        precluster_time = (int)time(NULL) - (int)precluster_start;
	}

	//Everyone should have the same assignment vector
	MPI_Bcast(ce.assignment, k, MPI_INT, 0, MPI_COMM_WORLD);

	ce.num_songs_in_each_cluster = (int*)malloc(num_clusters * sizeof(int));
	ce.cluster_song_relation = (int**)malloc(num_clusters * sizeof(int*));
	ce.X_by_cluster = (double***)malloc(num_clusters * sizeof(double**));
	if(myparas.bias_enabled)
		ce.bias_by_cluster = (double**)malloc(num_clusters * sizeof(double*));
	else
		ce.bias_by_cluster = NULL;



	HPARAS lower_hparas_array[num_clusters];
	for(i = 0; i < num_clusters; i++)
	{
		lower_hparas_array[i].use_random_init = 1;
		lower_hparas_array[i].num_clusters = num_clusters;
		lower_hparas_array[i].verbose = myparas.verbose;
	}

	int t = 0;
	//when num_nodes > num_clusters
	double llhood_by_cluster[num_clusters];
	//when num_nodes <= num_clusters
	double llhood_by_node[comm_sz];

	int cnm_style = (comm_sz > num_clusters); 
	CNMap cnm;
	NCMap ncm;
	int group_idx;
	int current_cluster_idx;

	int temp_sum = 0;
	for(i = 0; i < num_clusters; i++)
	{
		ce.num_songs_in_each_cluster[i] =  find_cluster_song_list(ce.assignment, k, i, ce.cluster_song_relation + i);
		lower_hparas_array[i].num_subset_songs = ce.num_songs_in_each_cluster[i];
		temp_sum += ce.num_songs_in_each_cluster[i];
		if(myrank == 0)
			printf("Cluster %d has %d songs.\n", i, ce.num_songs_in_each_cluster[i]);

		if(myparas.inter_cluster_transition_type == 0 || myparas.inter_cluster_transition_type == 2)
		{
			ce.X_by_cluster[i] = zerosarray(ce.num_songs_in_each_cluster[i] + num_clusters, d);
			if(myparas.bias_enabled)
				ce.bias_by_cluster[i] = (double*)calloc(ce.num_songs_in_each_cluster[i] + num_clusters, sizeof(double));
		}
		else if(myparas.inter_cluster_transition_type == 1)
		{
			ce.X_by_cluster[i] = zerosarray(ce.num_songs_in_each_cluster[i] + 2 * (num_clusters - 1), d);
			if(myparas.bias_enabled)
				ce.bias_by_cluster[i] = (double*)calloc(ce.num_songs_in_each_cluster[i] + 2 * (num_clusters - 1), sizeof(double));

		}
	}
	assert(temp_sum == pd.num_songs);

	MPI_Barrier(MPI_COMM_WORLD);


	int* nnz_array = (int*)malloc(num_clusters * sizeof(int));
	PHASH* temp_tcount;
	for(i = 0; i < ce.num_clusters; i++)
	{
		temp_tcount = create_cluster_hash(pd, myparas.transition_range, ce.cluster_song_relation[i], ce.num_songs_in_each_cluster[i], ce.assignment, num_clusters, i, myparas.inter_cluster_transition_type);
		nnz_array[i] = temp_tcount -> num_used;
		free_hash(temp_tcount);
	}
	if(cnm_style)
	{
		if(myrank == 0)
			printf("We have more processes than clusters.\n");
		cnm = create_CNmap(num_clusters, comm_sz, ce.num_songs_in_each_cluster, nnz_array);
		group_idx = cnm.node_cluster_map[myrank];
		MPI_Group_incl(upper_group, cnm.num_nodes_for_each_cluster[group_idx], cnm.cluster_node_map[group_idx] ,&lower_group);
		MPI_Comm_create(MPI_COMM_WORLD, lower_group, &lower_comm);
		MPI_Comm_rank(lower_comm, &mynewrank);
	}
	else
	{
		if(myrank == 0)
			printf("We have more clusters than processes.\n");
		ncm = create_NCmap(num_clusters, comm_sz, ce.num_songs_in_each_cluster, nnz_array);
	}
	free(nnz_array);
	MPI_Barrier(MPI_COMM_WORLD);

	//double** transitions_to_each_cluster = zerosarray(k, num_clusters);


	double* comm_vec = (double*)malloc((k + 2 * (num_clusters - 1)) * d * sizeof(double));


	if(cnm_style)
	{
		lower_hparas_array[group_idx].tcount = create_cluster_hash(pd, myparas.transition_range, ce.cluster_song_relation[group_idx], ce.num_songs_in_each_cluster[group_idx], ce.assignment, num_clusters, group_idx, myparas.inter_cluster_transition_type);
	}
	else
	{
		for(i = 0; i < ncm.num_clusters_on_each_node[myrank]; i++)
		{
			current_cluster_idx = ncm.node_cluster_map[myrank][i];
			lower_hparas_array[current_cluster_idx].tcount = create_cluster_hash(pd, myparas.transition_range, ce.cluster_song_relation[current_cluster_idx], ce.num_songs_in_each_cluster[current_cluster_idx], ce.assignment, num_clusters, current_cluster_idx, myparas.inter_cluster_transition_type);
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
		printf("Intra-cluster transition percentage: %f%%.\n", compute_percentage(count_intra_cluster_transitions(ce.assignment, k, num_clusters, pd), pd.num_transitions));

	MPI_Barrier(MPI_COMM_WORLD);
	//Build Transition to cluster matrix
	/*
	if(myrank == 0)
		printf("Building song-cluster transition table....\n");
	for(i = 0; i < num_clusters; i++)
	{
		for(j = 0; j < ce.num_songs_in_each_cluster[i]; j++)
		{
			find_transitions_to_each_cluster(j, lower_hparas_array[i].tcount, transitions_to_each_cluster[ce.cluster_song_relation[i][j]], num_clusters, i, myparas.inter_cluster_transition_type);

		}
	}
	if(myrank == 0)
		printf("Done.\n");

	double intra_count = 0.0;
	for(i = 0; i < k; i++)
		intra_count += transitions_to_each_cluster[i][ce.assignment[i]];
	if(myrank == 0)
		printf("Intra-cluster transition percentage: %f%%.\n", compute_percentage((int)intra_count, pd.num_transitions));
		*/

	MPI_Barrier(MPI_COMM_WORLD);
    int component_time_array[num_clusters];
	if(comm_sz > num_clusters)
	{

		if(mynewrank == 0)
			printf("Lower-level embedding for cluster %d starts with %d songs and %d processes working on it...\n", group_idx, ce.num_songs_in_each_cluster[group_idx], cnm.num_nodes_for_each_cluster[group_idx]);
		if(myparas.bias_enabled)
			llhood_by_cluster[group_idx] = logistic_embed_component_with_comm(myparas, ce.X_by_cluster[group_idx], ce.bias_by_cluster[group_idx], lower_hparas_array[group_idx], lower_comm, component_time_array + group_idx);
		else
			llhood_by_cluster[group_idx] = logistic_embed_component_with_comm(myparas, ce.X_by_cluster[group_idx], NULL, lower_hparas_array[group_idx], lower_comm, component_time_array + group_idx);

		if(mynewrank == 0)
			printf("Lower-level embedding for cluster %d with %d songs is done. %d processes contributed.\n", group_idx, ce.num_songs_in_each_cluster[group_idx], cnm.num_nodes_for_each_cluster[group_idx]);

		//for(i = 0; i < num_clusters; i++)
		//	free_hash(lower_hparas_array[i].tcount);
		free_hash(lower_hparas_array[group_idx].tcount);

		if(myrank == 0)
		{
			for(i = 1; i < num_clusters; i++)
			{
				int temp_int;
				if(myparas.inter_cluster_transition_type == 0 || myparas.inter_cluster_transition_type == 2)
					temp_int = ce.num_songs_in_each_cluster[i] + num_clusters;
				else if(myparas.inter_cluster_transition_type == 1)
					temp_int = ce.num_songs_in_each_cluster[i] + 2 * (num_clusters - 1);
				MPI_Recv(comm_vec, temp_int * d, MPI_DOUBLE, cnm.cluster_node_map[i][0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				unpack_mat(comm_vec, ce.X_by_cluster[i], temp_int, d);
				MPI_Recv(llhood_by_cluster + i, 1, MPI_DOUBLE, cnm.cluster_node_map[i][0], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(component_time_array, 1, MPI_INT, cnm.cluster_node_map[i][0], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if(myparas.bias_enabled)
				{
					MPI_Recv(ce.bias_by_cluster[i], temp_int, MPI_DOUBLE, cnm.cluster_node_map[i][0], 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}
		else
		{
			if(mynewrank == 0)
			{
				int temp_int;
				if(myparas.inter_cluster_transition_type == 0 || myparas.inter_cluster_transition_type == 2)
					temp_int = ce.num_songs_in_each_cluster[group_idx] + num_clusters;
				else if(myparas.inter_cluster_transition_type == 1)
					temp_int = ce.num_songs_in_each_cluster[group_idx] + 2 * (num_clusters - 1);
				pack_mat(ce.X_by_cluster[group_idx], comm_vec, temp_int, d);
				MPI_Send(comm_vec, temp_int * d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
				MPI_Send(llhood_by_cluster + group_idx, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
				MPI_Send(component_time_array + group_idx, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
				//print_vec(ce.bias_by_cluster[group_idx], temp_int); 
				if(myparas.bias_enabled)
					MPI_Send(ce.bias_by_cluster[group_idx], temp_int, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
			}
		}

		//for(i = 0; i < num_clusters; i++)
		//{
		//	int temp_int = ce.num_songs_in_each_cluster[i] + num_clusters;
		//	//Sender
		//	if(mynewrank == 0 && group_idx == i)
		//	{
		//		pack_mat(ce.X_by_cluster[i], comm_vec, temp_int, d);
		//	}
		//	MPI_Bcast(comm_vec, temp_int * d, MPI_DOUBLE, cnm.cluster_node_map[i][0], MPI_COMM_WORLD);
		//	MPI_Bcast(llhood_by_cluster + i, 1, MPI_DOUBLE, cnm.cluster_node_map[i][0], MPI_COMM_WORLD);
		//	if(myparas.bias_enabled)
		//		MPI_Bcast(ce.bias_by_cluster[i], temp_int, MPI_DOUBLE, cnm.cluster_node_map[i][0], MPI_COMM_WORLD);
		//	if(!(mynewrank == 0 && group_idx == i))
		//	{
		//		unpack_mat(comm_vec, ce.X_by_cluster[i], temp_int, d);
		//	}
		//}



		if(myrank == 0)
		{
			printf("All communication done.\n");
			printf("=====================================================\n");
			printf("Avg. training log-likelihood: %f.\n", sum_vec(llhood_by_cluster, num_clusters) / (double) pd.num_transitions);
			printf("=====================================================\n");
		}

		free_CNMap(cnm);
	}
	else
	{
		llhood_by_node[myrank] = 0.0;
		for(i = 0; i < ncm.num_clusters_on_each_node[myrank]; i++)
		{
			current_cluster_idx = ncm.node_cluster_map[myrank][i];
			printf("Lower-level embedding for cluster %d starts with %d songs and process %d working on it...\n", current_cluster_idx, ce.num_songs_in_each_cluster[current_cluster_idx], myrank);
			if(!myparas.do_ps)
				llhood_by_node[myrank] += logistic_embed_component(myparas, ce.X_by_cluster[current_cluster_idx], myparas.bias_enabled? ce.bias_by_cluster[current_cluster_idx] : NULL, lower_hparas_array[current_cluster_idx], component_time_array + current_cluster_idx);
			else
				llhood_by_node[myrank] += logistic_embed_component_ps(myparas, ce.X_by_cluster[current_cluster_idx], myparas.bias_enabled? ce.bias_by_cluster[current_cluster_idx] : NULL, lower_hparas_array[current_cluster_idx], component_time_array + current_cluster_idx);

			printf("Lower-level embedding for cluster %d with %d songs is done. process %d contributed.\n", current_cluster_idx, ce.num_songs_in_each_cluster[current_cluster_idx], myrank);
			free_hash(lower_hparas_array[current_cluster_idx].tcount);
		}

		//for(i = 0; i < num_clusters; i++)
			//free_hash(lower_hparas_array[i].tcount);

		//printf("Right before communication.\n");
		if(myrank == 0)
		{
			for(i = 1; i < comm_sz; i++)
			{
				for(j = 0; j < ncm.num_clusters_on_each_node[i]; j++)
				{
					int temp_int;
					int current_cluster_idx = ncm.node_cluster_map[i][j];
					if(myparas.inter_cluster_transition_type == 0 || myparas.inter_cluster_transition_type == 2)
						temp_int = ce.num_songs_in_each_cluster[current_cluster_idx] + num_clusters;
					else if(myparas.inter_cluster_transition_type == 1)
						temp_int = ce.num_songs_in_each_cluster[current_cluster_idx] + 2 * (num_clusters - 1);

					MPI_Recv(comm_vec, temp_int * d, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					unpack_mat(comm_vec, ce.X_by_cluster[current_cluster_idx], temp_int, d);
                    MPI_Recv(component_time_array + current_cluster_idx, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if(myparas.bias_enabled)
						MPI_Recv(ce.bias_by_cluster[current_cluster_idx], temp_int, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				MPI_Recv(llhood_by_node + i, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		else
		{
			for(j = 0; j < ncm.num_clusters_on_each_node[myrank]; j++)
			{
				int temp_int;
				int current_cluster_idx = ncm.node_cluster_map[myrank][j];
				if(myparas.inter_cluster_transition_type == 0 || myparas.inter_cluster_transition_type == 2)
					temp_int = ce.num_songs_in_each_cluster[current_cluster_idx] + num_clusters;
				else if(myparas.inter_cluster_transition_type == 1)
					temp_int = ce.num_songs_in_each_cluster[current_cluster_idx] + 2 * (num_clusters - 1);
				pack_mat(ce.X_by_cluster[current_cluster_idx], comm_vec, temp_int, d);
				MPI_Send(comm_vec, temp_int * d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(component_time_array + current_cluster_idx, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
				if(myparas.bias_enabled)
					MPI_Send(ce.bias_by_cluster[current_cluster_idx], temp_int, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
			}
			MPI_Send(llhood_by_node + myrank, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		}

		if(myrank == 0)
		{
			printf("All communication done.\n");
			printf("=====================================================\n");
			printf("Avg. training log-likelihood: %f.\n", sum_vec(llhood_by_node, comm_sz) / (double) pd.num_transitions);
			printf("=====================================================\n");
		}
		free_NCMap(ncm);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(myrank == 0)
	{
		int lme_time = (int)time(NULL) - (int)start_time;
		if(precomputation_time > 0.0)
			printf("Finished. It took %d seconds (%d seconds if precomputation included).\n", lme_time, lme_time + (int) precomputation_time);
		else
			printf("Finished. It took %d seconds.\n", lme_time);
        printf("Time statistics:\n");
        printf("Preclustering: %d seconds\n", precluster_time);
        printf("Sum of time spent on all LME components: %d\n", sum_int_vec(component_time_array, num_clusters));
        printf("Mean of time spent on all LME components: %d\n", (int)((double) sum_int_vec(component_time_array, num_clusters) /(double) num_clusters));
        printf("Max of time spent on all LME components: %d\n", extreme_int_vec(component_time_array, num_clusters, 1));
        printf("Writing to file %s\n", embedding_file);
        write_ClusterEmbedding_to_file(ce, embedding_file);
        printf("Done.\n");
	}

	//Array2Dfree(transitions_to_each_cluster, k, num_clusters);

	if(cnm_style)
	{
		MPI_Comm_free(&lower_comm);
		MPI_Group_free(&lower_group);
	}
	free(comm_vec);
	return ce;
}
#endif
//int reached_checkpoint(int rank, int size, int idx, int k, int num_checkpoints);
//{
//	assert(num_checkpoints >= k / size);
//}

//Old stuffs
//
//
//void find_trans_cluster_log_prob(double** hubs, int num_clusters, int d, int current_cluster_idx, double* p)
//{
//	int i;
//	int j;
//	double* tempk = (double*)malloc(num_clusters * sizeof(double));
//	double** delta = zerosarray(num_clusters, d);
//	double** tempkd = zerosarray(num_clusters, d);
//	for(j = 0; j < num_clusters; j++)
//		for(i = 0; i < d; i++)
//			delta[j][i] = hubs[current_cluster_idx][i] - hubs[j][i];
//	mat_mult(delta, delta, tempkd, num_clusters, d);
//	scale_mat(tempkd, num_clusters, d, -1.0);
//	sum_along_direct(tempkd, p, num_clusters, d, 1);
//	norm_wo_underflow(p, num_clusters, tempk);
//	Array2Dfree(delta, num_clusters, d);
//	Array2Dfree(tempkd, num_clusters, d);
//	free(tempk);
//}
//
//double** hierachical_logistic_embed(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters)
//{
//	int i;
//	int j;
//	int k = pd.num_songs;
//	int d = myparas.d;
//	*l = k;
//
//	double** X = randarray(k, d, 1.0);
//	int* current_assignment = (int*)malloc(k * sizeof(int));
//	int* last_assignment = (int*)malloc(k * sizeof(int));
//
//	HPARAS upper_hparas;
//	upper_hparas.use_random_init = 0;
//	upper_hparas.num_subset_songs = num_clusters;
//	upper_hparas.verbose = 0;
//
//	HPARAS lower_hparas_array[num_clusters];
//	for(i = 0; i < num_clusters; i++)
//	{
//		lower_hparas_array[i].use_random_init = 0;
//		lower_hparas_array[i].verbose = 0;
//	}
//
//	int* cluster_song_relation[num_clusters];
//	int num_songs_in_each_cluster[num_clusters];
//	double* local_counts_array[num_clusters];
//	double** X_decomposed[num_clusters];
//
//	double** upper_level_hubs;
//
//
//	kmeans(X, k, d, num_clusters, 1e-10, 0, current_assignment, &upper_level_hubs);
//
//	/*
//	   for(i = 0; i < k; i++)
//	   printf("%d\n", current_assignment[i]);
//	   */
//
//	int t = 0;
//	while(++t)
//	{
//		printf("Big iteration %d...\n", t);
//		memcpy(last_assignment, current_assignment, k * sizeof(int));
//
//
//		//Init hierachical parameters for the two levels.
//		upper_hparas.tcount = create_upper_hash(pd, myparas, num_clusters, current_assignment);
//		for(i = 0; i < num_clusters; i++)
//		{
//			num_songs_in_each_cluster[i] =  find_cluster_song_list(current_assignment, k, i, cluster_song_relation + i);
//			lower_hparas_array[i].num_subset_songs = num_songs_in_each_cluster[i];
//			/*
//			   if(num_songs_in_each_cluster[i] == 0)
//			   {
//			   printf("Cluster %d is empty.\n", i);
//			   exit(1);
//			   }
//			   */
//			printf("Cluster %d has %d songs.\n", i, num_songs_in_each_cluster[i]);
//			local_counts_array[i] = (double*)malloc(num_songs_in_each_cluster[i] * sizeof(double));
//			lower_hparas_array[i].tcount = create_lower_hash(pd, myparas, cluster_song_relation[i], num_songs_in_each_cluster[i], local_counts_array[i]);
//		}
//
//		/*
//		   for(i = 0; i < num_clusters; i++)
//		   for(j = 0; j < num_songs_in_each_cluster[i];j++)
//		   printf("%d\n", cluster_song_relation[i][j]);
//		   */
//		decompose_X(X, X_decomposed, d, k, num_clusters, cluster_song_relation, num_songs_in_each_cluster);
//		//printf("OK till here.\n");fflush(stdout);
//
//		logistic_embed_component(myparas, upper_level_hubs, NULL,  upper_hparas);
//		scale_mat(upper_level_hubs, num_clusters, d, 1e2);
//		printf("Upper-level embedding is done.\n");
//		for(i = 0; i < num_clusters; i++)
//		{
//			logistic_embed_component(myparas, X_decomposed[i], NULL, lower_hparas_array[i]);
//			printf("Lower-level embedding for cluster %d is done.\n", i);
//		}
//
//		recompose_X(X, X_decomposed, d, k, num_clusters, cluster_song_relation, num_songs_in_each_cluster, local_counts_array, upper_level_hubs);
//		assign_clusters(X, upper_level_hubs, k, d, num_clusters, current_assignment);
//
//		free_hash(upper_hparas.tcount);
//		for(i = 0; i < num_clusters; i++)
//		{
//			free(local_counts_array[i]);
//			free(cluster_song_relation[i]);
//			free_hash(lower_hparas_array[i].tcount);
//		}
//		int temp_hamming_dist = hamming_dist(current_assignment, last_assignment, k);
//		printf("Hamming distance with assignment vector of last iteration is %d.\n", temp_hamming_dist);
//
//		if(embedding_file != NULL)
//			write_embedding_to_file(X, k, d, embeddingfile, 0);
//
//		if(temp_hamming_dist == 0)
//			break;
//	}
//
//	if(embedding_file != NULL)
//		write_embedding_to_file(X, k, d, embeddingfile, 0);
//	free(current_assignment);
//	free(last_assignment);
//	Array2Dfree(upper_level_hubs, num_clusters, d);
//	return X;
//}
//
//double** hierachical_logistic_embed_fix_framework(PDATA pd, PARAS myparas, char* embedding_file, int* l, int num_clusters)
//{
//	int i;
//	int j;
//	int k = pd.num_songs;
//	int d = myparas.d;
//	*l = k;
//
//	double** X = randarray(k, d, 1.0);
//	int* current_assignment = (int*)malloc(k * sizeof(int));
//	int* last_assignment = (int*)malloc(k * sizeof(int));
//
//
//	HPARAS lower_hparas_array[num_clusters];
//	for(i = 0; i < num_clusters; i++)
//	{
//		lower_hparas_array[i].use_random_init = 0;
//		lower_hparas_array[i].num_clusters = num_clusters;
//		lower_hparas_array[i].verbose = 0;
//	}
//
//	int* cluster_song_relation[num_clusters];
//	int num_songs_in_each_cluster[num_clusters];
//	double** X_decomposed[num_clusters];
//
//	double** upper_level_hubs;
//
//
//	kmeans(X, k, d, num_clusters, 1e-10, 0, current_assignment, &upper_level_hubs);
//
//	int t = 0;
//	while(++t)
//	{
//		printf("Big iteration %d...\n", t);
//		memcpy(last_assignment, current_assignment, k * sizeof(int));
//		int temp_sum = 0;
//		for(i = 0; i < num_clusters; i++)
//		{
//			num_songs_in_each_cluster[i] =  find_cluster_song_list(current_assignment, k, i, cluster_song_relation + i);
//			lower_hparas_array[i].num_subset_songs = num_songs_in_each_cluster[i];
//			printf("Cluster %d has %d songs.\n", i, num_songs_in_each_cluster[i]);
//			temp_sum += num_songs_in_each_cluster[i];
//			lower_hparas_array[i].tcount = create_lower_hash_fix_framework(pd, myparas, cluster_song_relation[i], num_songs_in_each_cluster[i], current_assignment, num_clusters, i);
//			X_decomposed[i] = zerosarray(num_songs_in_each_cluster[i] + num_clusters, d);
//			Array2Dcopy(upper_level_hubs, X_decomposed[i], num_clusters, d);
//			//printf("Higher. Norm of the hubs: %f.\n", frob_norm(upper_level_hubs, num_clusters, d));
//			//printf("Higher2. Norm of the hubs: %f.\n", frob_norm(X_decomposed[i], num_clusters, d));
//			for(j = 0; j < num_songs_in_each_cluster[i]; j++)
//				memcpy(X_decomposed[i][j + num_clusters], X[cluster_song_relation[i][j]], d * sizeof(double));
//		}
//		printf("Total number of songs by summing: %d!!!\n", temp_sum);
//
//		for(i = 0; i < num_clusters; i++)
//		{
//			logistic_embed_component(myparas, X_decomposed[i], NULL, lower_hparas_array[i]);
//			printf("Lower-level embedding for cluster %d is done.\n", i);
//
//			//Sanity check: whether the positions of hub points have beeen moved or not
//			/*
//			   int a;
//			   int b;
//			   for(a = 0; a < num_clusters; a++)
//			   for(b = 0; b < d; b++)
//			   assert(upper_level_hubs[a][b] == X_decomposed[i][a][b]);
//			   printf("All assertion passed.\n");
//			   */
//		}
//
//		for(i = 0; i < num_clusters; i++)
//		{
//			for(j = 0; j < num_songs_in_each_cluster[i]; j++)
//			{
//				memcpy(X[cluster_song_relation[i][j]], X_decomposed[i][j + num_clusters],  d * sizeof(double));
//			}
//		}
//
//		//Updated the hub vectors
//		//group_means(X, current_assignment, upper_level_hubs, k, d, num_clusters);
//		double* local_transition_array;
//		int intra_cluster_transition_count = 0;
//		int ss;
//		int hs;
//		int hh;
//		int total;
//		for(i = 0; i < num_clusters; i++)
//		{
//			printf("Recalculating hub point for cluster %d...\n", i);
//			local_transition_array = (double*)calloc(num_songs_in_each_cluster[i], sizeof(double));
//			count_local_hub_song_transitions(lower_hparas_array[i].tcount, num_clusters, i, local_transition_array);
//			//print_vec(local_transition_array, num_songs_in_each_cluster[i]);
//			find_hub(X_decomposed[i], upper_level_hubs[i], num_songs_in_each_cluster[i], d, local_transition_array, 1.0, 1e-5, 0);
//			printf("Done.\n");
//			free(local_transition_array);
//			Array2Dfree(X_decomposed[i], num_songs_in_each_cluster[i] + num_clusters, d);
//
//			compute_transition_stats(lower_hparas_array[i].tcount, num_clusters, &ss, &hs, &hh, &total);
//			assert(total == (ss + hs + hh));
//			printf("Transition breakdowns: (ss: %f%%, hs: %f%%, hh: %f%%.)\n", compute_percentage(ss, total), compute_percentage(hs, total), compute_percentage(hh, total));
//			intra_cluster_transition_count += ss;
//			//intra_cluster_transition_count += count_local_song_song_transtions(lower_hparas_array[i].tcount, num_clusters);
//		}
//		printf("The percentage of intra-cluster transitions: %f%%.\n", 100.0 * (double) intra_cluster_transition_count / (double) pd.num_transitions);
//
//		assign_clusters(X, upper_level_hubs, k, d, num_clusters, current_assignment);
//
//		for(i = 0; i < num_clusters; i++)
//		{
//			free(cluster_song_relation[i]);
//			free_hash(lower_hparas_array[i].tcount);
//		}
//		int temp_hamming_dist = hamming_dist(current_assignment, last_assignment, k);
//		printf("Hamming distance with assignment vector of last iteration is %d.\n", temp_hamming_dist);
//
//		if(embedding_file != NULL)
//			write_embedding_to_file(X, k, d, embeddingfile, 0);
//
//		if(temp_hamming_dist == 0)
//			break;
//
//
//	}
//
//	if(embedding_file != NULL)
//		write_embedding_to_file(X, k, d, embeddingfile, 0);
//	free(current_assignment);
//	free(last_assignment);
//	Array2Dfree(upper_level_hubs, num_clusters, d);
//	return X;
//}
//
//void logistic_embed_component_fixed_framework(PARAS myparas,  double** X, HPARAS hparas)
//{
//	int n; //Number of training transitions
//	int i;
//	int j;
//	//int s;
//	//int q;
//	int t;
//	int fr;
//	int to;
//	int d = myparas.d;
//	int k = hparas.num_subset_songs;
//	int kk = hparas.num_subset_songs + hparas.num_clusters;
//	if(k <= 1)
//		return;
//	double eta;
//	int idx;
//	if (myparas.ita > 0)
//		eta = myparas.ita;
//	else
//		eta = k;
//	n = 0;
//	for(i = 0; i < ((hparas.tcount) -> length); i++)
//		if(!is_null_entry(((hparas.tcount) -> p)[i].key))
//			n += (int)((hparas.tcount) -> p)[i].val;
//	/*
//	   for(i = 0; i < pd.num_playlists; i ++)
//	   if(pd.playlists_length[i] > 0)
//	   n += pd.playlists_length[i] - 1;
//	   */
//	if(hparas.verbose)
//		printf("Altogether %d transitions.\n", n);fflush(stdout);
//
//	if(hparas.use_random_init)
//		randfill(X + hparas.num_clusters, k, d, 1.0);
//
//	double llhood = 0.0;
//	double llhoodprev;
//	double realn;
//
//	double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
//	int llhood_track_index = 0;
//
//	for (i = 0; i < myparas.num_llhood_track; i++)
//		last_n_llhoods[i] = 1.0;
//
//	time_t start_time;
//	clock_t start_clock, temp_clock;
//
//	double** cdx = zerosarray(kk, d);
//	double** delta = zerosarray(kk, d);
//	double** dv = zerosarray(kk, d);
//	double* p = (double*)calloc(kk, sizeof(double));
//	double* tempk = (double*)calloc(kk, sizeof(double));
//	double** tempkd = zerosarray(kk, d);
//	double* dufr = (double*)calloc(d, sizeof(double));
//	double temp_val;
//	HELEM* p_valid_entry;
//
//	//printf("Before. Norm of the hubs: %f.\n", frob_norm(X, hparas.num_clusters, d));
//
//	for(t = 0; t < hparas.num_subset_songs / 2; t++)
//	{
//		start_time = time(NULL);
//		start_clock = clock();
//		if(hparas.verbose)
//			printf("Little Iteration %d\n", t + 1);fflush(stdout);
//		llhoodprev = llhood;
//		llhood = 0.0;
//		realn = 0.0;
//		//for(s = 0; s < hparas.num_subset_songs; s++)
//		for(fr = 0; fr < hparas.num_subset_songs; fr++)
//		{
//			//fr = hparas.idx_vec[s];
//			for(i = 0; i < kk; i++)
//				memset(cdx[i], 0, d * sizeof(double));
//
//			for(j = 0; j < kk; j++)
//				for(i = 0; i < d; i++)
//					delta[j][i] = X[fr][i] - X[j][i];
//
//			mat_mult(delta, delta, tempkd, kk, d);
//			scale_mat(tempkd, kk, d, -1.0);
//			sum_along_direct(tempkd, p, kk, d, 1);
//			norm_wo_underflow(p, kk, tempk);
//			for(j = 0; j < kk; j++)
//				for(i = 0; i < d; i++)
//					dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
//			sum_along_direct(dv, dufr, kk, d, 0);
//
//			double accu_temp_vals = 0.0;
//			p_valid_entry = (hparas.tcount) -> p_first_tran[fr];
//			while(p_valid_entry != NULL)
//			{
//				to = (p_valid_entry -> key.to);
//				//q = find_in_list(to, hparas.idx_vec, hparas.num_subset_songs); 
//				temp_val = p_valid_entry -> val;
//				accu_temp_vals += temp_val;
//				if(to >= hparas.num_clusters)
//					add_vec(cdx[to], delta[to], d, 2.0 * temp_val);
//				/*
//				   Veccopy(dufr, tempd, d);
//				   add_vec(tempd, delta[to], d, 2.0);
//				   scale_vec(tempd, d, -1.0);
//				   add_vec(cdx[fr], tempd, d, temp_val);
//				   */
//				if(fr >= hparas.num_clusters)
//					add_vec(cdx[fr], delta[to], d, -2.0 * temp_val);
//				llhood += temp_val * p[to];
//				realn += temp_val;
//				p_valid_entry = (p_valid_entry -> pnext);
//			}
//			add_mat(cdx + hparas.num_clusters, dv + hparas.num_clusters, k, d, accu_temp_vals);
//			if(fr >= hparas.num_clusters)
//				add_vec(cdx[fr], dufr, d, -accu_temp_vals);
//			add_mat(X + hparas.num_clusters, cdx + hparas.num_clusters, k, d, eta / (double)n);
//
//			//write_embedding_to_file(X, kk, d, embeddingfile, 0);
//		}
//
//		llhood /= realn;
//
//
//		if(hparas.verbose)
//		{
//			printf("The iteration took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
//			printf("The iteration took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
//			printf("Norms of coordinates: %f\n", frob_norm(X, k, d));fflush(stdout);
//			printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
//		}
//		if(t > 0)
//		{
//			if(llhoodprev > llhood)
//			{
//				eta /= myparas.beta;
//				//eta = 0.0;
//				if(hparas.verbose)
//					printf("Reducing eta to: %f\n", eta);
//			}
//			else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
//			{
//				eta *= myparas.alpha;
//				if(hparas.verbose)
//					printf("Increasing eta to: %f\n", eta);
//			}
//		}
//		last_n_llhoods[llhood_track_index] = llhood;
//		double min_llhood = 1.0, max_llhood = -1000.0, total_llhood = 0.0;
//		for (i = 0; i < myparas.num_llhood_track; i++) {
//			total_llhood += last_n_llhoods[i];
//			if (last_n_llhoods[i] < min_llhood)
//				min_llhood = last_n_llhoods[i];
//			if (last_n_llhoods[i] > max_llhood)
//				max_llhood = last_n_llhoods[i];
//		}
//		if(hparas.verbose)
//		{
//			printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
//			printf("magnitude of gradient %f\n", frob_norm(cdx + hparas.num_clusters, k, d));
//		}
//
//		/*
//		   if(embedding_file != NULL)
//		   write_embedding_to_file(X, k, d, embeddingfile, 0);
//		   */
//
//		if (myparas.num_llhood_track > 0
//				&& fabs(min_llhood - max_llhood) < myparas.eps 
//				&& total_llhood < myparas.num_llhood_track) {
//			break;
//		}
//		else if (myparas.num_llhood_track <= 0
//				&& llhood >= llhoodprev
//				&& llhood - llhoodprev < myparas.eps 
//				&& t > myparas.least_iter)
//			break;
//
//
//		// update llhood tracker index
//
//		llhood_track_index++;
//		llhood_track_index %= myparas.num_llhood_track;
//	}
//
//	//printf("Done. Norm of the hubs: %f.\n", frob_norm(X, hparas.num_clusters, d));
//	Array2Dfree(cdx, kk , d);
//	Array2Dfree(delta, kk , d);
//	Array2Dfree(dv, kk , d);
//	free(p);
//	free(tempk);
//	Array2Dfree(tempkd, kk, d);
//	free(last_n_llhoods);
//	free(dufr);
//}
//
//PHASH* create_hash_transition_pair_table(PDATA pd, PARAS myparas, int shift)
//{
//	int i;
//	int j;
//	int t;
//	int idx;
//	PHASH* tcount;
//	//double** tcount_full;
//
//	HELEM temp_elem;
//	TPAIR temp_pair;
//	HELEM* p_valid_entry;
//
//	tcount = create_empty_hash(2 * myparas.transition_range  * pd.num_appearance, pd.num_songs + shift);
//
//	for(i = 0; i < pd.num_playlists; i ++)
//	{
//		if(pd.playlists_length[i] > 1)
//		{
//			for(j = 0; j < pd.playlists_length[i] - 1; j++)
//			{
//				for(t = 1; t <= myparas.transition_range; t++)
//				{
//					if(j + t < pd.playlists_length[i])
//					{
//						temp_pair.fr = pd.playlists[i][j] + shift;
//						temp_pair.to = pd.playlists[i][j + t] + shift;
//						if(temp_pair.fr >= shift && temp_pair.to >= shift)
//						{
//							idx = exist_in_hash(tcount, temp_pair);
//							if(idx < 0)
//							{
//								temp_elem.key = temp_pair;
//								temp_elem.val = 1.0 / (double) t;
//								add_entry(tcount, temp_elem);
//							}
//							else
//								update_with(tcount, idx, 1.0 / (double) t);
//						}
//					}
//				}
//			}
//		}
//	}
//
//	build_same_song_index(tcount);
//	return tcount;
//}
//
//
//
//PHASH* create_upper_hash(PDATA pd, PARAS myparas, int num_clusters, int* member_vec)
//{
//	int i;
//	int j;
//	int t;
//	int idx;
//	PHASH* tcount;
//	//double** tcount_full;
//
//	HELEM temp_elem;
//	TPAIR temp_pair;
//	HELEM* p_valid_entry;
//
//	tcount = create_empty_hash(2 * myparas.transition_range  * pd.num_appearance, num_clusters);
//
//	for(i = 0; i < pd.num_playlists; i ++)
//	{
//		if(pd.playlists_length[i] > 1)
//		{
//			for(j = 0; j < pd.playlists_length[i] - 1; j++)
//			{
//				for(t = 1; t <= myparas.transition_range; t++)
//				{
//					if(j + t < pd.playlists_length[i])
//					{
//						temp_pair.fr = member_vec[pd.playlists[i][j]];
//						temp_pair.to = member_vec[pd.playlists[i][j + t]];
//						if(temp_pair.fr >= 0 && temp_pair.to >= 0)
//						{
//							idx = exist_in_hash(tcount, temp_pair);
//							if(idx < 0)
//							{
//								temp_elem.key = temp_pair;
//								temp_elem.val = 1.0 / (double) t;
//								add_entry(tcount, temp_elem);
//							}
//							else
//								update_with(tcount, idx, 1.0 / (double) t);
//						}
//					}
//				}
//			}
//		}
//	}
//	build_same_song_index(tcount);
//	/*
//	   int temp_int = 0;
//	   for(i = 0; i < tcount -> length; i++)
//	   {
//	   if(!is_null_entry((tcount -> p)[i].key))
//	   temp_int++;
//	   } 
//	   printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(num_clusters * num_clusters)));
//	   printf("Transition matrix initialized.\n");
//	   */
//	return tcount;
//}
//
//PHASH* create_lower_hash(PDATA pd, PARAS myparas, int* cluster_idx_list, int cluster_song_count, double* local_hub_trans_count)
//{
//	int i;
//	int j;
//	int t;
//	int idx;
//	PHASH* tcount;
//	//double** tcount_full;
//
//	HELEM temp_elem;
//	TPAIR temp_pair;
//	HELEM* p_valid_entry;
//
//	memset(local_hub_trans_count, 0, cluster_song_count * sizeof(double));
//
//	tcount = create_empty_hash(2 * myparas.transition_range  * pd.num_appearance, pd.num_songs);
//
//	for(i = 0; i < pd.num_playlists; i ++)
//	{
//		if(pd.playlists_length[i] > 1)
//		{
//			for(j = 0; j < pd.playlists_length[i] - 1; j++)
//			{
//				for(t = 1; t <= myparas.transition_range; t++)
//				{
//					if(j + t < pd.playlists_length[i])
//					{
//						temp_pair.fr = find_in_sorted_list(pd.playlists[i][j], cluster_idx_list, cluster_song_count, 0);
//						temp_pair.to = find_in_sorted_list(pd.playlists[i][j + t], cluster_idx_list, cluster_song_count, 0);
//
//						if(temp_pair.fr >= 0 && temp_pair.to >= 0)
//						{
//							idx = exist_in_hash(tcount, temp_pair);
//							if(idx < 0)
//							{
//								temp_elem.key = temp_pair;
//								temp_elem.val = 1.0 / (double) t;
//								add_entry(tcount, temp_elem);
//							}
//							else
//								update_with(tcount, idx, 1.0 / (double) t);
//						}
//						else if(temp_pair.fr < 0 && temp_pair.to >= 0)
//							local_hub_trans_count[temp_pair.to] += 1.0 / (double) t;
//					}
//				}
//			}
//		}
//	}
//
//	build_same_song_index(tcount);
//	return tcount;
//}
//
//PHASH* create_lower_hash_fix_framework(PDATA pd, PARAS myparas, int* cluster_idx_list, int cluster_song_count, int* member_vec, int num_clusters, int current_cluster_id)
//{
//	int i;
//	int j;
//	int t;
//	int idx;
//	int fr;
//	int to;
//	PHASH* tcount;
//	//double** tcount_full;
//
//	HELEM temp_elem;
//	TPAIR temp_pair;
//
//	//memset(local_hub_trans_count, 0, cluster_song_count * sizeof(double));
//
//	tcount = create_empty_hash(2 * myparas.transition_range  * pd.num_appearance, cluster_song_count + num_clusters);
//
//	for(i = 0; i < pd.num_playlists; i ++)
//	{
//		if(pd.playlists_length[i] > 1)
//		{
//			for(j = 0; j < pd.playlists_length[i] - 1; j++)
//			{
//				for(t = 1; t <= myparas.transition_range; t++)
//				{
//					if(j + t < pd.playlists_length[i])
//					{
//						fr = pd.playlists[i][j];
//						to = pd.playlists[i][j + t];
//						if(fr >= 0 && to >= 0)
//						{
//							if(member_vec[fr] == current_cluster_id && member_vec[to] == current_cluster_id)
//							{
//								temp_pair.fr = find_in_sorted_list(fr, cluster_idx_list, cluster_song_count, 0) + num_clusters;
//								temp_pair.to = find_in_sorted_list(to, cluster_idx_list, cluster_song_count, 0) + num_clusters;
//								one_more_count(tcount, temp_pair, 1.0 / (double) t);
//
//							}
//							else if(member_vec[fr] != current_cluster_id && member_vec[to] == current_cluster_id)
//							{
//								temp_pair.fr = member_vec[fr];
//								temp_pair.to = current_cluster_id;
//								one_more_count(tcount, temp_pair, 1.0 / (double) t);
//								temp_pair.fr = current_cluster_id; 
//								temp_pair.to = find_in_sorted_list(to, cluster_idx_list, cluster_song_count, 0) + num_clusters;
//								one_more_count(tcount, temp_pair, 1.0 / (double) t);
//							}
//							//else if(member_vec[fr] == current_cluster_id && member_vec[to] != current_cluster_id)
//							else
//							{
//								temp_pair.fr = member_vec[fr];
//								temp_pair.to = member_vec[to];
//								one_more_count(tcount, temp_pair, 1.0 / (double) t);
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//
//	build_same_song_index(tcount);
//	return tcount;
//}
//
//void sort_tpair(HELEM* list_to_sort, int length)
//{
//	int i;
//	int j;
//	double key;
//	TPAIR key_idx;
//	//double val_array[length];  
//	if(length == 1)
//		return;
//	//Veccopy(p, val_array, length);
//	for(j = 1; j < length; j++)
//	{
//		key = list_to_sort[j].val;
//		key_idx = list_to_sort[j].key;
//		i = j - 1;
//		while(i >= 0 && list_to_sort[i].val < key)
//		{
//			list_to_sort[i + 1] = list_to_sort[i];
//			//val_array[i + 1] = val_array[i];
//			//tpair_array[i + 1] = tpair_array[i];
//			i--;
//		}
//		list_to_sort[i + 1].key = key_idx;
//		list_to_sort[i + 1].val = key;
//		//tpair_array[i + 1] = key_idx;
//		//val_array[i + 1] = key;
//	}
//}
//
//void initial_clustering(PHASH* tcount, int M)
//{
//	int k = tcount -> num_songs;
//	int i;
//	int j;
//	int t;
//	int a;
//	int b;
//	int a_member;
//	int b_member;
//	HELEM* helem_list = (HELEM*) malloc((tcount -> num_used) * sizeof(HELEM));
//	t = 0;
//	for(i = 0; i < (tcount -> length); i++)
//		if(!is_null_entry((tcount -> p)[i].key))
//			helem_list[t++] = (tcount -> p)[i];
//	/*
//	   {
//	   if((tcount -> p)[i].key.fr == 6432 && (tcount -> p)[i].key.to == 4217)
//	   printf("Bingo! %f\n", (tcount -> p)[i].val);
//	   }
//	   */
//	printf("Copy done, num used is %d.\n", tcount -> num_used);
//	sort_tpair(helem_list, tcount -> num_used);
//
//	/*
//	   for(i = 0; i < tcount -> num_used; i++)
//	   {
//	   if(helem_list[i].key.fr == 6432 && helem_list[i].key.to == 4217)
//	   printf("Bingo! %f\n", helem_list[i].val);
//	   }
//	   */
//	printf("Sort done.\n");
//
//	/*
//	   for(i = 0; i < (tcount -> num_used); i++)
//	   printf("%f\n", helem_list[i].val);
//	   */
//
//	int* cluster_membership_vec = (int*)malloc(k * sizeof(int));
//	for(i = 0; i < k; i++)
//		cluster_membership_vec[i] = -1;
//	int current_cluster_idx = 0;
//	int* cluster_counts = NULL;
//
//	for(i = 0; i < (tcount -> num_used); i++)
//	{
//		a = helem_list[i].key.fr;
//		b = helem_list[i].key.to;
//		/*
//		   if(a == 6432 && b == 4217)
//		   printf("Bingo!\n");
//		   */
//		a_member = cluster_membership_vec[a];
//		b_member = cluster_membership_vec[b];
//
//		//Both are clusterless. Create a new cluster.
//		if(a_member== -1 && b_member == -1)
//		{
//			if(cluster_counts == NULL)
//				cluster_counts = (int*)malloc(sizeof(int));
//			else
//				cluster_counts = (int*)realloc(cluster_counts, (current_cluster_idx + 1) * sizeof(int));
//			cluster_membership_vec[a] = current_cluster_idx;
//			cluster_membership_vec[b] = current_cluster_idx;
//			cluster_counts[current_cluster_idx++] = 2;
//		}
//		else if(a_member >= 0 && b_member == -1)
//		{
//			if(cluster_counts[a_member] < M)
//			{
//				cluster_membership_vec[b] = a_member;
//				cluster_counts[a_member]++;
//			}
//			else
//			{
//				cluster_counts = (int*)realloc(cluster_counts, (current_cluster_idx + 1) * sizeof(int));
//				cluster_membership_vec[b] = current_cluster_idx;
//				cluster_counts[current_cluster_idx++] = 1;
//			}
//		}
//		else if(b_member >= 0 && a_member == -1)
//		{
//			if(cluster_counts[b_member] < M)
//			{
//				cluster_membership_vec[a] = b_member;
//				cluster_counts[b_member]++;
//			}
//			else
//			{
//				cluster_counts = (int*)realloc(cluster_counts, (current_cluster_idx + 1) * sizeof(int));
//				cluster_membership_vec[a] = current_cluster_idx;
//				cluster_counts[current_cluster_idx++] = 1;
//			}
//		}
//		else if(a_member >= 0 && b_member >= 0 && a_member != b_member)
//		{
//			if(cluster_counts[a_member] + cluster_counts[b_member] <= M)
//			{
//				cluster_counts[a_member] += cluster_counts[b_member]; 
//				cluster_counts[b_member] = 0;
//				for(j = 0 ; j < k; j++)
//					if(cluster_membership_vec[j] == b_member)
//						cluster_membership_vec[j] = a_member;
//			}
//		}
//		if(cluster_membership_vec[a] < 0 || cluster_membership_vec[b] < 0)
//			printf("(%d, %d)\n", cluster_membership_vec[a], cluster_membership_vec[b]);
//	}
//
//	/*
//	   for(i = 0; i < current_cluster_idx; i++)
//	   printf("%d\n", cluster_counts[i]);
//	   */
//	//printf("number of different classes: %d\n", current_cluster_idx);
//	//sanity check
//	int temp_sum = 0;
//	for(i = 0; i < current_cluster_idx; i++)
//		temp_sum += cluster_counts[i];
//	//printf("total number of songs: %d. while the true number is %d\n", temp_sum, k);
//	/*
//	   for(i = 0; i < k; i++)
//	   if(cluster_membership_vec[i] < 0)
//	   printf("assertion failed: %d, %d\n", i, cluster_membership_vec[i]);
//	   */
//	//assert(cluster_membership_vec[i] >= 0);
//
//	int nonzero_cluster_count = 0;
//	for(i = 0; i < current_cluster_idx; i++)
//		if(cluster_counts[i] > 0)
//		{
//			printf("Cluster %d has %d songs.\n", i, cluster_counts[i]);
//			nonzero_cluster_count++;
//		}
//	printf("%d clusters in total.\n", nonzero_cluster_count);
//	free(helem_list);
//}
//
//void decompose_X(double** X, double** X_decomposed[], int d, int k, int num_clusters, int* cluster_song_relation[], int num_songs_in_each_cluster[])
//{
//	int i;
//	int j;
//	for(i = 0; i < num_clusters; i++)
//	{
//		X_decomposed[i] = zerosarray(num_songs_in_each_cluster[i], d);
//		for(j = 0; j < num_songs_in_each_cluster[i]; j++)
//		{
//			//printf("%d\n", cluster_song_relation[i][j]);fflush(stdout);
//			memcpy(X_decomposed[i][j], X[cluster_song_relation[i][j]], d * sizeof(double));
//		}
//	}
//}
//
//void recompose_X(double** X, double** X_decomposed[], int d, int k, int num_clusters, int* cluster_song_relation[], int num_songs_in_each_cluster[], double* local_counts_array[], double** upper_level_hubs)
//{
//	int i;
//	int j;
//	double temp_hub[d];
//	for(i = 0; i < num_clusters; i++)
//	{
//		find_hub(X_decomposed[i], temp_hub, num_songs_in_each_cluster[i], d, local_counts_array[i], 1e-5, 1e-10, 0);
//		add_vec(temp_hub, upper_level_hubs[i], d, -1.0);
//		for(j = 0; j < num_songs_in_each_cluster[i]; j++)
//		{
//			memcpy(X[cluster_song_relation[i][j]], X_decomposed[i][j],  d * sizeof(double));
//			add_vec(X[cluster_song_relation[i][j]], temp_hub, d, -1.0);
//		}
//		//X_decomposed[i] = zerosarray(num_songs_in_each_cluster[i], d);
//		Array2Dfree(X_decomposed[i], num_songs_in_each_cluster[i], d);
//	}
//}
//
//void count_local_hub_song_transitions(PHASH* tcount, int num_clusters, int current_cluster_id, double* parray)
//{
//	int to;
//	HELEM* p_valid_entry;
//	
//	p_valid_entry = tcount -> p_first_tran[current_cluster_id];
//	while(p_valid_entry != NULL)
//	{
//		to = (p_valid_entry -> key.to);
//		//printf("to: %d\n", to);
//		if(to >= num_clusters)
//			parray[to - num_clusters] += p_valid_entry -> val;
//		p_valid_entry = (p_valid_entry -> pnext);
//	}
//}
//
//int count_local_song_song_transtions(PHASH* tcount, int num_clusters)
//{
//	int fr;
//	int to;
//	HELEM* p_valid_entry;
//	int count = 0;
//	for(fr = num_clusters; fr < (tcount -> num_songs); fr++)
//	{
//		p_valid_entry = tcount -> p_first_tran[fr];
//		while(p_valid_entry != NULL)
//		{
//			to = (p_valid_entry -> key.to);
//			if(to < num_clusters)
//				printf("to : %d\n", to);
//			assert(to >= num_clusters);
//			count += (int) (p_valid_entry -> val);
//			p_valid_entry = (p_valid_entry -> pnext);
//		}
//	}
//	return count;
//}
//
//void compute_transition_stats(PHASH* tcount, int num_clusters, int* ss, int* hs, int* hh, int *total)
//{
//	int fr;
//	int to;
//	HELEM* p_valid_entry;
//	(*ss) = 0;
//	(*hs) = 0;
//	(*hh) = 0;
//	(*total) = 0;
//	for(fr = 0; fr < tcount -> num_songs; fr++)
//	{
//		p_valid_entry = tcount -> p_first_tran[fr];
//		while(p_valid_entry != NULL)
//		{
//			to = (p_valid_entry -> key.to);
//			if(fr >= num_clusters && to >= num_clusters)
//				(*ss) += (int) (p_valid_entry -> val);
//			else if(fr < num_clusters && to >= num_clusters)
//				(*hs) += (int) (p_valid_entry -> val);
//			else if(fr < num_clusters && to < num_clusters)
//				(*hh) += (int) (p_valid_entry -> val);
//			else
//			{
//				printf("Unexpected transition observed.\n");
//				exit(1);
//			}
//			(*total) += (int) (p_valid_entry -> val);
//			//count += (int) (p_valid_entry -> val);
//			p_valid_entry = (p_valid_entry -> pnext);
//		}
//	}
//}
