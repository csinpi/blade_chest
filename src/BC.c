#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "LogisticEmbed.h"
#include "HLogisticEmbed.h"
#include "BC.h"
#include "LogisticEmbed_common.h"
#include "PairHashTable.h"
#include "TransitionTable.h"
#include "EmbedIO.h"
#include "GamePairHash.h"
#define INT_BUF_SZ 2000000
#define DOUBLE_BUF_SZ 2000
#define FOR_TRAINING 0
#define FOR_VALIDATION 1
#define FOR_TESTING 2
#define FOR_STRING_SIZE 20

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

char trainfile[200];
char testfile[200];
char embeddingfile[200];
char matchupmatfile[200];

GRECORDS read_game_records_data(char* filename, int verbose)
{
	int i;
	int t;
	int tt;
	GRECORDS grs;
	int pa, pb, na, nb;
	char templine[INT_BUF_SZ];
	FILE* fp = fopen(filename, "r");
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_players = extract_tail_int(templine, "numPlayers: ");
	grs.all_players = (char**)malloc(grs.num_players * sizeof(char*));
	for(i = 0; i < grs.num_players; i++)
		grs.all_players[i] = (char*)malloc(100 * sizeof(char));
	for(i = 0; i < grs.num_players; i++)
	{
		if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		{
			t = 0;
			tt = 0;
			while(templine[t] != ' ')
				t++;
			t++;
			while(templine[t] != '\n')
				grs.all_players[i][tt++] = templine[t++];
			grs.all_players[i][tt] = '\0';
		}
	}

	if(verbose)
	{
		for(i = 0; i < grs.num_players; i++)
		{
			printf("%s\n", grs.all_players[i]);
		}
	}
	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_games = extract_tail_int(templine, "numGames: ");

	grs.all_games = (MATCH*)malloc(grs.num_games * sizeof(MATCH));
	grs.test_mask = (int*)malloc(grs.num_games * sizeof(int));

	t = 0;
	char for_string[FOR_STRING_SIZE];
	while(fgets(templine, INT_BUF_SZ, fp) != NULL)
	{
		if(templine[0] == 'F')
			grs.with_mask = 1;
		else
			grs.with_mask = 0;

		if(grs.with_mask)
		{
			int counter = 0;
			while(templine[counter] != ' ')
				counter++;
			memcpy(for_string, templine, counter * sizeof(char));
			for_string[counter + 1] = '\0';
			if(strcmp("FOR_TRAINING", for_string) == 0)
				grs.test_mask[t] = FOR_TRAINING;
			else if(strcmp("FOR_VALIDATION", for_string) == 0)
				grs.test_mask[t] = FOR_VALIDATION;
			else
				grs.test_mask[t] = FOR_TESTING;

			extract_game_detal_from_line(templine + counter + 1, &pa, &pb, &na, &nb);

		}
		else
			extract_game_detal_from_line(templine, &pa, &pb, &na, &nb);



		for(i = 0; i < na; i++)
		{
			grs.all_games[t].pa = pa;
			grs.all_games[t].pb = pb;
			grs.all_games[t].pw = pa;
			t++;
		}
		for(i = 0; i < nb; i++)
		{
			grs.all_games[t].pa = pa;
			grs.all_games[t].pb = pb;
			grs.all_games[t].pw = pb;
			t++;
		}
	}
	
	if(verbose)
		for(i = 0; i < grs.num_games; i++)
			printf("(%d, %d, %d)\n", grs.all_games[i].pa, grs.all_games[i].pb, grs.all_games[i].pw);


	close(fp);
	return grs;
}

void free_game_records_data(GRECORDS grs)
{
	int i;
	for(i = 0; i < grs.num_players; i++)
		free(grs.all_players[i]);
	free(grs.all_games);
	free(grs.test_mask);
}

void free_game_embedding(GEMBEDDING gebd)
{
	if(gebd.rankon)
		free(gebd.ranks);
	Array2Dfree(gebd.tvecs, gebd.k, gebd.d);
	Array2Dfree(gebd.hvecs, gebd.k, gebd.d);
}

void init_game_embedding(GEMBEDDING* p_gebd, int k, int d, int rankon, int modeltype)
{
	(p_gebd -> k) = k;
	(p_gebd -> d) = d;
	(p_gebd -> rankon) = rankon;
	(p_gebd -> modeltype) = modeltype;
	if(rankon)
		(p_gebd -> ranks) = (double*)calloc(k , sizeof(double));
	(p_gebd -> tvecs) = randarray(k, d, 1.0);
	(p_gebd -> hvecs) = randarray(k, d, 1.0);
	//(p_gebd -> tvecs) = zerosarray(k, d);
	//(p_gebd -> hvecs) = zerosarray(k, d);
}

void copy_game_embedding(GEMBEDDING* p_dest, GEMBEDDING* p_src)
{
	assert((p_src -> k) ==  (p_dest -> k));
	assert((p_src -> d) ==  (p_dest -> d));
	assert((p_src -> rankon) ==  (p_dest -> rankon));
	assert((p_src -> modeltype) ==  (p_dest -> modeltype));
	if(p_src -> rankon)
		memcpy(p_dest -> ranks, p_src -> ranks, (p_src -> k) * sizeof(double));
	Array2Dcopy(p_src -> tvecs, p_dest -> tvecs, (p_src -> k), (p_src -> d));
	Array2Dcopy(p_src -> hvecs, p_dest -> hvecs, (p_src -> k), (p_src -> d));
}

void extract_game_detal_from_line(char* str, int* pa, int* pb, int* na, int* nb)
{
	int i;
	int t;
	int tt;
	char buffer[INT_BUF_SZ];
	t = 0;
	tt = 0;
	while(str[t] != ':')
		buffer[tt++] = str[t++]; 
	t++;
	buffer[tt] = '\0';
	*pa = atoi(buffer); 

	tt = 0;
	while(str[t] != ' ')
		buffer[tt++] = str[t++]; 
	t++;
	buffer[tt] = '\0';
	*pb = atoi(buffer); 

	tt = 0;
	while(str[t] != ':')
		buffer[tt++] = str[t++]; 
	t++;
	buffer[tt] = '\0';
	*na = atoi(buffer); 

	tt = 0;
	while(str[t] != '\n' && str[t] != '\0')
		buffer[tt++] = str[t++]; 
	t++;
	buffer[tt] = '\0';
	*nb = atoi(buffer); 
}

GEMBEDDING game_embedding_train_test(GRECORDS grs, PARAS myparas)
{
	int i, j, t;
	srand(myparas.seed);
	int k = grs.num_players;
	int d = myparas.d;
	int rankon = myparas.rankon;

	if (d < 1 && rankon == 0)
	{
		printf("The model is empty according to parameter d and r.\n");
		exit(1);
	}

	GEMBEDDING main_embedding, last_embedding, best_embedding, ebd_grad;
	init_game_embedding(&main_embedding, k, d, rankon, myparas.modeltype);
	init_game_embedding(&last_embedding, k, d, rankon, myparas.modeltype);
	init_game_embedding(&best_embedding, k, d, rankon, myparas.modeltype);
	init_game_embedding(&ebd_grad, k, d, rankon, myparas.modeltype);

	//int* test_mask = (int*)calloc(grs.num_games, sizeof(int));

	//for(i = 0; i < grs.num_games; i++)
	//	if(i % myparas.train_ratio == 0)
	//		test_mask[i] = 1;
	//	else
	//		num_training_games++;

	//If original data does not have mask, randomly generate it here.
	if(!grs.with_mask)
	{
		srand(myparas.seed);
		shuffle_match_array(grs.all_games, grs.num_games);
		for(i = 0; i < grs.num_games; i++)
		{
			if(i % 10 <= 4)
			{
				grs.test_mask[i] = FOR_TRAINING;
				//num_training_games++;
			}
			else if(i % 10 <= 6)
				grs.test_mask[i] = FOR_VALIDATION;
			else
				grs.test_mask[i] = FOR_TESTING;
		}
	}

	int num_training_games = 0;
	for(i = 0; i < grs.num_games; i++)
		if(grs.test_mask[i] == FOR_TRAINING)
			num_training_games++;

	//for(i = 0; i < grs.num_games; i++)
	//	printf("%d\n", grs.test_mask[i]);

	int time_bomb = 0;
	int reboot_bomb = 0;

	int iteration_count = 0;
	double best_obj = -DBL_MAX;
	int best_iter = -1;

	double ll = 0.0;
	double realn = 0.0;

	double ll_test = 0.0;
	double realn_test = 0.0;
	double correct_test = 0.0;
	
	double ll_test_naive = 0.0;
	double correct_test_naive = 0.0;

	double ll_validate = 0.0;
	double realn_validate = 0.0;
	double correct_validate = 0.0;
	
	double ll_validate_naive = 0.0;
	double correct_validate_naive = 0.0;

	double obj, obj_prev, avg_ll, avg_ll_prev, avg_ll_test, avg_ll_test_naive, avg_ll_validate, avg_ll_validate_naive;

	obj_prev = -DBL_MAX;
	obj = -DBL_MAX;
	avg_ll = -DBL_MAX;
	avg_ll_prev = -DBL_MAX;

	int triple_id;
	int counter;

	MATCH game_triple;
	GELE game_ele;

	int a, b, winner, na, nb;
	double prob_a, prob_b, coeff, mf;


	//Variables for Addagrad
	double *grad1, *grad2, *grad3, *grad4, *reg_grad1, *reg_grad2, *reg_grad3, *reg_grad4;
	double* temp_d;
	grad1 = (double*)malloc(d * sizeof(double));
	grad2 = (double*)malloc(d * sizeof(double));
	grad3 = (double*)malloc(d * sizeof(double));
	grad4 = (double*)malloc(d * sizeof(double));
	
	reg_grad1 = (double*)malloc(d * sizeof(double));
	reg_grad2 = (double*)malloc(d * sizeof(double));
	reg_grad3 = (double*)malloc(d * sizeof(double));
	reg_grad4 = (double*)malloc(d * sizeof(double));

	temp_d = (double*)malloc(d * sizeof(double));

	double* game_distr = (double*)calloc(k, sizeof(double));

	GPHASH only_training_games = create_gp_hash(2 * grs.num_games, k);

	GPAIR history_result, temp_pair;
	int whatever, idx;

	//This is only for the baseline comparison
	GELE temp_gele;

	for(i = 0; i < grs.num_games; i++)
	{
		if(grs.test_mask[i] == FOR_TRAINING)
		{
			//printf("<%d, %d, %d>\n", grs.all_games[i].pa, grs.all_games[i].pb, grs.all_games[i].pw);
			temp_gele.key.fst = grs.all_games[i].pa;
			temp_gele.key.snd = grs.all_games[i].pb;

			game_distr[grs.all_games[i].pa] += 1.0;
			game_distr[grs.all_games[i].pb] += 1.0;

			if(grs.all_games[i].pa == grs.all_games[i].pw)
			{
				temp_gele.val.fst = 1;
				temp_gele.val.snd = 0;
			}
			else
			{
				temp_gele.val.fst = 0;
				temp_gele.val.snd = 1;
			}
			update_with_gele(temp_gele, &only_training_games);
		}
	}

	//printf("num_used: %d\n", only_training_games.num_used);

	GELE training_games_aggregated[only_training_games.num_used];
	t = 0;
	for(i = 0; i < only_training_games.length; i++)
		if(!gele_is_empty(only_training_games.array[i]))
			training_games_aggregated[t++] = only_training_games.array[i];

	GELE training_games_separated[num_training_games];

	t = 0;
	for(i = 0; i < grs.num_games; i++)
	{
		if(grs.test_mask[i] == FOR_TRAINING)
		{
			a = grs.all_games[i].pa;
			b = grs.all_games[i].pb;
			winner = grs.all_games[i].pw;
			training_games_separated[t].key.fst = a;
			training_games_separated[t].key.snd = b;
			if(a == winner)
			{
				training_games_separated[t].val.fst = 1;
				training_games_separated[t].val.snd = 0;
			}
			else
			{
				training_games_separated[t].val.fst = 0;
				training_games_separated[t].val.snd = 1;
			}
			t++;
		}
	}

	//print_vec(game_distr, k);


	//Initialize ebd_grad to contain all ones
	for(i = 0; i < k; i++)
	{
		for(j = 0; j < d; j++)
		{
			ebd_grad.tvecs[i][j] = 1.0;
			ebd_grad.hvecs[i][j] = 1.0;
		}
		if(rankon)
			ebd_grad.ranks[i] = 1.0;
	}

	double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
	int llhood_track_index = 0;

	for (i = 0; i < myparas.num_llhood_track; i++)
		last_n_llhoods[i] = 1.0;



	FILE* fp;
	double** true_matchup_mat;
	double** predict_matchup_mat = zerosarray(k, k);
	double** naive_matchup_mat = zerosarray(k, k);
	int true_m, true_n;
	if(matchupmatfile[0] != '\0')
	{
		fp = fopen(matchupmatfile, "r");
		read_double_mat(fp, &true_matchup_mat, &true_m, &true_n);
		fclose(fp);
		assert(true_m == k && true_n == k);
		//print_mat(true_matchup_mat, true_m, true_n);
	}



	int adjust_vec = 1;

	if(myparas.training_mode == 0)
	{

		//Computing obj and ll for initialization
		compute_ll_obj(&avg_ll, &obj, only_training_games.num_used, training_games_aggregated, main_embedding, myparas);

		while(iteration_count++ < myparas.max_iter)
		{
			copy_game_embedding(&last_embedding, &main_embedding);
			obj_prev = obj;
			avg_ll_prev = avg_ll;
			ll = 0.0;
			realn = 0.0;
			srand(myparas.seed);
			shuffle_gele_array(training_games_aggregated, (size_t)only_training_games.num_used);
			printf("Iteration %d\n", iteration_count);
			for(triple_id = 0; triple_id < only_training_games.num_used; triple_id++)
			{
				game_ele = training_games_aggregated[triple_id]; 
				a = game_ele.key.fst;
				b = game_ele.key.snd;
				na = game_ele.val.fst;
				nb = game_ele.val.snd;

				mf = matchup_fun(main_embedding, a, b, myparas.modeltype);
				prob_a = logistic_fun(mf);
				prob_b = 1.0 - prob_a;

				coeff = (na * prob_b - nb * prob_a);

				//printf("%f %f %f\n", prob_a, prob_b, coeff);
				realn += (na + nb);
				if(d > 0 && adjust_vec)
				{
					if(myparas.modeltype == 0 || myparas.modeltype == 1)
					{
						//Find the gradients
						Veccopy(main_embedding.hvecs[b], grad1, d);
						scale_vec(grad1, d, -2.0 * coeff);
						add_vec(grad1, main_embedding.tvecs[a], d, 2.0 * coeff);

						Veccopy(main_embedding.hvecs[a], grad2, d);
						scale_vec(grad2, d, -2.0 * coeff);
						add_vec(grad2, main_embedding.tvecs[b], d, 2.0 * coeff);

						Veccopy(main_embedding.hvecs[a], grad3, d);
						scale_vec(grad3, d, 2.0 * coeff);
						add_vec(grad3, main_embedding.tvecs[b], d, -2.0 * coeff);

						Veccopy(main_embedding.hvecs[b], grad4, d);
						scale_vec(grad4, d, 2.0 * coeff);
						add_vec(grad4, main_embedding.tvecs[a], d, -2.0 * coeff);

						if(myparas.modeltype == 1)
						{
							scale_vec(grad1, d, pow(2.0 * coeff, 2.0) / pow(vec_norm(grad1, d), 2.0));
							scale_vec(grad2, d, pow(2.0 * coeff, 2.0) / pow(vec_norm(grad2, d), 2.0));
							scale_vec(grad3, d, pow(2.0 * coeff, 2.0) / pow(vec_norm(grad3, d), 2.0));
							scale_vec(grad4, d, pow(2.0 * coeff, 2.0) / pow(vec_norm(grad4, d), 2.0));

						}

						//Update the embeddings
						add_vec(main_embedding.tvecs[a], grad1, d, myparas.eta);

						add_vec(main_embedding.hvecs[a], grad2, d, myparas.eta);

						add_vec(main_embedding.tvecs[b], grad3, d, myparas.eta);

						add_vec(main_embedding.hvecs[b], grad4, d, myparas.eta);


						//////////////////////////////////////////////////
						//if(game_distr[a] > 0.0)
						//{
						//	if(myparas.regularization_type == 0)
						//	{
						//		Veccopy(main_embedding.tvecs[a], reg_grad1, d);
						//		Veccopy(main_embedding.hvecs[a], reg_grad2, d);
						//	}
						//	else if(myparas.regularization_type == 1)
						//	{
						//		Veccopy(main_embedding.tvecs[a], reg_grad1, d);
						//		add_vec(reg_grad1, main_embedding.hvecs[a], d, -1.0);
						//		Veccopy(reg_grad1, reg_grad2, d);
						//		scale_vec(reg_grad2, d, -1.0);
						//	}
						//	else if(myparas.regularization_type == 2)
						//	{
						//		Veccopy(main_embedding.tvecs[a], reg_grad1, d);
						//		scale_vec(reg_grad1, d, 2.0);
						//		add_vec(reg_grad1, main_embedding.hvecs[a], d, -1.0);
						//		Veccopy(main_embedding.hvecs[a], reg_grad2, d);
						//		scale_vec(reg_grad2, d, 2.0);
						//		add_vec(reg_grad2, main_embedding.tvecs[a], d, -1.0);
						//	}

						//	scale_vec(reg_grad1, d, -2.0 * myparas.lambda * (na + nb) / game_distr[a]);
						//	scale_vec(reg_grad2, d, -2.0 * myparas.lambda * (na + nb) / game_distr[a]);

						//	add_vec(grad1, reg_grad1, d, 1.0);

						//	add_vec(grad2, reg_grad2, d, 1.0);

						//	add_vec(main_embedding.tvecs[a], reg_grad1, d, myparas.eta);
						//	add_vec(main_embedding.hvecs[a], reg_grad2, d, myparas.eta);
						//}

						//if(game_distr[b] > 0.0)
						//{
						//	if(myparas.regularization_type == 0)
						//	{
						//		Veccopy(main_embedding.tvecs[b], reg_grad3, d);
						//		Veccopy(main_embedding.hvecs[b], reg_grad4, d);
						//	}
						//	else if(myparas.regularization_type == 1)
						//	{
						//		Veccopy(main_embedding.tvecs[b], reg_grad3, d);
						//		add_vec(reg_grad3, main_embedding.hvecs[b], d, -1.0);
						//		Veccopy(reg_grad3, reg_grad4, d);
						//		scale_vec(reg_grad4, d, -1.0);
						//	}
						//	else if(myparas.regularization_type == 2)
						//	{
						//		Veccopy(main_embedding.tvecs[b], reg_grad3, d);
						//		scale_vec(reg_grad3, d, 2.0);
						//		add_vec(reg_grad3, main_embedding.hvecs[b], d, -1.0);
						//		Veccopy(main_embedding.hvecs[b], reg_grad4, d);
						//		scale_vec(reg_grad4, d, 2.0);
						//		add_vec(reg_grad4, main_embedding.tvecs[b], d, -1.0);
						//	}

						//	scale_vec(reg_grad3, d, -2.0 * myparas.lambda * (na + nb) / game_distr[b]);
						//	scale_vec(reg_grad4, d, -2.0 * myparas.lambda * (na + nb) / game_distr[b]);

						//	add_vec(grad3, reg_grad3, d, 1.0);

						//	add_vec(grad4, reg_grad4, d, 1.0);

						//	add_vec(main_embedding.tvecs[b], reg_grad3, d, myparas.eta);
						//	add_vec(main_embedding.hvecs[b], reg_grad4, d, myparas.eta);
						//}
						//////////////////////////////////////////////////
					}
					//The new inner product model
					else if(myparas.modeltype == 2)
					{
						Veccopy(main_embedding.hvecs[b], grad1, d);
						scale_vec(grad1, d, -1.0 * coeff);

						Veccopy(main_embedding.tvecs[b], grad2, d);
						scale_vec(grad2, d, 1.0 * coeff);

						Veccopy(main_embedding.hvecs[a], grad3, d);
						scale_vec(grad3, d, 1.0 * coeff);

						Veccopy(main_embedding.tvecs[a], grad4, d);
						scale_vec(grad4, d, -1.0 * coeff);

						//Update the embeddings
						add_vec(main_embedding.tvecs[a], grad1, d, myparas.eta);

						add_vec(main_embedding.hvecs[a], grad2, d, myparas.eta);

						add_vec(main_embedding.tvecs[b], grad3, d, myparas.eta);

						add_vec(main_embedding.hvecs[b], grad4, d, myparas.eta);
					}
				}
				if(rankon)
				{
					main_embedding.ranks[a] +=  coeff * myparas.eta;
					main_embedding.ranks[b] -=  coeff * myparas.eta;
				}
				ll += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
			}

			if(d > 0 && adjust_vec)
			{
				for(a = 0; a < k; a++)
				{

					if(myparas.regularization_type == 0)
					{
						Veccopy(main_embedding.tvecs[a], reg_grad1, d);
						Veccopy(main_embedding.hvecs[a], reg_grad2, d);
					}
					else if(myparas.regularization_type == 1)
					{
						Veccopy(main_embedding.tvecs[a], reg_grad1, d);
						add_vec(reg_grad1, main_embedding.hvecs[a], d, -1.0);
						Veccopy(reg_grad1, reg_grad2, d);
						scale_vec(reg_grad2, d, -1.0);
					}
					else if(myparas.regularization_type == 2)
					{
						Veccopy(main_embedding.tvecs[a], reg_grad1, d);
						scale_vec(reg_grad1, d, 2.0);
						add_vec(reg_grad1, main_embedding.hvecs[a], d, -1.0);
						Veccopy(main_embedding.hvecs[a], reg_grad2, d);
						scale_vec(reg_grad2, d, 2.0);
						add_vec(reg_grad2, main_embedding.tvecs[a], d, -1.0);
					}

					scale_vec(reg_grad1, d, -2.0 * myparas.lambda);
					scale_vec(reg_grad2, d, -2.0 * myparas.lambda);
					add_vec(main_embedding.tvecs[a], reg_grad1, d, myparas.eta);
					add_vec(main_embedding.hvecs[a], reg_grad2, d, myparas.eta);
				}
			}

			//if(rankon)
			//	for(a = 0; a < k; a++)
			//		main_embedding.ranks[a] -= 2.0 * myparas.lambda * main_embedding.ranks[a];



			//honestly computing ll
			compute_ll_obj(&avg_ll, &obj, only_training_games.num_used, training_games_aggregated, main_embedding, myparas);

			printf("Avg training log-likelihood: %f\n", avg_ll);
			printf("Training obj function: %f\n", obj);
			printf("Training prev_obj function: %f\n", obj_prev);

			if(isnan(obj))
				break;


			if(1)
			{
				if(obj < obj_prev)
				{
					copy_game_embedding(&main_embedding, &last_embedding);
					obj = obj_prev;
					avg_ll = avg_ll_prev;
					myparas.eta /= myparas.beta;
					printf("Rebooting... Reducing eta to %f\n\n", myparas.eta);
					reboot_bomb++;
					if(reboot_bomb > myparas.bomb_thresh)
						break;
					else
						continue;
				}
				else
					reboot_bomb = 0;

				//printf("%f %f\n", obj, obj_prev);
				//printf("%f\n", fabs(obj - obj_prev) / fabs(obj_prev));
				if(obj > obj_prev && fabs(obj - obj_prev) / fabs(obj_prev) < 0.01)
				{
					myparas.eta *= myparas.alpha;
					printf("Increasing eta to %f\n\n", myparas.eta);
				}
			}

			//if(myparas.modeltype == 0 && mat_norm_diff(main_embedding.tvecs, main_embedding.hvecs, k, d) / (k * d) < 1e-8)
			//{
			//	adjust_vec = 0;
			//	Array2Dcopy(main_embedding.tvecs, main_embedding.hvecs, k, d);
			//}



			if(obj > best_obj) 
			{
				best_obj = obj;
				copy_game_embedding(&best_embedding, &last_embedding);
				time_bomb = 0;
				best_iter = iteration_count;
			}
			else
				time_bomb++;

			last_n_llhoods[llhood_track_index] = obj;
			double min_llhood = DBL_MAX, max_llhood = -DBL_MAX, total_llhood = 0.0;
			for(i = 0; i < myparas.num_llhood_track; i++) {
				total_llhood += last_n_llhoods[i];
				if (last_n_llhoods[i] < min_llhood)
					min_llhood = last_n_llhoods[i];
				if (last_n_llhoods[i] > max_llhood)
					max_llhood = last_n_llhoods[i];
			}
			printf("min_obj %f, max_obj %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
			if (myparas.num_llhood_track > 0
					&& fabs(min_llhood - max_llhood) < myparas.eps 
					&& t > myparas.num_llhood_track) {
				break;
			}
			llhood_track_index++;
			llhood_track_index %= myparas.num_llhood_track;



			if(time_bomb > myparas.bomb_thresh)
				break;
			putchar('\n');
		}
		copy_game_embedding(&main_embedding, &best_embedding);
		printf("\nBest training model is from iteration %d\n", best_iter);
		printf("Adjusting the vectors at the end: %d\n", adjust_vec);
		printf("Final eta: %f\n", myparas.eta);
	}
	else if(myparas.training_mode == 1)
	{
		while(iteration_count++ < myparas.max_iter)
		{
			//Store the last embedding before the gradient step updates it
			copy_game_embedding(&last_embedding, &main_embedding);

			//Initilize all measurements
			ll = 0.0;
			realn = 0.0;

			//ll_test = 0.0;
			//realn_test = 0.0;
			//correct_test = 0.0;

			//ll_test_naive = 0.0;
			//correct_test_naive = 0.0;

			triple_id  = 0;
			printf("Iteration %d\n", iteration_count);
			for(counter = 0; counter < myparas.training_multiplier * only_training_games.num_used; counter++)
			//for(counter = 0; counter < myparas.training_multiplier * num_training_games; counter++)
			{
				triple_id = random_in_range(0, only_training_games.num_used);
				game_ele = training_games_aggregated[triple_id]; 
				//triple_id = random_in_range(0, num_training_games);
				//game_ele = training_games_separated[triple_id]; 
				a = game_ele.key.fst;
				b = game_ele.key.snd;
				na = game_ele.val.fst;
				nb = game_ele.val.snd;
				//winner = game_triple.pw;

				mf = matchup_fun(main_embedding, a, b, myparas.modeltype);
				prob_a = logistic_fun(mf);
				//printf("prob_a %f\n", prob_a);
				//assert(prob_a >= 0.0 && prob_a <= 1.0);
				prob_b = 1.0 - prob_a;

				//printf("%f, %f, %f, %f\n", main_embedding.ranks[a], main_embedding.ranks[b], prob_a, prob_b);
				coeff = (na * prob_b - nb * prob_a);
				//if(winner == a)
				//	coeff = prob_b;
				//else
				//	coeff = -prob_a;

				//triple used for training
				//if(test_mask[triple_id] == 0)
				//realn += 1.0;
				realn += (na + nb);
				if(d > 0)
				{
					//Accumulate the adagrad
					Veccopy(main_embedding.hvecs[b], grad1, d);
					scale_vec(grad1, d, -2.0 * coeff);
					add_vec(grad1, main_embedding.tvecs[a], d, 2.0 * coeff);

					Veccopy(main_embedding.hvecs[a], grad2, d);
					scale_vec(grad2, d, -2.0 * coeff);
					add_vec(grad2, main_embedding.tvecs[b], d, 2.0 * coeff);

					Veccopy(main_embedding.hvecs[a], grad3, d);
					scale_vec(grad3, d, 2.0 * coeff);
					add_vec(grad3, main_embedding.tvecs[b], d, -2.0 * coeff);

					Veccopy(main_embedding.hvecs[b], grad4, d);
					scale_vec(grad4, d, 2.0 * coeff);
					add_vec(grad4, main_embedding.tvecs[a], d, -2.0 * coeff);

					//Update the embeddings
					Veccopy(ebd_grad.tvecs[a], temp_d, d);
					pow_on_vec(temp_d, d, -0.5);
					vec_mult(temp_d, grad1, temp_d, d);
					add_vec(main_embedding.tvecs[a], temp_d, d, myparas.eta);

					Veccopy(ebd_grad.hvecs[a], temp_d, d);
					pow_on_vec(temp_d, d, -0.5);
					vec_mult(temp_d, grad2, temp_d, d);
					add_vec(main_embedding.hvecs[a], temp_d, d, myparas.eta);

					Veccopy(ebd_grad.tvecs[b], temp_d, d);
					pow_on_vec(temp_d, d, -0.5);
					vec_mult(temp_d, grad3, temp_d, d);
					add_vec(main_embedding.tvecs[b], temp_d, d, myparas.eta);

					Veccopy(ebd_grad.hvecs[b], temp_d, d);
					pow_on_vec(temp_d, d, -0.5);
					vec_mult(temp_d, grad4, temp_d, d);
					add_vec(main_embedding.hvecs[b], temp_d, d, myparas.eta);


					if(game_distr[a] > 0.0)
					{
						Veccopy(main_embedding.tvecs[a], reg_grad1, d);
						add_vec(reg_grad1, main_embedding.hvecs[a], d, -1.0);
						Veccopy(reg_grad1, reg_grad2, d);

						scale_vec(reg_grad1, d, -2.0 * myparas.lambda * (double)(na + nb) / game_distr[a]);
						scale_vec(reg_grad2, d, -2.0 * myparas.lambda * (double)(na + nb) / game_distr[a]);

						Veccopy(ebd_grad.tvecs[a], temp_d, d);
						pow_on_vec(temp_d, d, -0.5);
						vec_mult(temp_d, reg_grad1, temp_d, d);
						add_vec(main_embedding.tvecs[a], temp_d, d, myparas.eta);

						Veccopy(ebd_grad.hvecs[a], temp_d, d);
						pow_on_vec(temp_d, d, -0.5);
						vec_mult(temp_d, reg_grad2, temp_d, d);
						add_vec(main_embedding.hvecs[a], temp_d, d, myparas.eta);

						add_vec(grad1, reg_grad1, d, 1.0);

						add_vec(grad2, reg_grad2, d, 1.0);
					}

					if(game_distr[b] > 0.0)
					{
						Veccopy(main_embedding.tvecs[b], reg_grad3, d);
						add_vec(reg_grad3, main_embedding.hvecs[b], d, -1.0);
						Veccopy(reg_grad3, reg_grad4, d);
						scale_vec(reg_grad4, d, -1.0);

						scale_vec(reg_grad3, d, -2.0 * myparas.lambda * (double)(na + nb) / game_distr[b]);
						scale_vec(reg_grad4, d, -2.0 * myparas.lambda * (double)(na + nb) / game_distr[b]);


						Veccopy(ebd_grad.tvecs[b], temp_d, d);
						pow_on_vec(temp_d, d, -0.5);
						vec_mult(temp_d, reg_grad3, temp_d, d);
						add_vec(main_embedding.tvecs[b], temp_d, d, myparas.eta);

						Veccopy(ebd_grad.hvecs[b], temp_d, d);
						pow_on_vec(temp_d, d, -0.5);
						vec_mult(temp_d, reg_grad4, temp_d, d);
						add_vec(main_embedding.hvecs[b], temp_d, d, myparas.eta);

						add_vec(grad3, reg_grad3, d, 1.0);

						add_vec(grad4, reg_grad4, d, 1.0);
					}

					Veccopy(grad1, temp_d, d);
					pow_on_vec(temp_d, d, 2.0);
					add_vec(ebd_grad.tvecs[a], temp_d, d, 1.0); 

					Veccopy(grad2, temp_d, d);
					pow_on_vec(temp_d, d, 2.0);
					add_vec(ebd_grad.hvecs[a], temp_d, d, 1.0); 

					Veccopy(grad3, temp_d, d);
					pow_on_vec(temp_d, d, 2.0);
					add_vec(ebd_grad.tvecs[b], temp_d, d, 1.0); 

					Veccopy(grad4, temp_d, d);
					pow_on_vec(temp_d, d, 2.0);
					add_vec(ebd_grad.hvecs[b], temp_d, d, 1.0); 
				}
				if(rankon)
				{
					main_embedding.ranks[a] += pow(ebd_grad.ranks[a], -0.5) * coeff * myparas.eta;
					main_embedding.ranks[b] -= pow(ebd_grad.ranks[b], -0.5) * coeff * myparas.eta;

					ebd_grad.ranks[a] += pow(coeff, 2.0);
					ebd_grad.ranks[b] += pow(coeff, 2.0);
				}
				//ll += ((winner == a)? safe_log_logit(-mf) : safe_log_logit(mf));
				//ll += ((winner == a)? log(prob_a) : log(prob_b));
				//ll += (float) na * log(prob_a) + (float)nb * log(prob_b);
				ll += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
				//else
				//{
				//	realn_test += 1.0;
				//	//ll_test += ((winner == a)? safe_log_logit(-mf) : safe_log_logit(mf));
				//	ll_test += ((winner == a)? log(prob_a) : log(prob_b));

				//	if((winner == a && prob_a >= 0.5) || (winner == b && prob_b > 0.5))
				//		correct_test += 1.0;

				//	temp_pair.fst = a;
				//	temp_pair.snd = b;
				//	idx = find_gp_key(temp_pair, only_training_games, &whatever);
				//	if(idx < 0)
				//	{
				//		history_result.fst = 0;
				//		history_result.snd = 0;
				//	}
				//	else
				//		history_result = only_training_games.array[idx].val;

				//	if(((history_result.fst >= history_result.snd) && (winner == a)) || ((history_result.fst < history_result.snd) && (winner == b)))
				//		correct_test_naive += 1.0;

				//	if(history_result.fst == 0 || history_result.snd == 0)
				//	{
				//		history_result.fst++;
				//		history_result.snd++;
				//	}
				//	if(winner == a)
				//		ll_test_naive += log((double)history_result.fst / ((double)(history_result.fst + history_result.snd)));
				//	else
				//		ll_test_naive += log((double)history_result.snd / ((double)(history_result.fst + history_result.snd)));
				//}
			}

			//print_vec(main_embedding.ranks, k);

			avg_ll = ll / realn;
			//avg_ll_test = ll_test / realn_test;
			//avg_ll_test_naive = ll_test_naive / realn_test;

			printf("Avg training log-likelihood: %f\n", avg_ll);
			//printf("Avg testing log-likelihood: %f\n", avg_ll_test);
			//printf("Avg testing accuracy: %f\n", (correct_test / realn_test));
			//printf("Avg naive testing log-likelihood: %f\n", avg_ll_test_naive);
			//printf("Avg naive testing accuracy: %f\n", (correct_test_naive / realn_test));

			obj = avg_ll - myparas.lambda * (pow(frob_norm(main_embedding.tvecs, k, d), 2.0) + pow(frob_norm(main_embedding.hvecs, k, d), 2.0));
			printf("Training obj function: %f\n", obj);
			putchar('\n');




			if(obj > best_obj) 
			{
				best_obj = obj;
				copy_game_embedding(&best_embedding, &last_embedding);
				time_bomb = 0;
			}
			else
				time_bomb++;
			if(time_bomb > myparas.bomb_thresh)
				break;
		}
		copy_game_embedding(&main_embedding, &best_embedding);
	}

	//print_vec(main_embedding.ranks, k);
	
	//print_mat(main_embedding.tvecs, k, d);
	//printf("---------------------\n");
	//print_mat(main_embedding.hvecs, k, d);
	//printf("---------------------\n");
	//print_vec(main_embedding.ranks, k);


	//write_GameEmbedding_to_file(main_embedding, embeddingfile);

	//----------------------------- Final Test -----------------------------
	ll_test = 0.0;
	realn_test = 0.0;
	correct_test = 0.0;

	ll_test_naive = 0.0;
	correct_test_naive = 0.0;

	ll_validate = 0.0;
	realn_validate = 0.0;
	correct_validate = 0.0;

	ll_validate_naive = 0.0;
	correct_validate_naive = 0.0;


	for(triple_id = 0; triple_id < grs.num_games; triple_id++)
	{
		//printf("%d, %d\n", triple_id, grs.num_games);
		game_triple = grs.all_games[triple_id]; 
		a = game_triple.pa;
		b = game_triple.pb;
		winner = game_triple.pw;
		//printf("(%d, %d, %d, %d)\n", a, b, winner, grs.test_mask[triple_id]);

		mf = matchup_fun(main_embedding, a, b, myparas.modeltype);
		//printf("mf: %f\n", mf);
		prob_a = logistic_fun(mf);
		prob_b = 1.0 - prob_a;
		if(winner == a)
			coeff = prob_a;
		else
			coeff = -prob_b;


		if(grs.test_mask[triple_id] == FOR_VALIDATION)
		{
			realn_validate += 1.0;
			ll_validate += ((winner == a)? safe_log_logit(-mf) : safe_log_logit(mf));

			if((winner == a && prob_a >= 0.5) || (winner == b && prob_b > 0.5))
				correct_validate += 1.0;

			temp_pair.fst = a;
			temp_pair.snd = b;
			idx = find_gp_key(temp_pair, only_training_games, &whatever);
			if(idx < 0)
			{
				history_result.fst = 0;
				history_result.snd = 0;
			}
			else
				history_result = only_training_games.array[idx].val;

			if(((history_result.fst >= history_result.snd) && (winner == a)) || ((history_result.fst < history_result.snd) && (winner == b)))
				correct_validate_naive += 1.0;

			//if(history_result.fst == 0 || history_result.snd == 0)
			//{
				history_result.fst++;
				history_result.snd++;
			//}
			if(winner == a)
				ll_validate_naive += log((double)history_result.fst / ((double)(history_result.fst + history_result.snd)));
			else
				ll_validate_naive += log((double)history_result.snd / ((double)(history_result.fst + history_result.snd)));
		}

		if(grs.test_mask[triple_id] == FOR_TESTING)
		{
			realn_test += 1.0;
			ll_test += ((winner == a)? safe_log_logit(-mf) : safe_log_logit(mf));

			if((winner == a && prob_a >= 0.5) || (winner == b && prob_b > 0.5))
				correct_test += 1.0;

			temp_pair.fst = a;
			temp_pair.snd = b;
			idx = find_gp_key(temp_pair, only_training_games, &whatever);
			if(idx < 0)
			{
				history_result.fst = 0;
				history_result.snd = 0;
			}
			else
				history_result = only_training_games.array[idx].val;

			if(((history_result.fst >= history_result.snd) && (winner == a)) || ((history_result.fst < history_result.snd) && (winner == b)))
				correct_test_naive += 1.0;

			//if(history_result.fst == 0 || history_result.snd == 0)
			//{
				history_result.fst++;
				history_result.snd++;
			//}
			if(winner == a)
				ll_test_naive += log((double)history_result.fst / ((double)(history_result.fst + history_result.snd)));
			else
				ll_test_naive += log((double)history_result.snd / ((double)(history_result.fst + history_result.snd)));
		}
	}


	if(matchupmatfile[0] != '\0')
	{
		for(a = 0; a < k; a++)
		{
			for(b = a; b < k; b++)
			{
				if(a == b)
				{
					predict_matchup_mat[a][b] = 5.0;
					naive_matchup_mat[a][b] = 5.0;
				}
				else
				{
					mf = matchup_fun(main_embedding, a, b, myparas.modeltype);
					prob_a = logistic_fun(mf);
					predict_matchup_mat[a][b] = prob_a * 10.0;
					predict_matchup_mat[b][a] = 10.0 - predict_matchup_mat[a][b];

					temp_pair.fst = a;
					temp_pair.snd = b;
					idx = find_gp_key(temp_pair, only_training_games, &whatever);
					if(idx < 0)
					{
						history_result.fst = 0;
						history_result.snd = 0;
					}
					else
						history_result = only_training_games.array[idx].val;
					//if(history_result.fst == 0 || history_result.snd == 0)
					//{
						history_result.fst++;
						history_result.snd++;
					//}
					naive_matchup_mat[a][b] = 10.0 * (double)history_result.fst / ((double)(history_result.fst + history_result.snd));
					naive_matchup_mat[b][a] = 10.0 - naive_matchup_mat[a][b];
				}
			}
		}
		//print_mat(predict_matchup_mat, k, k);
		//putchar('\n');
		//print_mat(naive_matchup_mat, k, k);
	}

	//print_mat(predict_matchup_mat, k, k);
	//putchar('\n');
	//print_mat(naive_matchup_mat, k, k);








	//avg_ll = ll / realn;
	avg_ll_validate = ll_validate / realn_validate;
	avg_ll_validate_naive = ll_validate_naive / realn_validate;

	avg_ll_test = ll_test / realn_test;
	avg_ll_test_naive = ll_test_naive / realn_test;



	printf("\n---------- Final Result ----------\n");
	printf("\n---------- Validation ----------\n");

	printf("Avg validation log-likelihood: %f\n", avg_ll_validate);
	printf("Avg validation accuracy: %f\n", (correct_validate / realn_validate));
	printf("Avg naive validation log-likelihood: %f\n", avg_ll_validate_naive);
	printf("Avg naive validation accuracy: %f\n", (correct_validate_naive / realn_validate));

	printf("\n---------- Testing ----------\n");
	printf("Avg testing log-likelihood: %f\n", avg_ll_test);
	printf("Avg testing accuracy: %f\n", (correct_test / realn_test));
	printf("Avg naive testing log-likelihood: %f\n", avg_ll_test_naive);
	printf("Avg naive testing accuracy: %f\n", (correct_test_naive / realn_test));

	if(matchupmatfile[0] != '\0')
	{
		printf("\n---------- Reconstruction ----------\n");
		printf("Reconstruction norm diff: %f\n", mat_norm_diff(true_matchup_mat, predict_matchup_mat, k, k));
		printf("Reconstruction norm diff naive: %f\n", mat_norm_diff(true_matchup_mat, naive_matchup_mat, k, k));
		printf("Reconstruction error: %f\n", matchup_matrix_recover_error(true_matchup_mat, predict_matchup_mat, k));
		printf("Reconstruction error naive: %f\n", matchup_matrix_recover_error(true_matchup_mat, naive_matchup_mat, k));
	}

	//obj = avg_ll - myparas.lambda * (pow(frob_norm(main_embedding.tvecs, k, d), 2.0) + pow(frob_norm(main_embedding.hvecs, k, d), 2.0));
	//printf("Training obj function: %f\n", obj);
	//

	//printf("\n---------- Dumb Baseline ----------\n");
	//BModel bm =  train_baseline_model(&only_training_games, myparas, 0);
	//test_baseline_model(bm, grs, test_mask);
	//free_BModel(&bm);

	//print_gp_hash(only_training_games);
	
	//free(test_mask);
	free_game_embedding(last_embedding);
	free_game_embedding(best_embedding);
	free_game_embedding(ebd_grad);
	free_gp_hash(only_training_games);

	free(grad1);
	free(grad2);
	free(grad3);
	free(grad4);

	free(reg_grad1);
	free(reg_grad2);
	free(reg_grad3);
	free(reg_grad4);

	free(temp_d);
	free(game_distr);

	free(last_n_llhoods);

	if(matchupmatfile[0] != '\0')
	{
		Array2Dfree(true_matchup_mat, true_m, true_n);
		Array2Dfree(predict_matchup_mat, k, k);
		Array2Dfree(naive_matchup_mat, k, k);
	}
	//printf("%d\n", grs.with_mask);
	
	//print_int_vec(grs.test_mask, grs.num_games);
	return main_embedding;
}

//double* BTL_train_test(GRECORDS grs, PARAS myparas)
//{
//	int i;
//	int t;
//	//double* ranks = randvec(grs.num_players, 1.0);
//	double* ranks = linspace_vec(0.0, 1.0, grs.num_players);
//	double* ranks_g = (double*)malloc(grs.num_players * sizeof(double));
//	for(i = 0; i < grs.num_players; i++)
//		ranks_g[i] = 1.0;
//	int train_ratio = 5;
//	int* test_mask = (int*)calloc(grs.num_games, sizeof(int));
//	double prob_a;
//	double prob_b;
//	for(i = 0; i < grs.num_games; i++)
//		if(i % train_ratio == 0)
//			test_mask[i] = 1;
//	double train_ll, test_ll, train_n, test_n = 0.0;
//	double llhoodprev;
//	double eta =  myparas.eta;
//	double coeff;
//
//	double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
//	int llhood_track_index = 0;
//
//	for(i = 0; i < myparas.num_llhood_track; i++)
//		last_n_llhoods[i] = 1.0;
//
//	for(t = 0; t < 1e3; t++)
//	{
//		printf("Iteration %d\n", t + 1);
//		llhoodprev = train_ll;
//		train_ll = 0.0;
//		test_ll = 0.0;
//		train_n = 0.0;
//		test_n = 0.0;
//		for(i = 0; i < grs.num_games; i++)
//		{
//			prob_a = exp(safe_log_logit(- ranks[grs.all_games[i].pa] + ranks[grs.all_games[i].pb]));
//			//printf("%f\n", prob_a);
//			assert(prob_a >= 0.0 && prob_a <= 1.0);
//			prob_b = 1.0 - prob_a;
//			//printf("(%f, %f)\n", prob_a, prob_b);
//			if(test_mask[i] == 0)
//			{
//				//player a wins
//				if(grs.all_games[i].pa == grs.all_games[i].pw)
//					coeff = prob_b;
//				//player b wins
//				else
//					coeff = -prob_a;
//
//				ranks[grs.all_games[i].pa] += pow(ranks_g[grs.all_games[i].pa], -0.5) * coeff * eta;
//				ranks[grs.all_games[i].pb] -= pow(ranks_g[grs.all_games[i].pb], -0.5) * coeff * eta;
//				ranks_g[grs.all_games[i].pa] += pow(coeff, 2.0);
//				ranks_g[grs.all_games[i].pb] += pow(coeff, 2.0);
//
//				train_n += 1.0;
//				train_ll += (grs.all_games[i].pa == grs.all_games[i].pw)? log(prob_a) : log(prob_b);
//			}
//			else
//			{
//				test_n += 1.0;
//				test_ll += (grs.all_games[i].pa == grs.all_games[i].pw)? log(prob_a) : log(prob_b);
//			}
//		}
//
//		train_ll /= train_n;
//		test_ll /= test_n;
//		printf("Avg log-likelihood on train: %f\n", train_ll);
//		printf("Avg log-likelihood on test: %f\n", test_ll);
//
//		if(t > 0)
//		{
//			if(llhoodprev > train_ll)
//			{
//				eta /= myparas.beta;
//				printf("Reducing eta to: %f\n", eta);
//			}
//			else if(fabs((train_ll - llhoodprev) / train_ll) < 0.005)
//			{
//				eta *= myparas.alpha;
//				printf("Increasing eta to: %f\n", eta);
//			}
//		}
//		last_n_llhoods[llhood_track_index] = train_ll;
//		double min_llhood = DBL_MAX, max_llhood = -DBL_MAX, total_llhood = 0.0;
//		for(i = 0; i < myparas.num_llhood_track; i++) {
//			total_llhood += last_n_llhoods[i];
//			if (last_n_llhoods[i] < min_llhood)
//				min_llhood = last_n_llhoods[i];
//			if (last_n_llhoods[i] > max_llhood)
//				max_llhood = last_n_llhoods[i];
//		}
//		printf("min_obj %f, max_obj %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
//		if (myparas.num_llhood_track > 0
//				&& fabs(min_llhood - max_llhood) < myparas.eps 
//				&& t > myparas.num_llhood_track) {
//			break;
//		}
//		else if (myparas.num_llhood_track <= 0
//				&& train_ll >= llhoodprev
//				&& train_ll - llhoodprev < myparas.eps 
//				&& t > myparas.least_iter)
//			break;
//		// update llhood tracker index
//		llhood_track_index++;
//		llhood_track_index %= myparas.num_llhood_track;
//		putchar('\n');
//	}
//
//	//print_vec(ranks, grs.num_players);
//	free(test_mask);
//	free(ranks_g);
//	return ranks;
//
//}

//return -log(1 + exp(a))
double safe_log_logit(double a)
{
	if(a < 0.0)
		return -log(1.0 + exp(a));
	else
		return -a - log(1.0 + exp(-a));
}

int main(int argc, char* argv[])
{
	PARAS myparas = parse_paras(argc, argv, trainfile, embeddingfile);
	GRECORDS grs = read_game_records_data(trainfile, 0);
	//double* ranks = BTL_train_test(grs, myparas);
	GEMBEDDING ebd = game_embedding_train_test(grs, myparas);
	write_GameEmbedding_to_file(ebd, embeddingfile);
	free_game_embedding(ebd);
	//printf("out !!!\n");fflush(stdout);
	free_game_records_data(grs);
	//free(ranks);
	return 0;
}

PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile)
{
	//Default options
	
	PARAS myparas;
	myparas.d = 2;
	myparas.eps = 1e-4;
	myparas.eta = 1e-2;
	myparas.rankon = 1;
	myparas.lambda = 0.0;
	myparas.seed = 0;
	myparas.train_ratio = 2;
	myparas.max_iter = 10000;
	myparas.training_multiplier = 10;
	myparas.bomb_thresh = 100;
	myparas.alpha = 1.1;
	myparas.beta = 2.0;
	myparas.training_mode = 0;
	myparas.num_llhood_track = 10;
	myparas.regularization_type = 0;
	myparas.modeltype = 0;
	matchupmatfile[0] = '\0';
	myparas.eta_reduction_thresh = log(0.5);
	//myparas.do_normalization = 0;
	//myparas.method = 2;
	//myparas.d = 2;
	//myparas.ita = 0.01;
	//myparas.eps = 1e-5;
	//myparas.lambda = 1.0;
	//myparas.nu_multiplier = 1.0;
	//myparas.random_init = 1;
	//myparas.fast_collection = 0;
	//myparas.alpha = 1.1;
	//myparas.beta = 1.5;
	//myparas.least_iter = 250;
	//myparas.radius = 2;
	//myparas.regularization_type = 0; // default no regularization
	//myparas.stoc_grad = 1;
	//myparas.num_points = 1;
	//myparas.allow_self_transition = 1;
	//myparas.output_distr = 0;
	//myparas.grid_heuristic = 0;
	//myparas.regeneration_interval = 10;
	//myparas.bias_enabled = 0;
	//myparas.num_llhood_track = 10;
	//myparas.tagfile[0] = '\0';
	//myparas.tag_regularizer = 0;
	//myparas.hessian = 0;
	//myparas.landmark_heuristic = 0;
	//myparas.num_landmark = 0;
	//myparas.lowerbound_ratio = 0.3;
	//myparas.reboot_enabled = 0;
	//myparas.landmark_burnin_iter = 100;
	//myparas.use_hash_TTable = 1;
	//myparas.triple_dependency = 0;
	//myparas.angle_lambda = 1.0;
	//myparas.transition_range = 1;
	//myparas.num_threads = 0;
    //myparas.rand_option = 0;
    //myparas.init_file[0] ='\0';
    //myparas.ALL_candidate = 0;
	//myparas.candidate_mode = 1;
	//myparas.candidate_length_threshold = 0;
	//myparas.hedonic_enabled = 0;


	// don't use r, e, or n for anything which you plan to actually change.
	// (reserved in sge_unit.sh script)

	int i;
	for(i = 1; (i < argc) && argv[i][0] == '-'; i++)
	{
		switch(argv[i][1])
		{
			case 'd': i++; myparas.d = atoi(argv[i]); break;
			case 'e': i++; myparas.eps = atof(argv[i]); break;
			case 'i': i++; myparas.eta = atof(argv[i]); break;
			case 'r': i++; myparas.rankon = atoi(argv[i]); break;
			case 'l': i++; myparas.lambda = atof(argv[i]); break;
			case 'S': i++; myparas.seed = atof(argv[i]); break;
			case 'R': i++; myparas.train_ratio = atoi(argv[i]); break;
			case 'm': i++; myparas.max_iter = atoi(argv[i]); break;
			case 'u': i++; myparas.training_multiplier = atoi(argv[i]); break;
			case 'b': i++; myparas.bomb_thresh = atoi(argv[i]); break;
			case 'A': i++; myparas.alpha = atof(argv[i]); break;
			case 'B': i++; myparas.beta = atof(argv[i]); break;
			case 'T': i++; myparas.training_mode = atoi(argv[i]); break;
            case 'w': i++; myparas.num_llhood_track = atoi(argv[i]); break;
            case 't': i++; myparas.regularization_type = atoi(argv[i]); break;
            case 'M': i++; myparas.modeltype = atoi(argv[i]); break;
			case 'E': i++; strcpy(matchupmatfile, argv[i]); break;
			//case 'n': i++; myparas.do_normalization = atoi(argv[i]); break;
			//case 't': i++; myparas.method = atoi(argv[i]); break;
			//case 'r': i++; myparas.random_init = atoi(argv[i]); break;
			//case 'd': i++; myparas.d = atoi(argv[i]); break;
			//case 'i': i++; myparas.ita = atof(argv[i]); break;
			//case 'e': i++; myparas.eps = atof(argv[i]); break;
			//case 'l': i++; myparas.lambda = atof(argv[i]); break;
			//case 'f': i++; myparas.fast_collection= atoi(argv[i]); break;
			//case 's': i++; myparas.radius= atoi(argv[i]); break;
			//case 'a': i++; myparas.alpha = atof(argv[i]); break;
			//case 'b': i++; myparas.beta = atof(argv[i]); break;
			//case 'g':
			//		  i++;
			//		  if (argv[i][1] == '\0') {
			//			  myparas.regularization_type = atoi(argv[i]);
			//			  myparas.tag_regularizer = atoi(argv[i]);
			//			  printf("Both regularizers set to %d\n", myparas.regularization_type);
			//		  }
			//		  else {
			//			  char first_reg[2] = "\0\0";
			//			  char second_reg[2] = "\0\0";
			//			  first_reg[0] = argv[i][0];
			//			  second_reg[0] = argv[i][1];

            //              myparas.regularization_type = atoi(first_reg);
            //              myparas.tag_regularizer = atoi(second_reg);
            //              printf("Song regularizer set to %d\n", myparas.regularization_type);
            //              printf("Tag regularizer set to %d\n", myparas.tag_regularizer);
            //          }                
            //          break;
            //case 'h': i++; myparas.grid_heuristic = atoi(argv[i]); break;
            //case 'm': i++; myparas.landmark_heuristic = atoi(argv[i]); break;
            //case 'p': i++; myparas.bias_enabled = atoi(argv[i]); break;
            //case 'w': i++; myparas.num_llhood_track = atoi(argv[i]); break;
            //case 'c': i++; myparas.hessian = atoi(argv[i]); break;
            //case 'x': i++; strcpy(myparas.tagfile, argv[i]); break;
            //case 'q': i++; myparas.num_landmark = atoi(argv[i]); break;
            //case 'y': i++; myparas.lowerbound_ratio = atof(argv[i]); break;
            //case 'o': i++; myparas.reboot_enabled = atoi(argv[i]); break;
            //case '0': i++; myparas.landmark_burnin_iter = atoi(argv[i]); break;
            //case 'u': i++; myparas.nu_multiplier = atof(argv[i]); break;
            //case 'k': i++; myparas.use_hash_TTable = atoi(argv[i]); break;
            //case 'T': i++; myparas.triple_dependency = atoi(argv[i]); break;
            //case 'L': i++; myparas.angle_lambda = atof(argv[i]); break;
            //case 'N': i++; myparas.transition_range = atoi(argv[i]); break;
            //case 'D': i++; myparas.num_threads = atoi(argv[i]); break;
            //case '9': i++; myparas.rand_option = atoi(argv[i]); break;
            //case 'I': i++; strcpy(myparas.init_file, argv[i]); break;
            //case 'A': i++; myparas.ALL_candidate = atoi(argv[i]); break;
            //case 'M': i++; myparas.candidate_mode = atoi(argv[i]); break;
            //case 'S': i++; myparas.candidate_length_threshold = atoi(argv[i]); break;
            //case 'H': i++; myparas.hedonic_enabled = atoi(argv[i]); break;
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

void shuffle_match_array(MATCH* array, size_t n)
{
	if(n > 1) 
	{
		size_t i;
		for (i = 0; i < n - 1; i++) 
		{
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			MATCH t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}

void shuffle_gele_array(GELE* array, size_t n)
{
	if(n > 1) 
	{
		size_t i;
		for (i = 0; i < n - 1; i++) 
		{
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			GELE t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}


int random_in_range (unsigned int min_val, unsigned int max_val)
{
	int base_random = rand(); /* in [0, RAND_MAX] */
	if (RAND_MAX == base_random) return random_in_range(min_val, max_val);
	/* now guaranteed to be in [0, RAND_MAX) */
	int range       = max_val - min_val,
		remainder   = RAND_MAX % range,
		bucket      = RAND_MAX / range;
	/* There are range buckets, plus one smaller interval
	 *      within remainder of RAND_MAX */
	if (base_random < RAND_MAX - remainder) {
		return min_val + base_random/bucket;
	} else {
		return random_in_range (min_val, max_val);
	}
}

double matchup_fun(GEMBEDDING gebd, int a, int b, int modeltype)
{
	double mf;
	if(gebd.d == 0) 
		mf = 0.0;
	else if(modeltype == 0)
		mf = vec_diff(gebd.hvecs[b], gebd.tvecs[a], gebd.d) - vec_diff(gebd.hvecs[a], gebd.tvecs[b], gebd.d);
	else if(modeltype == 1)
		mf = log(vec_diff(gebd.hvecs[b], gebd.tvecs[a], gebd.d)) - log(vec_diff(gebd.hvecs[a], gebd.tvecs[b], gebd.d));
	else
		mf = innerprod(gebd.hvecs[a], gebd.tvecs[b], gebd.d) - innerprod(gebd.tvecs[a], gebd.hvecs[b], gebd.d);
	if(gebd.rankon)
		mf = mf + gebd.ranks[a] - gebd.ranks[b];
	return mf;
}


double vec_diff(const double* x, const double* y, int length)
{
	int i;
	double temp = 0.0;
	for(i = 0; i < length; i++)
		temp += pow(x[i] - y[i], 2);
	return temp;
}

double logistic_fun(double a)
{
	return 1.0 / (1.0 + exp(-a));
}

void write_GameEmbedding_to_file(GEMBEDDING ge, char* filename)
{
	FILE*  fp = fopen(filename, "w");
	fprintf(fp, "numplayers %d\n", ge.k);
	fprintf(fp, "d %d\n", ge.d);
	fprintf(fp, "rankon %d\n", ge.rankon);
	fprintf(fp, "modeltype %d\n", ge.modeltype);
	fprintf(fp, "tvecs\n");
	fprint_mat(ge.tvecs, ge.k, ge.d, fp);
	fprintf(fp, "hvecs\n");
	fprint_mat(ge.hvecs, ge.k, ge.d, fp);
	if(ge.rankon)
	{
		fprintf(fp, "ranks\n");
		fprint_vec(ge.ranks, ge.k, fp);
	}
	fclose(fp);
}





//Basline related items
void init_BModel(BModel* p_bm, int num_players, int d, GPHASH* aggregated_games)
{
	(p_bm -> num_players) = num_players;
	(p_bm -> d) = d;
	(p_bm -> sigma) = create_gp_hash(aggregated_games -> length, num_players);
	int i;
	for(i = 0; i < aggregated_games -> length; i++)
	{
		(p_bm -> sigma).array[i].key = (aggregated_games -> array)[i].key;
		if((aggregated_games -> array)[i].val.fst >= (aggregated_games -> array)[i].val.snd)

			(p_bm -> sigma).array[i].val.fst = 1;
		else
			(p_bm -> sigma).array[i].val.fst = -1;
		(p_bm -> sigma).array[i].val.snd = 0;
	}
	(p_bm -> X) = randarray(num_players, d, 1.0);
}

void free_BModel(BModel* p_bm)
{
	Array2Dfree(p_bm -> X, p_bm -> num_players, p_bm -> d);
	free_gp_hash(p_bm -> sigma);
}

void copy_BModel(BModel* src, BModel* dest)
{
	dest -> num_players = src -> num_players;
	dest -> d = src -> d;
	Array2Dcopy(src -> X, dest -> X, src -> num_players, src -> d);
	copy_gp_hash(&(src -> sigma), &(dest -> sigma));
}

double matchup_fun_bm(BModel bm, int a, int b, double* sigma_ab)
{
	//print_mat(bm.X, bm.num_players, bm.d);
	GPAIR temp_gp;
	temp_gp.fst = a;
	temp_gp.snd = b;
	int no_use;
	int idx =  find_gp_key(temp_gp, bm.sigma, &no_use);
	*sigma_ab = 1.0; 

	if(idx >= 0)
		(*sigma_ab) = (double)(bm.sigma.array)[idx].val.fst;

	assert((*sigma_ab) == 1.0 || (*sigma_ab) == -1.0);
	//return (*sigma_ab) * pow(vec_diff_norm2(bm.X[a], bm.X[b], bm.d), 2.0);
	return (*sigma_ab) * vec_diff_norm2(bm.X[a], bm.X[b], bm.d);
}

BModel train_baseline_model(GPHASH* aggregated_games, PARAS myparas, int verbose)
{
	BModel bm, bm_best, bm_last;
	init_BModel(&bm, aggregated_games -> num_players, myparas.d, aggregated_games);
	init_BModel(&bm_best, aggregated_games -> num_players, myparas.d, aggregated_games);
	init_BModel(&bm_last, aggregated_games -> num_players, myparas.d, aggregated_games);
	int iteration_count = 0;
	double llhood, llhood_prev, realn, avg_ll = 0.0;
	double best_ll = -DBL_MAX;
	GELE game_ele;
	int a, b, na, nb;
	double sigma_ab;
	double coeff;
	double mf, prob_a, prob_b;
	double* temp_d = (double*)malloc(myparas.d * sizeof(double));
	int time_bomb;

	GELE space_free[aggregated_games -> num_used];
	int t = 0;
	int i;
	int triple_id;
	for(i = 0; i < aggregated_games -> length; i++)
		if(!gele_is_empty((aggregated_games -> array)[i]))
			space_free[t++] = (aggregated_games -> array)[i];

	while(iteration_count++ < myparas.max_iter)
	{
		copy_BModel(&bm, &bm_last);
		llhood_prev = llhood;
		llhood = 0.0;
		realn = 0.0;
		shuffle_gele_array(space_free, (size_t)(aggregated_games -> num_used));
		if(verbose)
			printf("Baseline iteration %d\n", iteration_count);
		for(triple_id = 0; triple_id < aggregated_games -> num_used; triple_id++)
		{
			game_ele = space_free[triple_id]; 
			a = game_ele.key.fst;
			b = game_ele.key.snd;
			na = game_ele.val.fst;
			nb = game_ele.val.snd;
			//printf("(%d, %d, %d, %d)\n", a, b, na, nb);
			assert(a < b);
			mf =  matchup_fun_bm(bm, a, b, &sigma_ab);
			//printf("mf: %f\n", mf);
			//printf("really: %f\n", 1.0 / (1.0 + exp(-mf)));
			prob_a = logistic_fun(mf);
			prob_b = 1.0 - prob_a;
			//printf("pa, pb: (%f, %f)\n", prob_a, prob_b);

			coeff = (na * prob_b - nb * prob_a);

			memcpy(temp_d, bm.X[a], myparas.d);
			add_vec(temp_d, bm.X[b], myparas.d, -1.0);
			scale_vec(temp_d, myparas.d, sigma_ab / vec_diff_norm2(bm.X[a], bm.X[b], myparas.d));

			add_vec(bm.X[a], temp_d, myparas.d, myparas.eta * coeff );
			add_vec(bm.X[b], temp_d, myparas.d, -1.0 * myparas.eta * coeff);

			realn += (na + nb);
			llhood += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
		}
		avg_ll = llhood / realn;
		//printf("realn: %f\n", realn);
		if(verbose)
			printf("Baseline Avg training log-likelihood: %f\n", avg_ll);
		if(isnan(avg_ll))
			break;

		if(avg_ll > best_ll) 
		{
			best_ll = avg_ll;
			copy_BModel(&bm_last, &bm_best);
			time_bomb = 0;
		}
		else
			time_bomb++;
		if(time_bomb > myparas.bomb_thresh)
			break;
	}
	printf("Baseline best training log-likelihood: %f\n", best_ll);
	copy_BModel(&bm_best, &bm);
	free_BModel(&bm_best);
	free_BModel(&bm_last);
	free(temp_d);
	return bm;
}

void test_baseline_model(BModel bm, GRECORDS grs, int* test_mask)
{
	int i, triple_id;
	int a, b, w,  na, nb;
	double prob_a, prob_b, mf, coeff, sigma_ab;
	double ll_validate, ll_test, realn_validate, realn_test, correct_validate, correct_test = 0.0;
	for(triple_id = 0; triple_id < grs.num_games; triple_id++)
	{
		a = grs.all_games[triple_id].pa;
		b = grs.all_games[triple_id].pb;
		w = grs.all_games[triple_id].pw;
		if(a == w)
		{
			na = 1;
			nb = 0;
		}
		else
		{
			na = 0;
			nb = 1;
		}
		mf =  matchup_fun_bm(bm, a, b, &sigma_ab);
		prob_a = logistic_fun(mf);
		prob_b = 1.0 - prob_a;
		coeff = (na * prob_b - nb * prob_a);
		if(test_mask[triple_id] == FOR_VALIDATION)
		{
			realn_validate += (na + nb);
			ll_validate += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
			if((w == a && prob_a >= 0.5) || (w == b && prob_b > 0.5))
				correct_validate++;
		}
		if(test_mask[triple_id] == FOR_TESTING)
		{
			realn_test += (na + nb);
			ll_test += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
			if((w == a && prob_a >= 0.5) || (w == b && prob_b > 0.5))
				correct_test++;
		}
	}
	printf("Avg baseline validation log-likelihood: %f\n", ll_validate / realn_validate);
	printf("Avg baseline validation accuracy: %f\n", (correct_validate / realn_validate));
	printf("Avg baseline testing log-likelihood: %f\n", ll_test / realn_test);
	printf("Avg baseline testing accuracy: %f\n", (correct_test / realn_test));

}

void compute_ll_obj(double* avg_ll, double* obj, int num_used, GELE* training_games_aggregated, GEMBEDDING main_embedding, PARAS myparas)
{
	double ll = 0.0;
	double realn = 0.0;
	double mf, prob_a, prob_b;
	int triple_id, a, b, na, nb, i, j;
	GELE game_ele;

	for(triple_id = 0; triple_id < num_used; triple_id++)
	{
		game_ele = training_games_aggregated[triple_id]; 
		a = game_ele.key.fst;
		b = game_ele.key.snd;
		na = game_ele.val.fst;
		nb = game_ele.val.snd;

		mf = matchup_fun(main_embedding, a, b, myparas.modeltype);
		prob_a = logistic_fun(mf);
		prob_b = 1.0 - prob_a;

		realn += (na + nb);
		ll += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
	}
	*avg_ll = ll / realn;

	if(myparas.regularization_type == 0)
		*obj = ll - myparas.lambda * (pow(frob_norm(main_embedding.tvecs, main_embedding.k, main_embedding.d), 2.0) + pow(frob_norm(main_embedding.hvecs, main_embedding.k, main_embedding.d), 2.0));
	else if(myparas.regularization_type == 1)
	{
		*obj = ll;
		for(i = 0; i < main_embedding.k; i++)
			for(j = 0; j < main_embedding.d; j++)
				*obj -= myparas.lambda * pow(main_embedding.tvecs[i][j] - main_embedding.hvecs[i][j], 2);
	}

	else if(myparas.regularization_type == 2)
	{
		*obj = ll - myparas.lambda * (pow(frob_norm(main_embedding.tvecs, main_embedding.k, main_embedding.d), 2.0) + pow(frob_norm(main_embedding.hvecs, main_embedding.k, main_embedding.d), 2.0));
		for(i = 0; i < main_embedding.k; i++)
			for(j = 0; j < main_embedding.d; j++)
				*obj -= myparas.lambda * pow(main_embedding.tvecs[i][j] - main_embedding.hvecs[i][j], 2);
	}

	*obj /= realn;
}

double matchup_matrix_recover_error(double** true_matchup_mat, double** predict_matchup_mat, int k)
{
	int i, j;
	int num_error = 0;
	int num_count = 0;
	for(i = 0; i < k; i++)
	{
		for(j = 0; j < k; j++)
		{
			if(true_matchup_mat[i][j] != 5.0 )
			{
				num_count++;
				if(sign_fun(true_matchup_mat[i][j] - 5.0) != sign_fun(predict_matchup_mat[i][j] - 5.0))
					num_error++;
			}
		}
	}
	return (double)num_error / num_count;
}

int sign_fun(double a)
{
	if(a > 0.0)
		return 1;
	else if(a < 0.0)
		return -1;
	else
		return 0;
}
