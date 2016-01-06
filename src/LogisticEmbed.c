#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include "LogisticEmbed.h"
#include "LogisticEmbed_common.h"
#include "PairHashTable.h"
#include "TripleHashTable.h"
#include "TransitionTable.h"
#include "GridTable.h"
#include "LandmarkTable.h"
#include "ThreadJob.h"

char trainfile[200];
char embeddingfile[200];
char songdistrfile[200];
char transdistrfile[200];


int main(int argc, char* argv[])
{
    PARAS myparas = parse_paras(argc, argv, trainfile, embeddingfile);
    PDATA pd = read_playlists_data(trainfile, 1);

    double** X;



    int l;
    //Experiental code with triple dependency
    if(myparas.triple_dependency == 1)
    {
        X = logistic_embed_triple_dependency(pd, myparas, embeddingfile, &l);
    }
    //Test multithreaded alg
    else if(myparas.num_threads >= 1)
    {
        X = logistic_embed_sp_mt(pd, myparas, embeddingfile, &l);
    }
    else if(myparas.method == 1)
    {
        myparas.num_points = 2;
        myparas.stoc_grad = 0;
        if (myparas.bias_enabled) {
            printf("ERROR: bias not properly implemented for use without stochastic gradient yet, exiting.\n");
            exit(1);
        }
        X = logistic_embed_allinone(pd, myparas, embeddingfile, &l);
    }
    else if(myparas.method == 2)
    {
        myparas.num_points = 1;
        X = logistic_embed_allinone(pd, myparas, embeddingfile, &l);
    }
    else if(myparas.method == 3)
    {
        myparas.num_points = 1;
        myparas.allow_self_transition = 0;
        X = logistic_embed_allinone(pd, myparas, embeddingfile, &l);
    }
    else if(myparas.method == 4)
    {
        myparas.num_points = 2;
        myparas.stoc_grad = 1;
        X = logistic_embed_allinone(pd, myparas, embeddingfile, &l);
    }
    else if(myparas.method == 5)
    {
        myparas.num_points = 3;
        X = logistic_embed_allinone(pd, myparas, embeddingfile, &l);
    }

    Array2Dfree(X, l, myparas.d);

    free_playlists_data(pd);
    return 0;
}

double** logistic_embed_allinone(PDATA pd, PARAS myparas, char* embedding_file, int* l)
{
    if(myparas.num_points != 1 && myparas.num_points != 2 && myparas.num_points != 3)
    {
        printf("The model must use 1, 2, or 3 points per song.\n");
        exit(1);
    }

    int myfast = 0, mygrid = 0, mylm = 0;
    if(myparas.fast_collection)
        myfast = 1;
    if(myparas.grid_heuristic)
        mygrid = 1;
    if(myparas.landmark_heuristic)
        mylm = 1;
    if(myfast + mygrid + mylm > 1)
    {
        printf("Error: should choose only one training heuristic\n");
        exit(1);
    }

    if(myparas.transition_range <= 0)
    {
        printf("Transition range cannot be less than one (Currently %d).\n", myparas.transition_range);
        exit(1);
    }

    int using_tags = (myparas.tagfile[0] != '\0');
    TDATA td;
    if(using_tags)
        td = read_tag_data(myparas.tagfile);
    double** X;
    double* bias_terms = 0;
    int n; //Number of training transitions
    int i;
    int j;
    int s;
    int t;
    int fr;
    int to;
    int d = myparas.d;
    int k = pd.num_songs;
    double z;
    double eta;
    if (myparas.ita > 0)
        eta = myparas.ita;
    else
        eta = k;
    n = 0;
    for(i = 0; i < pd.num_playlists; i ++)
        if(pd.playlists_length[i] > 0)
            n += pd.playlists_length[i] - 1;
    printf("Altogether %d transitions.\n", n);fflush(stdout);



    int m;
    if(using_tags)
        m = td.num_tags;

    if(!using_tags)
    {
        *l = k * myparas.num_points;
        if (myparas.random_init)
            X = randarray(k * myparas.num_points, d, 1.0);
        else
            X = zerosarray(k * myparas.num_points, d);
        if (myparas.bias_enabled)
            bias_terms = calloc(k, sizeof(double));
    }
    else
    {
        *l = (k + m) * myparas.num_points;
        if (myparas.random_init)
            X = randarray((k + m) * myparas.num_points, d, 1.0);
        else
            X = zerosarray((k + m) * myparas.num_points, d);
        if (myparas.bias_enabled)
            bias_terms = calloc(k, sizeof(double));
    }

    double** cdx;
    if(!using_tags)
        cdx = zerosarray(k * myparas.num_points, d);
    else
        cdx = zerosarray((k + m) * myparas.num_points, d);



    double** reg_x = 0;
    double* bias_grad = 0;
    if (myparas.bias_enabled)
        bias_grad = (double*)calloc(k, sizeof(double));
    if (myparas.regularization_type > 0 || myparas.tag_regularizer > 0)
        if (!using_tags)
            reg_x = zerosarray(k * myparas.num_points, d);
        else
            reg_x = zerosarray((k + m) * myparas.num_points, d);
    double* dufr = (double*)calloc(d, sizeof(double));
    double* tempd = (double*)calloc(d, sizeof(double));
    double* mid_dufr = 0;
    double* mid_tempd = 0;
    if (myparas.num_points == 3) {
        mid_dufr = (double*)calloc(d, sizeof(double));
        mid_tempd = (double*)calloc(d, sizeof(double));
    }
    double llhood = 0.0;
    double llhoodprev;
    double realn;
    double reg_objective = 0.0;

    double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
    int llhood_track_index = 0;

    for (i = 0; i < myparas.num_llhood_track; i++)
        last_n_llhoods[i] = 1.0;

    //double** tcount = zerosarray(k, k);
    PHASH* tcount;
    double** tcount_full;

    HELEM temp_elem;
    TPAIR temp_pair;
    int idx;
    double temp_val;

    time_t start_time;
    clock_t start_clock, temp_clock;



    if(myparas.use_hash_TTable)
    {
        tcount = create_empty_hash(2 * myparas.transition_range  * n, k);

        printf("Transition matrix created.\n");
        for(i = 0; i < pd.num_playlists; i ++)
        {
            if(pd.playlists_length[i] > 1)
            {
                for(j = 0; j < pd.playlists_length[i] - 1; j++)
                {
                    for(t = 1; t <= myparas.transition_range; t++)
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

        //Sanity check
        int temp_int = 0;
        for(i = 0; i < tcount -> length; i++)
        {
            if(!is_null_entry((tcount -> p)[i].key))
                temp_int++;
        } 
        printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(k * k)));
        printf("Transition matrix initialized.\n");
    }
    else
    {
        tcount_full = zerosarray(k, k);
        printf("Full transition matrix created.\n");
        for(i = 0; i < pd.num_playlists; i ++)
        {
            if(pd.playlists_length[i] > 1)
            {
                for(j = 0; j < pd.playlists_length[i] - 1; j++)
                {
                    for(t = 1; t <=myparas.transition_range; t++)
                    {
                        if(j + t < pd.playlists_length[i])
                        {
                            fr = pd.playlists[i][j];
                            to = pd.playlists[i][j + t];
                            tcount_full[fr][to] += 1.0 / (double) t;
                        }
                    }
                }
            }
        }
        int temp_int = 0;
        for(i = 0; i < k; i++)
            for(j = 0; j < k; j++)
                if(tcount_full[i][j] > 0.0)
                    temp_int++;
        printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(k * k)));
        printf("Full transition matrix initialized.\n");
    }



    TRANSITIONTABLE ttable;
    TRANSITIONTABLE BFStable;

    //Even for the grid heuristic, we still need to add the radius one neighbors to our list
    if(myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
    {
        init_transition_table(&ttable, k);
        printf("Transition table created.\n");
        for(i = 0; i < pd.num_playlists; i++)
        {
            if(pd.playlists_length[i] > 1)
            {
                for(j = 0; j < pd.playlists_length[i] - 1; j++)
                {
                    fr = pd.playlists[i][j];
                    to = pd.playlists[i][j + 1];
                    if(!exist_in_transition_table(&ttable, fr, to))
                        insert_in_transition_table(&ttable, fr, to);
                }
            }
        }
        printf("Transition table initialized.\n");
        if(myparas.fast_collection)
        {
            start_time = time(NULL);
            BFS_on_transition_table(&ttable, &BFStable, myparas.radius);
            printf("BFS table built. It took %d seconds.\n", (int)(time(NULL) - start_time));
        }
    }

    //Add random songs to the collection
    if(myparas.fast_collection == 2)
    {
        start_time = time(NULL);
        printf("Augmenting the collection with random songs.\n");
        srand(time(NULL));
        for(i = 0; i < k; i++)
        {
            int l = BFStable.parray[i].length;
            //printf("Length before random augmentation: %d\n", l);

            if(l >= k / 2)
            {
                //printf("The number of songs in the pruning collection is already bigger than half of all songs.\n");
                //printf("Please switch to the method without pruning.\n");
                continue;
                //exit(1);
            }

            while(1)
            {
                int temp_int = rand() % k;
                if(!exist_in_transition_table(&BFStable, i, temp_int))
                {
                    insert_in_transition_table(&BFStable, i, temp_int);
                    if(BFStable.parray[i].length == 2 * l)
                        break;

                }
            }
            //printf("Length after random augmentation: %d\n", BFStable.parray[i].length);

        }
        printf("Done. It took %d seconds.\n", (int)(time(NULL) - start_time));
    }






    FILE* song_distr_file;
    FILE* trans_distr_file;
    int write_distr_files;
    double* song_sep_ll;

    if(myparas.output_distr)
        printf("Output likelihood distribution file turned on.\n");

    // initialized masks for regularization computation. Will only be modified
    // after this if stochastic gradient descent is used.
    int* reg_upper = (int*) calloc(myparas.num_points, sizeof(int));
    int* reg_lower = (int*) calloc(myparas.num_points, sizeof(int));
    int* t_reg_upper = (int*) calloc(myparas.num_points, sizeof(int));
    int* t_reg_lower = (int*) calloc(myparas.num_points, sizeof(int));

    for (i = 0; i < myparas.num_points; i++) {
        reg_lower[i] = 0;
        reg_upper[i] = k;
        t_reg_lower[i] = 0;
        t_reg_upper[i] = m;
    }

    double* d_hess_fr = 0;
    double* d_hess_to = 0;
    double** hessian = 0;
    double exp_z_fr = 0;
    if (myparas.hessian) {
        d_hess_fr = (double*) calloc(d, sizeof(double));
        d_hess_to = (double*) calloc(d, sizeof(double));
        hessian = zerosarray(k * myparas.num_points, d);
    } 

    GRIDTABLE gt;
    if(myparas.grid_heuristic)
        printf("Using grid heuristic with radius %d.\n", myparas.radius);
    LMTABLE lmt;
    if(myparas.landmark_heuristic)
    {
        if(myparas.num_landmark == 0)
            myparas.num_landmark = (int)((double)k / 50.0);
        printf("Using landmark heuristic.\n");
        printf("Number of landmarks: %d\n", myparas.num_landmark);
        //printf("Number of nearest landmarks: %d\n", myparas.radius);
        printf("Ratio of songs that are from the landmarks: %f\n", myparas.lowerbound_ratio);
    }

    double** realX;
    if(using_tags)
    {
        realX = zerosarray(k * myparas.num_points, d);
        calculate_realX(X, realX, td, k, m, d, myparas.num_points);
    }
    else
        realX = X;

    int optimization_rebooted = 0;

    for(t = 0; t < 100000; t++)
    {
        if(myparas.grid_heuristic && t % myparas.regeneration_interval == 0)
        {
            printf("Generating new grid table.\n");
            temp_clock = clock();
            Init_Grid_Table(&gt, realX + (myparas.num_points - 1) * k, k, d, ceil(pow(k / 10, 1.0 / ((float)d))));
            printf("Buiding hash grid took %f seconds of cpu time\n", ((float)clock() - (float)temp_clock) / CLOCKS_PER_SEC);fflush(stdout);
            printf("%d segements on each axis\n", gt.num_grids_per_axis);

        }

        if(myparas.landmark_heuristic && t % myparas.regeneration_interval == 0 && t <= myparas.landmark_burnin_iter)
        {
            printf("Generating new landmark table.\n");
            temp_clock = clock();

            if(myparas.landmark_heuristic == 1)
                Init_LMTable(&lmt, realX + (myparas.num_points - 1) * k, k, d, myparas.num_landmark, 1); 
            else if(myparas.landmark_heuristic == 2)
                Init_LMTable_kmeans(&lmt, realX + (myparas.num_points - 1) * k, k, d, 5); 
            else
            {
                printf("Invalid landmark heuristic parameter.\n");
                exit(1);
            }

            printf("Buiding landmark table took %f seconds of cpu time\n", ((float)clock() - (float)temp_clock) / CLOCKS_PER_SEC);fflush(stdout);
        }

        if(myparas.landmark_heuristic  && t > myparas.landmark_burnin_iter)
            printf("Landmark table held still.\n");

        write_distr_files = (myparas.output_distr) && (t > 0) && (t % 10 == 0);

        if(write_distr_files)
        {
            song_distr_file = fopen(songdistrfile, "w");
            trans_distr_file = fopen(transdistrfile, "w");
            song_sep_ll = (double*)calloc(k, sizeof(double));
        }
        start_time = time(NULL);
        start_clock = clock();
        printf("Iteration %d\n", t + 1);fflush(stdout);
        llhoodprev = llhood;
        llhood = 0.0;
        realn = 0.0;

        if(!myparas.stoc_grad)
        {
            if(!using_tags)
                for(i = 0; i < k * myparas.num_points; i++)
                    memset(cdx[i], 0, d * sizeof(double));
            else
            {
                for(i = 0; i < (k + m) * myparas.num_points; i++)
                    memset(cdx[i], 0, d * sizeof(double));
                calculate_realX(X, realX, td, k, m, d, myparas.num_points);
            }
        }

        int ttable_count;
        int wtable_count;

        if(myparas.grid_heuristic || myparas.landmark_heuristic)
        {
            ttable_count = 0;
            wtable_count = 0;
        }

        for(fr = 0; fr < k; fr++)
        {

            if(myparas.stoc_grad)
            {
                if(!using_tags)
                    for(i = 0; i < k * myparas.num_points; i++)
                        memset(cdx[i], 0, d * sizeof(double));
                else
                {
                    for(i = 0; i < (k + m) * myparas.num_points; i++)
                        memset(cdx[i], 0, d * sizeof(double));
                    calculate_realX(X, realX, td, k, m, d, myparas.num_points);
                }
                if (myparas.hessian) {
                    for (i = 0; i < k * myparas.num_points; i++)
                        memset(hessian[i], 0, d * sizeof(double));
                    memset(d_hess_fr, 0, d * sizeof(double));
                    memset(d_hess_to, 0, d * sizeof(double));
                }
                if (myparas.bias_enabled)
                    // reset the bias gradients
                    memset(bias_grad, 0, k * sizeof(double));
            }
            int collection_size;
            int* collection_idx;

            int size_ttable;
            int* idx_ttable;
            int size_gtable;
            int* idx_gtable;


            if(myparas.fast_collection)
            {
                collection_size = (BFStable.parray)[fr].length;
                if(collection_size == 0)
                    continue;
                collection_idx = (int*)malloc(collection_size * sizeof(int));
                LINKEDELEM* tempp = (BFStable.parray)[fr].head;
                for(i = 0; i < collection_size; i++)
                {
                    collection_idx[i] = tempp -> idx; 
                    tempp = tempp -> pnext;
                }
            }
            else if(myparas.grid_heuristic || myparas.landmark_heuristic)
            {
                size_ttable = (ttable.parray)[fr].length;
                if(size_ttable == 0)
                    continue;
                idx_ttable = (int*)malloc(size_ttable * sizeof(int));
                LINKEDELEM* tempp = (ttable.parray)[fr].head;
                for(i = 0; i < size_ttable; i++)
                {
                    idx_ttable[i] = tempp -> idx; 
                    tempp = tempp -> pnext;
                }

                if(myparas.grid_heuristic)
                {
                    idx_gtable = get_nearby_indices(&gt, realX[(myparas.num_points - 1) * k + fr], &size_gtable, myparas.radius);

                    //printf("ratio t/g: %f\n", (float)size_ttable / (float)size_gtable);
                }
                if(myparas.landmark_heuristic)
                {
                    //idx_gtable = get_neaby_songs_by_landmark(lmt, myparas.radius, &size_gtable, realX + (myparas.num_points - 1) * k, realX[(myparas.num_points - 1) * k + fr], k, d);
                    idx_gtable = get_neaby_songs_by_landmark_with_lowerbound(lmt, (int)(((double)k) * myparas.lowerbound_ratio), &size_gtable, realX + (myparas.num_points - 1) * k, realX[(myparas.num_points - 1) * k + fr], k, d);
                }
                collection_idx = merge_two_lists(&collection_size, idx_ttable, size_ttable, idx_gtable, size_gtable);
                free(idx_ttable);
                free(idx_gtable);
                ttable_count += size_ttable;
                wtable_count += collection_size;
            }
            else
                collection_size = k;
            double** delta = zerosarray(collection_size, d);
            double** dv = zerosarray(collection_size, d);
            double* p = (double*)calloc(collection_size, sizeof(double));
            double* tempk = (double*)calloc(collection_size, sizeof(double));
            double** tempkd = zerosarray(collection_size, d);
            // for midpoint method
            // this array will contain the differences (x_a - x_s)
            double** mid_delta = 0;
            // to contain the component-wise squares of the mid_delta vectors
            double** mid_tempkd = 0;
            // to contain vector products -(x_a - x_s)^2
            double* mid_p = 0;
            // to function like ddv but for mid-point coordinate updates. Will
            // contain 2exp(-(u_a - v_s)^2 - (x_a - x_s)^2) * (x_a - x_s)
            //       / sum over s in S of exp(-(u_a - v_s)^2 - (x_a - x_2)^2)
            // then we can set dl(a,b)/dx_p to -2delta(a,p)(x_a - x_b)
            // + 2delta(a,p)(sum over s in S of mid_ddv[s])
            //
            // and dl(a,b)/dx_q to -2delta(b,q)(x_a - x_b)
            // + 2*mid_ddv[q]
            double** mid_dv = 0;

            if (myparas.num_points > 2) {
                mid_delta = zerosarray(collection_size, d);
                mid_tempkd = zerosarray(collection_size, d);
                mid_p = (double*)calloc(collection_size, sizeof(double));
                mid_dv = zerosarray(collection_size, d);
            }    

            for(j = 0; j < collection_size; j++)
            {
                for(i = 0; i < d; i++)
                {
                    if(myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
                        delta[j][i] = realX[fr][i] - realX[(myparas.num_points - 1) * k + collection_idx[j]][i];
                    else
                        delta[j][i] = realX[fr][i] - realX[(myparas.num_points - 1) * k + j][i];

                }
            }
            if (myparas.num_points == 3) {
                for(j = 0; j < collection_size; j++)
                {
                    for(i = 0; i < d; i++)
                    {
                        if(myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic) {
                            mid_delta[j][i] = realX[k + fr][i] - realX[k + collection_idx[j]][i];
                        }
                        else {
                            mid_delta[j][i] = realX[k + fr][i] - realX[k + j][i];
                        }
                    }
                }
                mat_mult(mid_delta, mid_delta, mid_tempkd, collection_size, d);
                sum_along_direct(mid_tempkd, mid_p, collection_size, d, 1);
                add_vec(p, mid_p, collection_size, -1.0);
                // p isn't actually getting modified by the above line since
                // it is overwritten altogether by sum_along_direct in a
                // couple of lines
            }

            mat_mult(delta, delta, tempkd, collection_size, d);
            // tempkd[i][j] = delta(fr,i)[j]^2
            scale_mat(tempkd, collection_size, d, -1.0);
            sum_along_direct(tempkd, p, collection_size, d, 1);
            // p[i] = -||delta(fr,i)||^2
            // now p[j] has -||\Delta(fr,j)||^2
            // add bias[j] to each p[j] if bias is enabled
            if (myparas.bias_enabled)
                if (myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
                    for (i = 0; i < collection_size; i++)
                        p[i] += bias_terms[collection_idx[i]];
                else
                    add_vec(p, bias_terms, collection_size, 1.0);

            //printf("Trying to avoid underflow\n");
            double max_val = p[0];
            for(i = 0; i < collection_size; i++)
                max_val = p[i] > max_val? p[i] : max_val;
            vec_scalar_sum(p, -max_val, collection_size);
            Veccopy(p, tempk, collection_size);
            exp_on_vec(tempk, collection_size);
            // tempk[i] = exp(-||delta(fr,i)||^2)
            double temp_sum;
            if(myparas.allow_self_transition)
                temp_sum = sum_vec(tempk, collection_size);
            else
            {
                temp_sum = 0.0;
                for(i = 0; i < collection_size; i++)
                    if(!(myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic))
                        temp_sum += (i != fr)? tempk[i] : 0.0;
                    else
                        temp_sum += (collection_idx[i] != fr)? tempk[i] : 0.0;
            }
            // temp_sum = Z(fr)
            vec_scalar_sum(p, -log(temp_sum), collection_size);
            // exp(p[i]) = exp(-||delta(fr,i)||^2) / Z(fr)

            if (myparas.num_points < 3)
                for(j = 0; j < collection_size; j++)
                    for(i = 0; i < d; i++)
                        dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
            else
                for(j = 0; j < collection_size; j++)
                    for(i = 0; i < d; i++) {
                        dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
                        mid_dv[j][i] = -2.0 * exp(p[j]) * mid_delta[j][i];
                    }

            // dv[i] = -2 * exp(-||delta(fr,i)||^2) * delta(fr,i) / Z(fr)

            sum_along_direct(dv, dufr, collection_size, d, 0);
            if (myparas.num_points == 3)
                sum_along_direct(mid_dv, mid_dufr, collection_size, d, 0);

            if (myparas.hessian) {

                // preliminary code for using the diagonal of the Hessian
                // matrix to determine learning rates

                // dxx here is the change in diagonal hessian component from
                // this stoc grad step. Need to multiply these things by entry
                // in transition matrix.
                exp_z_fr = temp_sum;

                double min_val = 0;
                for (i = 0; i < collection_size; i++)
                    for (j = 0; j < d; j++) {
                        d_hess_fr[j] += -2 * dv[i][j] + 2 * tempk[i] * (2 * tempkd[i][j] * tempkd[i][j] + 1);
                    }


                for (j = 0; j < d; j++) {
                    d_hess_fr[j] /= exp_z_fr;
                    d_hess_fr[j] -= collection_size * 2.0;
                }

            }


            double accu_temp_vals = 0.0;



            for(to = 0; to < k; to++)
            {
                if(myparas.allow_self_transition || (!myparas.allow_self_transition && fr != to))
                {
                    if(myparas.use_hash_TTable)
                    {
                        temp_pair.fr = fr;
                        temp_pair.to = to;
                        idx = exist_in_hash(tcount, temp_pair); 
                    }
                    else
                        idx = tcount_full[fr][to] > 0.0 ? 1 : -1;
                    if(idx >= 0)
                    {
                        if(myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
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
                            if(s < 0)
                            {
                                printf("Error: The next song is not in the collection. fr = %d, to = %d\n", fr, to);
                                exit(1);
                            }
                        }
                        else
                            s = to;

                        if(myparas.use_hash_TTable)
                            temp_val = retrieve_value_with_idx(tcount, idx);
                        else
                            temp_val = tcount_full[fr][to];
                        accu_temp_vals += temp_val;
                        if (myparas.bias_enabled) {
                            if (myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
                                for (i = 0; i < collection_size; i++)
                                    bias_grad[collection_idx[i]] += -1.0 * temp_val * exp(p[i]);
                            else
                                for (i = 0; i < k; i++)
                                    bias_grad[i] += -1.0 * temp_val * exp(p[i]);

                            bias_grad[s] += temp_val;
                        }
                        if (myparas.hessian) {
                            double min_val = 0.0;

                            for (j = 0; j < d; j++) {
                                d_hess_to[j] += -2 * dv[s][j] + 2 * tempk[s] * (2 * tempkd[s][j] * tempkd[s][j] + 1);
                            }

                            min_val -= 0.01;
                            for (j = 0; j < d; j++) {
                                d_hess_to[j] /= exp_z_fr;
                            }

                            add_vec(hessian[fr], d_hess_fr, d, temp_val);
                            if (s != fr || myparas.allow_self_transition)
                                for (j = 0; j < d; j++)
                                    hessian[s][j] -= 2.0;
                            if (myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
                                for (i = 0; i < collection_size; i++)
                                    if (collection_idx[i] != fr || myparas.allow_self_transition)
                                        add_vec(hessian[collection_idx[i]], d_hess_to, d, temp_val);
                                    else
                                        for (i = 0; i < k; i++)
                                            if (i != fr || myparas.allow_self_transition)
                                                add_vec(hessian[i], d_hess_to, d, temp_val);
                        }


                        add_vec(cdx[(myparas.num_points - 1) * k + to], delta[s], d, 2.0 * temp_val);
                        if(using_tags)
                            for(i = 0; i < td.num_tags_for_song[to]; i++)
                                add_vec(cdx[myparas.num_points * k + (myparas.num_points - 1) * m + td.tags[to][i]], delta[s], d, 2.0 * temp_val / ((double) td.num_tags_for_song[to]));
                        Veccopy(dufr, tempd, d);
                        add_vec(tempd, delta[s], d, 2.0);
                        scale_vec(tempd, d, -1.0);
                        add_vec(cdx[fr], tempd, d, temp_val);
                        if(using_tags)
                            for(i = 0; i < td.num_tags_for_song[fr]; i++)
                                add_vec(cdx[myparas.num_points * k + td.tags[fr][i]], tempd, d, temp_val / ((double) td.num_tags_for_song[fr]));
                        if (myparas.num_points == 3) {
                            add_vec(cdx[k + to], mid_delta[s], d, 2.0 * temp_val);
                            Veccopy(mid_dufr, mid_tempd, d);
                            add_vec(mid_tempd, mid_delta[s], d, 2.0);
                            scale_vec(mid_tempd, d, -1.0);
                            add_vec(cdx[k + fr], mid_tempd, d, temp_val);
                        }

                        llhood += temp_val * p[s];
                        realn += temp_val;

                        if(write_distr_files)
                        {
                            song_sep_ll[fr] += temp_val * p[s];
                            song_sep_ll[to] += temp_val * p[s];
                            fprintf(trans_distr_file, "%d %f\n", (int)temp_val, temp_val * p[s]);
                        }
                    }
                }
            }

            if(myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
            {
                for(i = 0; i < collection_size; i++)
                    for(j = 0; j < d; j++) {
                        cdx[(myparas.num_points - 1) * k + collection_idx[i]][j] += dv[i][j] * accu_temp_vals;
                        if (collection_idx[i] != fr && myparas.num_points == 3)
                            cdx[k + collection_idx[i]][j] += mid_dv[i][j] * accu_temp_vals;
                    }
                if(using_tags)
                {
                    for(i = 0; i < collection_size; i++)
                    {
                        int current_song = collection_idx[i];
                        for(j = 0; j < td.num_tags_for_song[current_song]; j++)
                        {
                            add_vec(cdx[myparas.num_points * k + (myparas.num_points - 1) * m + td.tags[current_song][j]], dv[i], d, accu_temp_vals / ((double) td.num_tags_for_song[current_song]));
                        }
                    }
                }
            }
            else {
                add_mat(cdx + (myparas.num_points - 1) * k, dv, collection_size, d, accu_temp_vals);
                if(using_tags)
                    for(i = 0; i < collection_size; i++)
                        for(j = 0; j < td.num_tags_for_song[i]; j++)
                            add_vec(cdx[myparas.num_points * k + (myparas.num_points - 1) * m + td.tags[i][j]], dv[i], d, accu_temp_vals / ((double) td.num_tags_for_song[i]));

                if (myparas.num_points == 3)
                    for (i = 0; i < collection_size; i++)
                        for (j = 0; j < d; j++)
                            if (i != fr)
                                cdx[k + i][j] += mid_dv[i][j] * accu_temp_vals;
            }

            if(myparas.fast_collection || myparas.grid_heuristic || myparas.landmark_heuristic)
                free(collection_idx);
            Array2Dfree(delta, collection_size, d);
            Array2Dfree(dv, collection_size, d);
            free(p);
            free(tempk);
            Array2Dfree(tempkd, collection_size, d);
            if (myparas.num_points == 3) {
                free(mid_p);
                Array2Dfree(mid_dv, collection_size, d);
                Array2Dfree(mid_delta, collection_size, d);
                Array2Dfree(mid_tempkd, collection_size, d);
            }

            if(myparas.stoc_grad) {
                if (myparas.regularization_type > 0) {
                    reg_lower[0] = fr;
                    reg_upper[0] = fr; 
                    reg_lower[myparas.num_points - 1] = 0;
                    reg_upper[myparas.num_points - 1] = k;
                    if (myparas.num_points == 3) {
                        reg_lower[1] = 0;
                        reg_upper[1] = k;
                    }
                    compute_regularization_term(myparas, X, reg_x, k, d,
                            reg_lower, reg_upper, &pd,
                            myparas.regularization_type);

                }

                if (myparas.tag_regularizer > 0 && using_tags) {
                    compute_regularization_term(myparas, X + (k * myparas.num_points),
                            reg_x + (k * myparas.num_points), m, d, t_reg_lower,
                            t_reg_upper, &pd, myparas.tag_regularizer);
                }

                if (myparas.regularization_type > 0) {
                    // have to do some odd scaling to make sure scales are
                    // consistent between the from point regularizer addition
                    // and the to point additions (since the to points get the
                    // regularization term added k-1 times as often as the from
                    // points otherwise).

                    if (myparas.num_points > 1) {
                        add_vec(cdx[fr], reg_x[fr], d, 1.0);
                        if (myparas.num_points == 3)
                            scale_vec(reg_x[k + fr], d, k - 1.0);
                        add_mat(cdx + k, reg_x + k, (myparas.num_points - 1) * k,
                                d, 1.0 / (k - 1.0));
                    }
                    else {
                        scale_vec(reg_x[fr], d, k - 1.0);
                        add_mat(cdx, reg_x, k, d, 1.0 / (k - 1.0));
                    }
                }

                if (myparas.tag_regularizer > 0) {
                    add_mat(cdx + k * myparas.num_points,
                            reg_x + k * myparas.num_points, m, d, 1.0 / (k - 1.0));
                }

                if(!using_tags) {
                    if (!myparas.hessian) {
                        add_mat(X, cdx, k * myparas.num_points, d, eta / (double)n);
                    }
                    else if (myparas.hessian == 1) {

                        double divisor = 0.0;
                        for (i = 0; i < k * myparas.num_points; i++)
                            for (j = 0; j < d; j++) {
                                if (fabs(hessian[i][j]) < 0.0001)
                                    divisor = 0.0001;
                                else
                                    divisor = fabs(hessian[i][j]);
                                X[i][j] += (eta / (divisor * (double)n)) * cdx[i][j];
                            }


                    }
                    else if (myparas.hessian == 2) {

                        double hessian_norm = frob_norm(hessian, k * myparas.num_points, d)
                            / (k * myparas.num_points * d);

                        if (hessian_norm < 0.0001)
                            hessian_norm = 1000;
                        else
                            hessian_norm = 1.0 / hessian_norm;

                        add_mat(X, cdx, k * myparas.num_points, d, hessian_norm * eta / (double)n);

                    }
                    if (myparas.bias_enabled)
                        add_vec(bias_terms, bias_grad, k, eta / (double)n);
                }
                else {
                    add_mat(X, cdx, (k + m) * myparas.num_points, d, eta / (double)n);
                    if (myparas.bias_enabled)
                        add_vec(bias_terms, bias_grad, k, eta / (double)n);
                }
            }
        }

        llhood /= realn;

        double reg_objective = 0.0;

        // Regularization

        if (!using_tags) {
            if ((myparas.regularization_type > 0) && !myparas.stoc_grad) {

                reg_objective =
                    compute_regularization_term(myparas, X, reg_x, k, d, reg_lower,
                            reg_upper, &pd, myparas.regularization_type);

                add_mat(cdx, reg_x, k, d, 1.0);

                // average, to scale to llhood
                reg_objective /= realn;
                llhood += reg_objective;

            }
            else if ((myparas.regularization_type > 0) && myparas.stoc_grad) {
                reg_objective =
                    compute_regularization_term(myparas, X, 0, k, d, 0,
                            0, &pd, myparas.regularization_type);
                reg_objective /= realn;
                llhood += reg_objective;
            }
        }
        else {
            if (myparas.regularization_type > 0 && myparas.stoc_grad)
                reg_objective += compute_regularization_term(myparas, X, 0, k, d, 0,
                        0, &pd, myparas.regularization_type) / realn;

            if (myparas.tag_regularizer > 0 && myparas.stoc_grad && using_tags)
                reg_objective += compute_regularization_term(myparas,
                        X + k * myparas.num_points, 0, m, d, 0, 0, &pd,
                        myparas.tag_regularizer) / realn;
        }

        llhood += reg_objective;


        printf("The iteration took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
        printf("The iteration took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
        if(myparas.num_points == 3)
            printf("Norms of landing v, middle point, and exit u: %f\t%f\t%f\n", frob_norm(realX + 2 * k, k, d), frob_norm(realX + k, k, d), frob_norm(realX, k, d));fflush(stdout);
        if(myparas.num_points == 2)
            printf("Norms of landing v and exit u: %f\t%f\n", frob_norm(realX + k, k, d), frob_norm(realX, k, d));fflush(stdout);
        if(myparas.num_points == 1)
            printf("Norms of coordinates: %f\n", frob_norm(realX, k, d));fflush(stdout);
        if (myparas.regularization_type == 0) {
            printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
        }
        else {
            printf("Avg regularized log-likelihood on train: %f\n", llhood);fflush(stdout);
        }

        if (myparas.bias_enabled)
            printf("Norm of bias term vector: %f\n",
                    vec_norm(bias_terms, k));

        if (myparas.hessian == 2) {
            double hessian_norm = frob_norm(hessian, k * myparas.num_points, d)
                / (k * myparas.num_points * d);
            printf("Hessian norm %f\n", 1.0 / hessian_norm);
            printf("Final step size: %f\n", eta / hessian_norm);
        }

        if(myparas.grid_heuristic || myparas.landmark_heuristic)
            printf("%f %f ratio t/w: %f\n", (float)ttable_count / (float)k, (float)wtable_count / (float)k,(float)ttable_count / (float)wtable_count);



        if(t > 0)
        {
            if(llhoodprev > llhood)
            {
                eta /= myparas.beta;
                //eta = 0.0;
                printf("Reducing eta to: %.30f\n", eta);
            }
            else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
            {
                eta *= myparas.alpha;
                printf("Increasing eta to: %.30f\n", eta);
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

        printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));

        if (myparas.bias_enabled)
            printf("magnitude of gradient and bias gradient %f \n", frob_norm(cdx, k * myparas.num_points, d) + vec_norm(bias_grad, k));
        else
            printf("magnitude of gradient %f\n", frob_norm(cdx, k * myparas.num_points, d));


        printf("Avg norms of each point: ");
        for(i = 0; i < myparas.num_points; i++)
        {
            if(!using_tags)
                printf("%f ", avg_norm(X + i * k , k, d));
            else
            {
                printf("%f ", avg_norm(X + i * k , k, d));
                printf("%f ", avg_norm(X + myparas.num_points * k + i * m , m, d));
            }
        }
        putchar('\n');
        // break regardless of least_iter if the range of llhood over the last
        // n iterations is less than epsilon and at least n iterations have
        // been completed

        // windowed convergence criterion and old one-step difference
        // criterion are now mutually exclusive

        if (myparas.num_llhood_track > 0
                && fabs(min_llhood - max_llhood) < myparas.eps 
                && total_llhood < myparas.num_llhood_track) {
            if (optimization_rebooted == 0 && myparas.reboot_enabled) {
                printf("Convergence criterion satisfied once, rebooting from here.\n");
                for (i = 0; i < myparas.num_llhood_track; i++)
                    last_n_llhoods[i] = 1.0;
                eta = sqrt(n);
                myparas.alpha = 1.0;
                myparas.beta = 2.0;
                optimization_rebooted = 1;
            }
            else
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

        if(!using_tags)
            write_embedding_to_file(X, myparas.num_points * k, d, embedding_file, bias_terms);
        else
            write_embedding_to_file(X, myparas.num_points * (k + m), d, embedding_file, bias_terms);
        // need to make sure to write bias terms at some point too
        if(!myparas.stoc_grad)
        {
            if(!using_tags)
                add_mat(X, cdx, k * myparas.num_points, d, eta / (double)n);
            else
                add_mat(X, cdx, (k + m) * myparas.num_points, d, eta / (double)n);

        }


        if(write_distr_files)
        {
            for(i = 0; i < k; i++)
                fprintf(song_distr_file, "%d %f\n", (int)(pd.id_counts[i]), song_sep_ll[i]);
            fclose(song_distr_file);
            fclose(trans_distr_file);
            free(song_sep_ll);
        }

        if((myparas.grid_heuristic || myparas.landmark_heuristic) && t % myparas.regeneration_interval == myparas.regeneration_interval - 1 && t < myparas.landmark_burnin_iter)
        {
            if(myparas.grid_heuristic)
            {
                Free_Grid_Table(&gt);
                printf("Old grid table freed.\n");
            }
            if(myparas.landmark_heuristic)
            {
                Free_LMTable(&lmt);
                printf("Old landmark table freed.\n");
            }
        }
        putchar('\n');
    }

    free(dufr);
    free(tempd);
    if(!using_tags)
        Array2Dfree(cdx, k * myparas.num_points, d);
    else
        Array2Dfree(cdx, (k + m) * myparas.num_points, d);
    if (myparas.num_points == 3) {
        free(mid_dufr);
        free(mid_tempd);
    }
    if (myparas.regularization_type > 0)
        Array2Dfree(reg_x, k * myparas.num_points, d);
    if(myparas.use_hash_TTable)
        free_hash(tcount);
    else
        Array2Dfree(tcount_full, k, k);
    if(myparas.fast_collection)
    {
        free_transition_table(&ttable);
        free_transition_table(&BFStable);
        printf("Transition table freed.\n");
    }

    if(myparas.grid_heuristic)
    {
        free_transition_table(&ttable);
        Free_Grid_Table(&gt);
        printf("Grid table freed.\n");
    }
    if(myparas.landmark_heuristic)
    {
        free_transition_table(&ttable);
        Free_LMTable(&lmt);
        printf("Landmark table freed.\n");
    }
    if(using_tags)
        free_tag_data(td);
    if(using_tags)
        Array2Dfree(realX, k * myparas.num_points, d);
    if (myparas.hessian) {
        free(d_hess_fr);
        free(d_hess_to);
        Array2Dfree(hessian, k * myparas.num_points, d);
    }
    return X;
}

double compute_regularization_term(PARAS myparas, double** X, double** dl_x, int num_samples, int dim, int* low_indices, int* upper_indices, PDATA* pd, int reg_type) {

    // set dl_x = 0 to compute objective, otherwise objective will be left as
    // 0.0.
    // Must be provided with regularization type as input argument since the
    // function is called independently for tag regularizer and song
    // regularizer, and is agnostic about which thing it is regularizing.

    // computes the contribution to the update of u, v, and x locations based
    // on the specified mode of regularization. dl_du, dl_dv, and (if midpoint
    // method is used) dl_dx should all be allocated but do not necessarily
    // have to be equal to zero matrices. Each of those should be a matrix
    // with num_samples rows and dim columns. myparas.lambda is used to weight
    // the contribution, and the weighted objective function contribution is
    // the return value.
    //
    // Currently the following regularization options are implemented:
    //
    // 0) No regularization
    // 1) Squared L2 norm regularization of u, v, and (if midpoint method is
    //    used) x.
    // 2) Squared L2 norm of the difference between u and v for each song.
    // 3) Combination of the terms used in methods 1 and 2. Coefficient for method 2
    //    gets multiplied by nu_multiplier.
    // 4) Wild regularization term that tries to minimize the diameter of any
    //    set of embeddings of songs appearing in the same playlist.
    //
    // To use this function, just give the current embedding in the matrix X
    // and add the resulting dl_du, dl_dv, and dl_dx to the gradient of the
    // objective function.

    int i, j, k;
    double reg_objective = 0.0;
    int last = myparas.num_points - 1;

    if (dl_x != 0)
        for (i = 0; i < num_samples * myparas.num_points; i++)
            memset(dl_x[i], 0, dim * sizeof(double));

    if (reg_type == 1 || reg_type == 3) {

        // L2 regularization

        // for positive coordinate x, take a step in direction -2 * x. For
        // negative coordinate x, take a step in direction 2 * x.

        // u, v coordinates

        if (dl_x != 0) {
            // if function was not called for reg_objective calculation
            for (i = 0; i < myparas.num_points; i++) {
                for (j = low_indices[i]; j < upper_indices[i]; j++) {
                    for (k = 0; k < dim; k++) {
                        dl_x[i * num_samples + j][k] +=
                            -2.0 * myparas.lambda * X[i * num_samples + j][k];
                    }
                }
            }
        }
        else
            reg_objective += myparas.lambda * frob_norm(X, num_samples, dim);

    }

    if (reg_type == 2 || reg_type == 3) {

        // Constrain each song's u and v to be near each other.

        float difference = 0.0;
        double lambda = 0.0;
        if (reg_type == 3)
            lambda = myparas.nu_multiplier * myparas.lambda;
        else
            lambda = myparas.lambda;

        if (dl_x != 0) { // calculating gradient, not objective
            for (i = 0; i < num_samples; i++) {
                if ( ((low_indices[last] <= i) && (i < upper_indices[last]))
                        || ((low_indices[0] <= i) && (i < upper_indices[0])) ) {
                    for (j = 0; j < dim; j++) {
                        difference =
                            -1.0 * (X[i][j] - X[last * num_samples + i][j]);

                        if ((low_indices[0] <= i) && (i < upper_indices[0]))
                            dl_x[i][j] += 2.0 * lambda * difference;

                        if ((low_indices[last] <= i)
                                && (i < upper_indices[last]))
                            dl_x[last * num_samples + i][j] +=
                                -2.0 * lambda * difference;
                    }
                }
            }
        }
        else { // compute objective only
            for (i = 0; i < num_samples; i++) {
                for (j = 0; j < dim; j++) {
                    difference =
                        -1.0 * (X[i][j] - X[last * num_samples + i][j]);
                    reg_objective += lambda * pow(difference, 2.0);
                }
            }
        }
    }


    if (myparas.regularization_type == 4) {

        // Try to force playlists to be embedded in a compact ball
        // Do this with a regularization term for the maximum of a metric over
        // all playlists and all pairs within that playlist. The metric can
        // be:
        // d(x,y) = ||u_x - u_y||^2 + ||v_x - v_y||^2
        //        + ||u_x - v_y||^2 + ||v_x - u_y||^2

        double max_dist = -1.0;
        double pair_dist = 0.0;
        int x, y;
        int max_i = -1, max_s1 = -1, max_s2 = -1;

        for (i = 0; i < pd->num_playlists; i++) {
            for (x = 0; x < pd->playlists_length[i]; x++) {
                for (y = x + 1; y < pd->playlists_length[i]; y++) {
                    pair_dist = R4_distance_metric(myparas, X,
                            pd->playlists[i][x], pd->playlists[i][y], num_samples);
                    if (pair_dist > max_dist) {
                        max_dist = pair_dist;
                        max_i = i;
                        max_s1 = pd->playlists[i][x];
                        max_s2 = pd->playlists[i][y];
                    }
                }
            }
        }

        reg_objective = myparas.lambda * max_dist;

        // now we have the pair of songs that maximize the distance metric.

        // first clear dl_x

        for (i = 0; i < myparas.num_points * num_samples; i++) {
            for (j = 0; j < dim; j++) {
                dl_x[i][j] = 0.0;
            }
        }

        int v0 = last * num_samples;

        // -dl/du_s1_i = -2(u_s1_i - u_s2_i) - 2(u_s1_i - v_s2_i)
        if ((low_indices[0] <= max_s1) && (max_s1 < upper_indices[0]))
            for (j = 0; j < dim; j++)
                dl_x[max_s1][j] +=
                    myparas.lambda *
                    (-2.0 * (X[max_s1][j] - X[max_s2][j])
                     - 2.0 * (X[max_s1][j] - X[v0 + max_s2][j]));

        // -dl/du_s2_i = 2(u_s1_i - u_s2_i) + 2(v_s1_i - u_s2_i)
        if ((low_indices[0] <= max_s2) && (max_s2 < upper_indices[0]))
            for (j = 0; j < dim; j++)
                dl_x[max_s2][j] +=
                    myparas.lambda *
                    (2.0 * (X[max_s1][j] - X[max_s2][j])
                     + 2.0 * (X[v0 + max_s1][j] - X[max_s2][j]));

        // -dl/dv_s1_i = -2(v_s1_i - v_s2_i) - 2(v_s1_i - u_s2_i)
        if ((low_indices[last] <= max_s1) && (max_s1 < upper_indices[last]))
            for (j = 0; j < dim; j++)
                dl_x[v0 + max_s1][j] +=
                    myparas.lambda *
                    (-2.0 * (X[v0 + max_s1][j] - X[v0 + max_s2][j])
                     - 2.0 * (X[v0 + max_s1][j] - X[max_s2][j]));

        // -dl/dv_s2_i = 2(v_s1_i - v_s2_i) + 2(u_s1_i - v_s2_i)
        if ((low_indices[last] <= max_s2) && (max_s2 < upper_indices[last]))
            for (j = 0; j < dim; j++)
                dl_x[v0 + max_s2][j] +=
                    myparas.lambda *
                    (2.0 * (X[v0 + max_s1][j] - X[v0 + max_s2][j])
                     + 2.0 * (X[max_s1][j] - X[v0 + max_s2][j]));


    }
    else if (myparas.regularization_type == 5) {

        // for each playlist, calculate the centroid of the u's and v's in
        // that playlist, then we want to minimize distance to that centroid,
        // so we should calculate subgradient for each point summing over
        // all the playlists in which it appears

        // accumulator for centroid calculation
        double* centroid = (double*) calloc(dim, sizeof(double));
        // counters for multiple of u, v to add to dl_x
        //        double* u_counters = (double*) calloc(num_samples, sizeof(double));
        //        double* v_counters = (double*) calloc(num_samples, sizeof(double));

        double dxm_temp = 0.0;

        // first will need to clear the dl_x vector

        for (i = 0; i < myparas.num_points * num_samples; i++)
            for (j = 0; j < dim; j++)
                dl_x[i][j] = 0.0;

        // for each playlist P

        for (i = 0; i < pd->num_playlists; i++) {

            // reset centroid
            for (j = 0; j < dim; j++)
                centroid[j] = 0.0;

            // for each song s in playlist
            for (j = 0; j < pd->playlists_length[i]; j++) {

                // add to centroid a multiple of 1 / 2|P| of u_s and v_s
                // actually adding just multiple of 1 / |P| since the 2
                // cancels later from the square in the regularizer
                add_vec(centroid, X[pd->playlists[i][j]], dim,
                        1.0 / pd->playlists_length[i]);
                add_vec(centroid, X[last + pd->playlists[i][j]], dim,
                        1.0 / pd->playlists_length[i]);

            }

            // now centroid for this playlist is calculated.
            // for each u in the playlist, subtract centroid
            // from dl_u and add 1/|P| * u to dl_u, do same for v.

            for (j = 0; j < pd->playlists_length[i]; j++) {

                // only do it if this song should have its gradient calculated
                if ((low_indices[0] <= pd->playlists[i][j]) &&
                        (pd->playlists[i][j] < upper_indices[0])) {

                    // need to add term to the regularizer magnitude as well
                    // magnitude of difference between u and centroid, plus
                    // same for v and centroid

                    // adds the difference vector u - m to dl_x, adding up its
                    // norm for the reg_objective in the process

                    for (k = 0; k < dim; k++) {
                        dxm_temp = X[pd->playlists[i][j]][k] - centroid[k];
                        dl_x[pd->playlists[i][j]][k] += myparas.lambda *
                            dxm_temp / pd->playlists_length[i];
                        reg_objective += myparas.lambda * dxm_temp * dxm_temp;
                    }

                    //                    add_vec(dl_x[pd.playlists[i][j]], dxm_temp, dim,
                    //                        1.0 / pd.playlists_length[i]);

                    // no need to add this entire vector every time.
                    // Let's change it to a double-valued counter per song and
                    // add 1 / |P| to it every time the song appears, then
                    // add that multiple of the song after the iteration
                    // through the playlists finishes.

                    //                    add_vec(dl_x[playlists[i][j]], X[playlists[i][j]],
                    //                        dim, 1.0 / pd.playlists_length[i]);

                    //                    u_counters[pd.playlists[i][j]] +=
                    //                        1.0 / pd.playlists_length[i];

                }

                if ((low_indices[last] <= pd->playlists[i][j]) &&
                        (pd->playlists[i][j] < upper_indices[last])) {

                    for (k = 0; k < dim; k++) {
                        dxm_temp =
                            X[last * num_samples + pd->playlists[i][j]][k]
                            - centroid[k];
                        dl_x[last * num_samples + pd->playlists[i][j]][k]
                            += myparas.lambda *
                            dxm_temp / pd->playlists_length[i];
                        reg_objective += myparas.lambda * dxm_temp * dxm_temp;
                    }

                    //                    add_vec(dl_x[last + pd.playlists[i][j]], centroid, dim,
                    //                        -1.0);
                    //                    add_vec(dl_x[last + playlists[i][j]],
                    //                        X[last + playlists[i][j]], dim,
                    //                        1.0 / pd.playlists_length[i]);
                    //                    v_counters[pd.playlists[i][j]] +=
                    //                        1.0 / pd.playlists_length[i];

                }

                // compensate for any terms not added to the reg_objective due
                // to not being in the desired indices range

                for (k = 0; k < low_indices[0]; k++)
                    reg_objective += myparas.lambda * pow(X[pd->playlists[i][j]][k]
                            - centroid[k], 2);
                for (k = upper_indices[0]; k < num_samples; k++)
                    reg_objective += myparas.lambda * pow(X[pd->playlists[i][j]][k]
                            - centroid[k], 2);
                for (k = 0; k < low_indices[last]; k++)
                    reg_objective += myparas.lambda * pow(X[last * num_samples
                            + pd->playlists[i][j]][k] - centroid[k], 2);
                for (k = upper_indices[last]; k < num_samples; k++)
                    reg_objective += myparas.lambda * pow(X[last * num_samples
                            + pd->playlists[i][j]][k] - centroid[k], 2);

            }
        }

        // here add multiple of u, v, to corresponding dl_x vector

        //        for (i = 0; i < num_samples; i++) {
        //            if (u_counters[i] != 0.0)
        //                add_vec(dl_x[i], X[i], dim, u_counters[i]);
        //            if (v_counters[i] != 0.0)
        //                add_vec(dl_x[last + i], X[last + i], dim, v_counters[i]);
        //        }

        // free centroid, u_counters, v_counters

        free(centroid);
        //        free(u_counters);
        //        free(v_counters);

    }

    return -1.0 * reg_objective;

}

double R4_distance_metric(PARAS myparas, double** X, int x_ind, int y_ind, int k) {

    // First idea is:
    // d(x,y) = ||u_x - u_y||^2 + ||v_x - v_y||^2
    //        + ||u_x - v_y||^2 + ||v_x - u_y||^2

    double result = 0.0;
    int i, j;
    int last = myparas.num_points - 1;

    for (i = 0; i < myparas.d; i++) {
        // uu term
        result += pow(X[x_ind][i] - X[y_ind][i], 2);
        // vv
        result += pow(X[last * k + x_ind][i] - X[last * k + y_ind][i], 2);
        // uv
        result += pow(X[x_ind][i] - X[last * k + y_ind][i], 2);
        // vu
        result += pow(X[last * k + x_ind][i] - X[y_ind][i], 2);
    }

    return result;

}

double** logistic_embed_triple_dependency(PDATA pd, PARAS myparas, char* embeddingfile, int* l)
{
    double** X;
    int n; //Number of training transitions
    int i;
    int j;
    int s;
    int t;
    int prev;
    int fr;
    int to;
    int d = myparas.d;
    int k = pd.num_songs;
    double z;
    double eta;
    int idx;
    if (myparas.ita > 0)
        eta = myparas.ita;
    else
        eta = k;
    n = 0;
    for(i = 0; i < pd.num_playlists; i ++)
        if(pd.playlists_length[i] > 0)
            n += pd.playlists_length[i] - 1;
    printf("Altogether %d transitions.\n", n);fflush(stdout);


    *l = k;
    X = randarray(k, d, 1.0);

    double llhood = 0.0;
    double llhoodprev;
    double realn;

    time_t start_time;
    clock_t start_clock, temp_clock;
    ////////////////////////////////////////////////
    //Experimental code with long-range dependency
    //Don't turn on -T flag if you don't want these
    //features
    //

    THASH* triple_count;
    TTRIPLE temp_triple;
    THELEM triple_elem;


    if(myparas.num_points != 1)
    {
        printf("Currently only single-point model is supported with triple dependency.\n");
        exit(1);
    }
    triple_count = t_create_empty_hash(2 * n);
    printf("Triple matrix created.\n");
    for(i = 0; i < pd.num_playlists; i++)
    {
        if(pd.playlists_length[i] > 2)
        {
            for(j = 0; j < pd.playlists_length[i] - 2; j++)
            {
                temp_triple.prev = pd.playlists[i][j];
                temp_triple.fr = pd.playlists[i][j + 1];
                temp_triple.to = pd.playlists[i][j + 2];

                //printf("(%d, %d, %d)\n", temp_triple.prev, temp_triple.fr, temp_triple.to);

                if(temp_triple.prev >= 0 && temp_triple.fr >= 0 && temp_triple.to >= 0)
                {
                    //printf("ok till here\n");fflush(stdout);
                    idx = t_exist_in_hash(triple_count, temp_triple);
                    if(idx < 0)
                    {
                        triple_elem.key = temp_triple;
                        triple_elem.val = 1.0;
                        t_add_entry(triple_count, triple_elem);
                    }
                    else
                        t_update_with(triple_count, idx, 1.0);
                }
            }
        }
    }
    printf("Triple matrix initialized.\n");
    //Examing some statistics of the triple matrix
    /*
       double max_val = -1.0;
       for(i = 0; i < triple_count -> length; i++)
       {
       triple_elem = *((triple_count -> p) + i);
       if(!t_is_null_entry(triple_elem.key))
       max_val = triple_elem.val > max_val? triple_elem.val : max_val;
    //printf("%f\n", triple_elem.val);
    }
    printf("%f\n", max_val);
    */
    /////////////////////////////////////////////////

    double** cdx = zerosarray(k, d);
    double** delta = zerosarray(k, d);
    double* delta_vec = (double*)calloc(d, sizeof(double));
    double** dv = zerosarray(k, d);
    double* p = (double*)calloc(k, sizeof(double));
    double* tempk = (double*)calloc(k, sizeof(double));
    double** tempkd = zerosarray(k, d);

    for(t = 0; t < 100000; t++)
    {
        start_time = time(NULL);
        start_clock = clock();
        printf("Iteration %d\n", t + 1);fflush(stdout);
        llhoodprev = llhood;
        llhood = 0.0;
        realn = 0.0;

        for(s = 0; s < triple_count -> length; s++)
        {
            triple_elem = *((triple_count -> p) + s);
            if(!t_is_null_entry(triple_elem.key))
            {
                for(i = 0; i < k; i++)
                    memset(cdx[i], 0, d * sizeof(double));
                prev = triple_elem.key.prev;
                fr = triple_elem.key.fr;
                to = triple_elem.key.to;
                double temp_val = triple_elem.val;

                for(i = 0; i < d; i++)
                    delta_vec[i] = X[fr][i] - X[prev][i] ;
                for(j = 0; j < k; j++)
                    for(i = 0; i < d; i++)
                        delta[j][i] = X[fr][i] - X[j][i];

                mat_mult(delta, delta, tempkd, k, d);
                //scale_mat(tempkd, k, d, -1.0);
                sum_along_direct(tempkd, p, k, d, 1);
                scale_vec(p, k, -1.0);

                //for(i = 0; i < k; i++)
                //p[i] -= myparas.angle_lambda * innerprod(delta[i], delta_vec, d);

                norm_wo_underflow(p, k, tempk);

                /*
                   double temp_sum = 0.0;
                   for(i = 0; i < k; i++)
                   temp_sum += exp(p[i]);
                   printf("temp_sum is %f\n", temp_sum);
                   */
                llhood += temp_val * p[to];
                realn += temp_val;
            }
        }
        llhood /= realn;
        printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
    }
    Array2Dfree(cdx, k , d);
    Array2Dfree(delta, k , d);
    Array2Dfree(dv, k , d);
    free(p);
    free(tempk);
    free(delta_vec);
    Array2Dfree(tempkd, k, d);

}

double** logistic_embed_sp_mt(PDATA pd, PARAS myparas, char* embeddingfile, int* l)
{

    printf("Multi-threaded version of the algorithm, with number of threads %d.\n", myparas.num_threads);
    double** X;
    int n; //Number of training transitions
    int i;
    int j;
    int s;
    int t;
    int prev;
    int fr;
    int to;
    int d = myparas.d;
    int k = pd.num_songs;
    double z;
    double eta;
    double etan;
    int idx;
    if (myparas.ita > 0)
        eta = myparas.ita;
    else
        eta = k;
    n = 0;
    for(i = 0; i < pd.num_playlists; i ++)
        if(pd.playlists_length[i] > 0)
            n += pd.playlists_length[i] - 1;
    printf("Altogether %d transitions.\n", n);fflush(stdout);


    *l = k;
    X = randarray(k, d, 1.0);

    double* bias_terms = 0;
    double* bias_grad = 0;
    if(myparas.bias_enabled)
    {
        bias_terms = calloc(k, sizeof(double));
        bias_grad = calloc(k, sizeof(double));
    }

    double llhood = 0.0;
    double llhoodprev;
    double realn;

    double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
    int llhood_track_index = 0;

    for (i = 0; i < myparas.num_llhood_track; i++)
        last_n_llhoods[i] = 1.0;

    time_t start_time;
    clock_t start_clock, temp_clock;


    double temp_val;

    double accu_temp_vals = 0.0;

    PHASH* tcount;

    HELEM temp_elem;
    TPAIR temp_pair;

    tcount = create_empty_hash(2 * myparas.transition_range  * n, k);

    printf("Transition matrix created.\n");
    for(i = 0; i < pd.num_playlists; i ++)
    {
        if(pd.playlists_length[i] > 1)
        {
            for(j = 0; j < pd.playlists_length[i] - 1; j++)
            {
                for(t = 1; t <= myparas.transition_range; t++)
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
    int temp_int = 0;
    for(i = 0; i < tcount -> length; i++)
    {
        if(!is_null_entry((tcount -> p)[i].key))
            temp_int++;
    } 
    printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(k * k)));
    printf("Transition matrix initialized.\n");

    HELEM* p_valid_entry;



    ///////////////////////////////////
    //Threads

    pthread_t thread_array[myparas.num_threads];
    pthread_attr_t attr;
    int rc;
    void* status; 
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);



    //Multi-thread varialbes

    pthread_rwlock_t lock;
    pthread_rwlock_init(&lock, NULL);

    FSTP thread_para_array[myparas.num_threads];
    int block_interval = (int)((double) k / (double) myparas.num_threads);
    int next_starting_idx = 0;
    for(i = 0; i < myparas.num_threads; i++)
    {
        init_FSTP(thread_para_array + i, i, k, d, X, bias_terms, tcount, (i == 0)?  0 : next_starting_idx, (i == myparas.num_threads - 1)?  k - next_starting_idx : block_interval, &lock, &etan, &llhood, &realn);

        //Assign blocks to different threads
        //thread_para_array[i].block_starting_idx = (i == 0)?  0 : next_starting_idx;
        //thread_para_array[i].block_length = (i == myparas.num_threads - 1)?  k - next_starting_idx : block_interval;
        next_starting_idx = next_starting_idx + thread_para_array[i].block_length;
        printf("Thread %d handles %d rows\n", i, thread_para_array[i].block_length);
    }






    //

    time_t start_time_total = time(NULL);
    for(t = 0; t < 100000; t++)
    {
        start_time = time(NULL);
        start_clock = clock();
        printf("Iteration %d\n", t + 1);fflush(stdout);
        llhoodprev = llhood;
        llhood = 0.0;
        realn = 0.0;
        etan = eta / (double)n;

        for(i = 0; i < myparas.num_threads; i++)
        {
            rc = pthread_create(&thread_array[i], &attr, parallel_on_from_songs, (void*)(thread_para_array + i));
            if(rc)
            {
                printf("Error in creating thread. Return code: %d\n", rc);
                exit(1);
            }
        }

        for(i = 0; i < myparas.num_threads; i++)
        {
            rc = pthread_join(thread_array[i], &status);
            if(rc)
            {
                printf("Error in joining thread. Return code: %d\n", rc);
                exit(1);
            }
        }

        llhood /= realn;


        printf("The iteration took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
        printf("The iteration took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
        printf("Norms of coordinates: %f\n", frob_norm(X, k, d));fflush(stdout);
        printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
        if (myparas.bias_enabled)
            printf("Norm of bias term vector: %f\n", vec_norm(bias_terms, k));
        if(t > 0)
        {
            if(llhoodprev > llhood)
            {
                eta /= myparas.beta;
                //eta = 0.0;
                printf("Reducing eta to: %f\n", eta);
            }
            else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
            {
                eta *= myparas.alpha;
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
        printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
        //printf("magnitude of gradient %f\n", frob_norm(cdx, k, d));

        write_embedding_to_file(X, k, d, embeddingfile, bias_terms);

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

	printf("Finished. Total time spent is %d seconds\n", (int)(time(NULL) - start_time_total));fflush(stdout);

    pthread_attr_destroy(&attr);
    pthread_rwlock_destroy(&lock);
    free(last_n_llhoods);
    free_hash(tcount);
    if(myparas.bias_enabled)
        free(bias_terms);
    for(i = 0; i < myparas.num_threads; i++)
        free_FSTP(thread_para_array + i);
    return X;
}

//double** logistic_embed_sp_mt(PDATA pd, PARAS myparas, char* embeddingfile, int* l)
//{
//
//    printf("Multi-threaded version of the algorithm, with number of threads %d.\n", myparas.num_threads);
//    double** X;
//    int n; //Number of training transitions
//    int i;
//    int j;
//    int s;
//    int t;
//    int prev;
//    int fr;
//    int to;
//    int d = myparas.d;
//    int k = pd.num_songs;
//    double z;
//    double eta;
//    double etan;
//    int idx;
//    if (myparas.ita > 0)
//        eta = myparas.ita;
//    else
//        eta = k;
//    n = 0;
//    for(i = 0; i < pd.num_playlists; i ++)
//        if(pd.playlists_length[i] > 0)
//            n += pd.playlists_length[i] - 1;
//    printf("Altogether %d transitions.\n", n);fflush(stdout);
//
//
//    *l = k;
//    X = randarray(k, d, 1.0);
//    
//    double llhood = 0.0;
//    double llhoodprev;
//    double realn;
//
//    double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
//    int llhood_track_index = 0;
//
//    for (i = 0; i < myparas.num_llhood_track; i++)
//	last_n_llhoods[i] = 1.0;
//
//    time_t start_time;
//    clock_t start_clock, temp_clock;
//
//    double** cdx = zerosarray(k, d);
//    double** delta = zerosarray(k, d);
//    double** dv = zerosarray(k, d);
//    double* p = (double*)calloc(k, sizeof(double));
//    double* tempk = (double*)calloc(k, sizeof(double));
//    double* tempd = (double*)calloc(d, sizeof(double));
//    double** tempkd = zerosarray(k, d);
//    double* dufr = (double*)calloc(d, sizeof(double));
//
//    double temp_val;
//
//    double accu_temp_vals = 0.0;
//
//    PHASH* tcount;
//
//    HELEM temp_elem;
//    TPAIR temp_pair;
//
//    tcount = create_empty_hash(2 * myparas.transition_range  * n, k);
//
//    printf("Transition matrix created.\n");
//    for(i = 0; i < pd.num_playlists; i ++)
//    {
//	if(pd.playlists_length[i] > 1)
//	{
//	    for(j = 0; j < pd.playlists_length[i] - 1; j++)
//	    {
//		for(t = 1; t <= myparas.transition_range; t++)
//		{
//		    if(j + t < pd.playlists_length[i])
//		    {
//			temp_pair.fr = pd.playlists[i][j];
//			temp_pair.to = pd.playlists[i][j + t];
//
//			if(temp_pair.fr >= 0 && temp_pair.to >= 0)
//			{
//			    idx = exist_in_hash(tcount, temp_pair);
//			    if(idx < 0)
//			    {
//				temp_elem.key = temp_pair;
//				temp_elem.val = 1.0 / (double) t;
//				add_entry(tcount, temp_elem);
//			    }
//			    else
//				update_with(tcount, idx, 1.0 / (double) t);
//			}
//		    }
//		}
//	    }
//	}
//    }
//
//    build_same_song_index(tcount);
//
//    //Sanity check
//    int temp_int = 0;
//    for(i = 0; i < tcount -> length; i++)
//    {
//	if(!is_null_entry((tcount -> p)[i].key))
//	    temp_int++;
//    } 
//    printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(k * k)));
//    printf("Transition matrix initialized.\n");
//
//    HELEM* p_valid_entry;
//
//
//
//    ///////////////////////////////////
//    //Threads
//
//    pthread_t thread_array[myparas.num_threads];
//    pthread_attr_t attr;
//    int rc;
//    void* status; 
//    pthread_attr_init(&attr);
//    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
//    int routine = 0;
//
//
//
//    //Multi-thread varialbes
//    THREADSTRUCT thread_para_array[myparas.num_threads];
//    int block_interval = (int)((double) k / (double) myparas.num_threads);
//    int next_starting_idx = 0;
//    for(i = 0; i < myparas.num_threads; i++)
//    {
//	thread_para_array[i].thread_no = i;
//
//	//Execute the first routine first
//	thread_para_array[i].proutine = &routine;
//
//	//Assign blocks to different threads
//	thread_para_array[i].block_starting_idx = (i == 0)?  0 : next_starting_idx;
//	thread_para_array[i].block_length = (i == myparas.num_threads - 1)?  k - next_starting_idx : block_interval;
//	next_starting_idx = next_starting_idx + thread_para_array[i].block_length;
//	printf("Thread %d handles %d rows\n", i, thread_para_array[i].block_length);
//	
//	//Initialize
//	thread_para_array[i].k = k;
//	thread_para_array[i].d = d;
//	thread_para_array[i].delta = delta;
//	thread_para_array[i].tempkd = tempkd;
//	thread_para_array[i].p = p;
//	thread_para_array[i].dv = dv;
//	thread_para_array[i].dufr = dufr;
//	thread_para_array[i].cdx = cdx;
//	thread_para_array[i].X = X;
//	thread_para_array[i].petan = &etan;
//	thread_para_array[i].paccu_temp_vals = &accu_temp_vals;
//
//
//	//routine, eta, and accu_temp_vals need to be change on the fly
//    }
//    
//
//
//
//
//
//    //
//
//    for(t = 0; t < 100000; t++)
//    {
//	start_time = time(NULL);
//	start_clock = clock();
//	printf("Iteration %d\n", t + 1);fflush(stdout);
//	llhoodprev = llhood;
//	llhood = 0.0;
//	realn = 0.0;
//	for(fr = 0; fr < k; fr++)
//	{
//	    if(tcount -> p_first_tran[fr] == NULL)
//	    {
//		//printf("Skipped one from song\n");
//		continue;
//	    }
//	    for(i = 0; i < k; i++)
//		memset(cdx[i], 0, d * sizeof(double));
//
//            for(j = 0; j < k; j++)
//                for(i = 0; i < d; i++)
//		    delta[j][i] = X[fr][i] - X[j][i];
//
//
//	    /*
//            mat_mult(delta, delta, tempkd, k, d);
//            scale_mat(tempkd, k, d, -1.0);
//            sum_along_direct(tempkd, p, k, d, 1);
//	    */
//	    //Phase 1
//	    routine = 0;
//	    for(i = 0; i < myparas.num_threads; i++)
//	    {
//		rc = pthread_create(&thread_array[i], &attr, run_on_block, (void*)(thread_para_array + i));
//		if(rc)
//		{
//		    printf("Error in creating thread. Return code: %d\n", rc);
//		    exit(1);
//		}
//	    }
//
//	    for(i = 0; i < myparas.num_threads; i++)
//	    {
//		rc = pthread_join(thread_array[i], &status);
//		if(rc)
//		{
//		    printf("Error in joining thread. Return code: %d\n", rc);
//		    exit(1);
//		}
//	    }
//
//	    norm_wo_underflow(p, k, tempk);
//	    for(j = 0; j < k; j++)
//		for(i = 0; i < d; i++)
//		    dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
//            sum_along_direct(dv, dufr, k, d, 0);
//
//            accu_temp_vals = 0.0;
//
//	    p_valid_entry = tcount -> p_first_tran[fr];
//	    while(p_valid_entry != NULL)
//            {
//		to = (p_valid_entry -> key.to);
//		temp_val = p_valid_entry -> val;
//
//		accu_temp_vals += temp_val;
//
//
//		add_vec(cdx[to], delta[to], d, 2.0 * temp_val);
//		Veccopy(dufr, tempd, d);
//		add_vec(tempd, delta[to], d, 2.0);
//		scale_vec(tempd, d, -1.0);
//		add_vec(cdx[fr], tempd, d, temp_val);
//		llhood += temp_val * p[to];
//		realn += temp_val;
//		p_valid_entry = (p_valid_entry -> pnext);
//	    }
//
//	    /*
//	    add_mat(cdx, dv, k, d, accu_temp_vals);
//	    add_mat(X, cdx, k, d, eta / (double)n);
//	    */
//	    //Phase 2
//	    routine = 1;
//	    etan = eta / (double)n;
//	    for(i = 0; i < myparas.num_threads; i++)
//	    {
//		rc = pthread_create(&thread_array[i], &attr, run_on_block, (void*)(thread_para_array + i));
//		if(rc)
//		{
//		    printf("Error in creating thread. Return code: %d\n", rc);
//		    exit(1);
//		}
//	    }
//
//	    for(i = 0; i < myparas.num_threads; i++)
//	    {
//		rc = pthread_join(thread_array[i], &status);
//		if(rc)
//		{
//		    printf("Error in joining thread. Return code: %d\n", rc);
//		    exit(1);
//		}
//	    }
//	}
//
//        llhood /= realn;
//
//
//        printf("The iteration took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
//        printf("The iteration took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
//	printf("Norms of coordinates: %f\n", frob_norm(X, k, d));fflush(stdout);
//	printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
//        if(t > 0)
//        {
//            if(llhoodprev > llhood)
//            {
//                eta /= myparas.beta;
//		//eta = 0.0;
//                printf("Reducing eta to: %f\n", eta);
//            }
//            else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
//            {
//                eta *= myparas.alpha;
//                printf("Increasing eta to: %f\n", eta);
//            }
//        }
//        last_n_llhoods[llhood_track_index] = llhood;
//        double min_llhood = 1.0, max_llhood = -1000.0, total_llhood = 0.0;
//        for (i = 0; i < myparas.num_llhood_track; i++) {
//            total_llhood += last_n_llhoods[i];
//            if (last_n_llhoods[i] < min_llhood)
//                min_llhood = last_n_llhoods[i];
//            if (last_n_llhoods[i] > max_llhood)
//                max_llhood = last_n_llhoods[i];
//        }
//        printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
//	printf("magnitude of gradient %f\n", frob_norm(cdx, k, d));
//
//	write_embedding_to_file(X, k, d, embeddingfile, 0);
//
//        if (myparas.num_llhood_track > 0
//                && fabs(min_llhood - max_llhood) < myparas.eps 
//                && total_llhood < myparas.num_llhood_track) {
//                break;
//        }
//        else if (myparas.num_llhood_track <= 0
//            && llhood >= llhoodprev
//            && llhood - llhoodprev < myparas.eps 
//            && t > myparas.least_iter)
//            break;
//
//
//        // update llhood tracker index
//
//        llhood_track_index++;
//        llhood_track_index %= myparas.num_llhood_track;
//    }
//
//    //pthread_attr_destroy(&attr);
//    Array2Dfree(cdx, k , d);
//    Array2Dfree(delta, k , d);
//    Array2Dfree(dv, k , d);
//    free(p);
//    free(tempk);
//    Array2Dfree(tempkd, k, d);
//    free(last_n_llhoods);
//    free(dufr);
//    free(tempd);
//    free_hash(tcount);
//    return X;
//}

PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile)
{
    //Default options
    PARAS myparas;
    myparas.do_normalization = 0;
    myparas.method = 2;
    myparas.d = 2;
    myparas.ita = 300;
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
    myparas.bias_enabled = 0;
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

    if((i + 3) < argc)
    {
        strcpy(songdistrfile, argv[i + 2]);
        strcpy(transdistrfile, argv[i + 3]);
        myparas.output_distr = 1;
    }

    return myparas;
}


//A pure single point model without many options for further modification
/*
   double** logistic_embed_sp_mt(PDATA pd, PARAS myparas, char* embeddingfile, int* l)
   {

   printf("Multi-threaded version of the algorithm, with number of threads %d.\n", myparas.num_threads);
   double** X;
   int n; //Number of training transitions
   int i;
   int j;
   int s;
   int t;
   int prev;
   int fr;
   int to;
   int d = myparas.d;
   int k = pd.num_songs;
   double z;
   double eta;
   int idx;
   if (myparas.ita > 0)
   eta = myparas.ita;
   else
   eta = k;
   n = 0;
   for(i = 0; i < pd.num_playlists; i ++)
   if(pd.playlists_length[i] > 0)
   n += pd.playlists_length[i] - 1;
   printf("Altogether %d transitions.\n", n);fflush(stdout);


 *l = k;
 X = randarray(k, d, 1.0);

 double llhood = 0.0;
 double llhoodprev;
 double realn;

 double* last_n_llhoods = calloc(myparas.num_llhood_track, sizeof(double));
 int llhood_track_index = 0;

 for (i = 0; i < myparas.num_llhood_track; i++)
 last_n_llhoods[i] = 1.0;

 time_t start_time;
 clock_t start_clock, temp_clock;

 double** cdx = zerosarray(k, d);
 double** delta = zerosarray(k, d);
 double** dv = zerosarray(k, d);
 double* p = (double*)calloc(k, sizeof(double));
 double* tempk = (double*)calloc(k, sizeof(double));
 double* tempd = (double*)calloc(d, sizeof(double));
 double** tempkd = zerosarray(k, d);
 double* dufr = (double*)calloc(d, sizeof(double));

 double temp_val;

 PHASH* tcount;
 double** tcount_full;

 HELEM temp_elem;
 TPAIR temp_pair;

 if(myparas.use_hash_TTable)
 {
 tcount = create_empty_hash(2 * myparas.transition_range  * n);

 printf("Transition matrix created.\n");
 for(i = 0; i < pd.num_playlists; i ++)
 {
 if(pd.playlists_length[i] > 1)
{
    for(j = 0; j < pd.playlists_length[i] - 1; j++)
    {
        for(t = 1; t <= myparas.transition_range; t++)
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

//Sanity check
int temp_int = 0;
for(i = 0; i < tcount -> length; i++)
{
    if(!is_null_entry((tcount -> p)[i].key))
        temp_int++;
} 
printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(k * k)));
printf("Transition matrix initialized.\n");
}
else
{
    tcount_full = zerosarray(k, k);
    printf("Full transition matrix created.\n");
    for(i = 0; i < pd.num_playlists; i ++)
    {
        if(pd.playlists_length[i] > 1)
        {
            for(j = 0; j < pd.playlists_length[i] - 1; j++)
            {
                for(t = 1; t <=myparas.transition_range; t++)
                {
                    if(j + t < pd.playlists_length[i])
                    {
                        fr = pd.playlists[i][j];
                        to = pd.playlists[i][j + t];
                        tcount_full[fr][to] += 1.0 / (double) t;
                    }
                }
            }
        }
    }
    int temp_int = 0;
    for(i = 0; i < k; i++)
        for(j = 0; j < k; j++)
            if(tcount_full[i][j] > 0.0)
                temp_int++;
    printf("Nonezero entry in the transition matrix is %f%%.\n", (100.0 * (float)temp_int) / ((float)(k * k)));
    printf("Full transition matrix initialized.\n");
}

for(t = 0; t < 100000; t++)
{
    start_time = time(NULL);
    start_clock = clock();
    printf("Iteration %d\n", t + 1);fflush(stdout);
    llhoodprev = llhood;
    llhood = 0.0;
    realn = 0.0;
    for(fr = 0; fr < k; fr++)
    {
        for(i = 0; i < k; i++)
            memset(cdx[i], 0, d * sizeof(double));

        for(j = 0; j < k; j++)
            for(i = 0; i < d; i++)
                delta[j][i] = X[fr][i] - X[j][i];

        mat_mult(delta, delta, tempkd, k, d);
        scale_mat(tempkd, k, d, -1.0);
        sum_along_direct(tempkd, p, k, d, 1);
        norm_wo_underflow(p, k, tempk);
        for(j = 0; j < k; j++)
            for(i = 0; i < d; i++)
                dv[j][i] = -2.0 * exp(p[j]) * delta[j][i];
        sum_along_direct(dv, dufr, k, d, 0);
        double accu_temp_vals = 0.0;
        for(to = 0; to < k; to++)
        {
            if(myparas.use_hash_TTable)
            {
                temp_pair.fr = fr;
                temp_pair.to = to;
                idx = exist_in_hash(tcount, temp_pair); 
            }
            else
                idx = tcount_full[fr][to] > 0.0 ? 1 : -1;
            if(idx >= 0)
            {
                if(myparas.use_hash_TTable)
                    temp_val = retrieve_value_with_idx(tcount, idx);
                else
                    temp_val = tcount_full[fr][to];

                accu_temp_vals += temp_val;


                add_vec(cdx[to], delta[to], d, 2.0 * temp_val);
                Veccopy(dufr, tempd, d);
                add_vec(tempd, delta[to], d, 2.0);
                scale_vec(tempd, d, -1.0);
                add_vec(cdx[fr], tempd, d, temp_val);
                llhood += temp_val * p[to];
                realn += temp_val;
            }
        }
        add_mat(cdx, dv, k, d, accu_temp_vals);
        add_mat(X, cdx, k, d, eta / (double)n);
    }

    llhood /= realn;


    printf("The iteration took %d seconds\n", (int)(time(NULL) - start_time));fflush(stdout);
    printf("The iteration took %f seconds of cpu time\n", ((float)clock() - (float)start_clock) / CLOCKS_PER_SEC);fflush(stdout);
    printf("Norms of coordinates: %f\n", frob_norm(X, k, d));fflush(stdout);
    printf("Avg log-likelihood on train: %f\n", llhood);fflush(stdout);
    if(t > 0)
    {
        if(llhoodprev > llhood)
        {
            eta /= myparas.beta;
            //eta = 0.0;
            printf("Reducing eta to: %f\n", eta);
        }
        else if(fabs((llhood - llhoodprev) / llhood) < 0.005)
        {
            eta *= myparas.alpha;
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
    printf("min_llhood %f, max_llhood %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
    printf("magnitude of gradient %f\n", frob_norm(cdx, k, d));

    write_embedding_to_file(X, k, d, embeddingfile, 0);

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
Array2Dfree(cdx, k , d);
Array2Dfree(delta, k , d);
Array2Dfree(dv, k , d);
free(p);
free(tempk);
Array2Dfree(tempkd, k, d);
free(last_n_llhoods);
free(dufr);
free(tempd);
if(myparas.use_hash_TTable)
    free_hash(tcount);
    else
    Array2Dfree(tcount_full, k, k);
    return X;
    }
*/
