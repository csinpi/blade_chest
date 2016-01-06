# Blade-chest model

## Overview

Blade-chest (BC) is a software developed by [Shuo Chen](http://www.cs.cornell.edu/~shuochen/) from the Department of Computer Science, Cornell University. It learns from matchup or comparison data for future prediction. It is capable of handling intransitivity which is not covered in many conventional methods. Please see the [our paper](http://www.cs.cornell.edu/~shuochen/pubs/wsdm16_chen.pdf) for more details. This program is granted free of charge for research and education purposes. However you must obtain a license from the authors to use it for commercial purposes. Since it is free, there is no warranty for it.

## Build

A simple "make" will do. It will create a binary ../bin/BC, which serves for training and testing.

## Usage

### Format of the game record data

The format is very intuitive and mostly human-readable, as demonstrated in the files within datasets folder. The first line is the total number of players, followed by lines that contain all the players' IDs(starting from 0) names. Then there is one line for total number of games. The rest are the game records. Each record could be prefixed with a tag from "FOR\_TRAINING", "FOR\_VALIDATION" or "FOR\_TESTING", indicating what this line of record is for. Otherwise, when no record is prefixed, the records will be randomly divided into 5:2:3 ratio for training, validation and testing. The main body of the record takes a "a:b x:y" format, meaning a beats b x times and loses to b y times.

Format of the matchup matrix data (optional, only useful for running the matrix reconstruction experiment in our paper):
The format is intuitive as well. One can see it from the example file datasets/sf4/sf4mat.txt. It is a matrix of integers in [1, 9], and diagonal elements are 0s. Rows are separated with '\n' and columns with ' '.

### Running the program

BC is used in the following format:

BC [options] data\_file model\_file

Available options are:

-d						int               Dimensionality of the embedding (default 2)

-e						float             Error allowed for termination (default 1e-4)

-i						float             Learning rate (default 1e-2)

-l						float             Regularization coefficient (default 0.0)

-r						[0, 1]            Including the bias terms (1) or not (0) (default 1)

-S						int               The seed for random number generator to create different training, validation, and testing split (default 0)

-a						float             Adaptively increase the learning rate by this number if the improvement of the training objective function is too small (default 1.1, not recomeended to change if you run the code on our datasets)

-b						float             Adaptively decrease the learning rate by this number if the training likelihood deteriorates (default 2.0, not recomeended to change if you run the code on our datasets)

-t						[0, 1, 2]         0: 2-norm regularizer on blade and chest vectors 
  						                  1: regularizer on the distances between blade and chest vectors  (default)
  						                  2: sum of type 0 and type 1 as regularizer 

-M						[0, 2]            dist model (0) or inner model (2) (default 0. Note that 1 is another matchup function we experiemnted with, but did perform as well. You can try it if you like.)


-E						path              the matchup matrix data if you are running the reconstruction experiment (default '\0', meaning it is not used)  

### Outputs

There are two outputs: training log and validation/testing results printed to stdout, and model stored in model\_file.

The log is human-readable, it contains information for each training iteration, test/validation log-likelihood and accuracy, results for the naive baslines (as a bonus), and the reconstruction results if you use -E to provide a matchup matrix file. Note that it is possible to run Bradley-Terry model by simply using "-d 0 -r 1" option.  

The first four lines of the model file contain numplayers, d, rankon (usd bias terms or not) and modeltype (basically -M). They are followed by tvecs and hvecs, the chest vectors and blade vectors (we called them tail vectors and head vectors initially). In the end, there is one line of "ranks" you turned -r on.

## Datasets

These datasets are collected and processed by [Shuo Chen](http://www.cs.cornell.edu/~shuochen/) from multiple public sources on the internet. Every dataset used in our paper is under /datasets except the peer grading ones which we do not have right to release. We do not own these data. Please cite each of the individual source if you use them for research or education purposes, and contact the source for any commercial use. Please see [our paper](http://www.cs.cornell.edu/~shuochen/pubs/wsdm16_chen.pdf) for details on the sources.   



## Bug Report

Please contact the author if you spot any bug in the software.

## References

If you use the software please cite the following paper:

[Shuo Chen, Thorsten Joachims. Modeling Intransitivity in Matchup and Comparison Data. The 9th ACM International Conference on Web Search and Data Mining (WSDM)](http://www.cs.cornell.edu/~shuochen/pubs/wsdm16_chen.pdf)
