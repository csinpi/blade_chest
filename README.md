-Overview

Blade-chest (BC) is a software developed by Shuo Chen (shuochen@cs.cornell.edu) from Deptment of Computer Science, Cornell University. It learns from sequence data to embed the elements that constitute the sequences. We originally used it in music playlists modeling. Please see the references for more details. This program is granted free of charge for research and education purposes. However you must obtain a license from the authors to use it for commercial purposes. Since it is free, there is no warranty for it.

-Build

A simple "make" will do. It will create a binary ../bin/BC, which serves for training and testing.

-Usage

Format of the game record data:
The format is very intuitive and mostly human-readable, as demonstrated in the files within datasets folder. The first line is the total number of players, followed by lines that contain all the players's IDs(starting from 0) names. Then there is one line for total number of games. The rest are the game records. It takes a "a:b x:y" format, meaning a beats b x times and loses to b y times.

-Bug Report

Please contact the authors if you spot any bug in the software.

-References

If you use the software please cite the following paper:

Shuo Chen, Thorsten Joachims. Modeling Intransitivity in Matchup and Comparison Data. The 9th ACM International Conference on Web Search and Data Mining (WSDM).
