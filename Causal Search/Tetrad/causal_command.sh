#!/bin/bash

# Causal CMD Documentation: https://bd2kccd.github.io/docs/causal-cmd/
java\
    -jar "causal-cmd-7.6.8-jar-with-dependencies.jar"\
    --dataset "../../Datasets/tetrad_set_reduced_imputed.csv"\
    --knowledge "../../Datasets/tetrad_bk_reduced.txt"\
    --data-type mixed\
    --delimiter comma\
    --numCategories 2\
    --missing-marker "*"\
    --algorithm boss-fci\
    --score dg-bic-score\
    --test dg-lr-test\
    --fractionResampleSize 10 \
    --resamplingWithReplacement \
    --numberResampling 1\
    --numStarts 1\
    --numThreads 5 \
    --resamplingEnsemble 2\
    --json-graph \
    --seed 4242\
    --out "bfci"\ 