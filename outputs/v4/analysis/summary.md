# v4 analysis summary (estimator = hsel)

## E1 — word position vs answer position (mean above nuisance floor; models above floor; r vs log10 params)

| task|shots | floor | word@neutral (span mean) | word@neutral (last token) | word@question | answer@neutral | answer@question | routing ratio (median, span mean) |
|---|---|---|---|---|---|---|---|
| contains_letter|s0 | 0.6951 |  |  |  | +0.118 (16/16; r=0.661) | +0.128 (15/16; r=0.7438) |  |
| contains_letter|s3 | 0.6776 |  |  |  | +0.103 (16/16; r=0.611) | +0.163 (16/16; r=0.6626) |  |
| count_distinct_lm|s0 | 0.2163 | +0.088 (16/16; r=-0.0535) | +0.077 (16/16; r=0.2532) | +0.090 (16/16; r=0.0112) | +0.072 (16/16; r=0.2184) | +0.079 (16/16; r=0.1682) | 0.8405 |
| count_distinct_lm|s3 | 0.2224 | +0.090 (16/16; r=0.2029) | +0.100 (16/16; r=0.6276) | +0.112 (16/16; r=0.4999) | +0.066 (16/16; r=0.1799) | +0.098 (16/16; r=0.56) | 0.9659 |
| count_distinct|s0 | 0.5414 | -0.200 (0/16; r=-0.0713) | -0.170 (0/16; r=0.4275) | -0.200 (0/16; r=-0.177) | -0.175 (0/16; r=-0.1371) | -0.143 (0/16; r=0.605) |  |
| count_distinct|s3 | 0.5325 | -0.199 (0/16; r=-0.2062) | -0.178 (0/16; r=-0.191) | -0.138 (0/16; r=0.3817) | -0.173 (0/16; r=0.1116) | -0.076 (3/16; r=0.6278) |  |
| counting_lm|s0 | 0.3694 |  |  |  | +0.078 (14/16; r=0.3083) | +0.151 (16/16; r=0.4275) |  |
| counting_lm|s3 | 0.4036 |  |  |  | +0.051 (16/16; r=0.4412) | +0.106 (14/16; r=0.5352) |  |
| counting|s0 | 0.5945 |  |  |  | -0.090 (0/16; r=0.5884) | -0.013 (8/16; r=0.5345) |  |
| counting|s3 | 0.6756 |  |  |  | -0.158 (0/16; r=0.7152) | -0.074 (2/16; r=0.6102) |  |
| extraction|s0 | 0.1999 |  |  |  | -0.006 (7/16; r=-0.2184) | +0.176 (16/16; r=0.7932) |  |
| extraction|s3 | 0.1709 |  |  |  | -0.000 (9/16; r=0.0598) | +0.275 (16/16; r=0.5951) |  |
| first_letter|s0 | 0.4573 | +0.508 (16/16; r=-0.175) | +0.434 (16/16; r=-0.1632) | +0.514 (16/16; r=-0.0177) | +0.436 (16/16; r=-0.4445) | +0.537 (16/16; r=0.393) | 1.0325 |
| first_letter|s3 | 0.475 | +0.492 (16/16; r=-0.298) | +0.415 (16/16; r=-0.2047) | +0.497 (16/16; r=-0.1667) | +0.396 (16/16; r=-0.3558) | +0.522 (16/16; r=0.3928) | 1.0445 |
| last_letter|s0 | 0.5466 | +0.177 (16/16; r=0.3285) | +0.314 (16/16; r=0.4731) | +0.265 (16/16; r=0.6176) | +0.115 (16/16; r=-0.0241) | +0.356 (16/16; r=0.7312) | 2.0329 |
| last_letter|s3 | 0.5516 | +0.157 (16/16; r=0.3355) | +0.296 (16/16; r=0.3513) | +0.242 (16/16; r=0.7036) | -0.004 (6/16; r=-0.115) | +0.393 (16/16; r=0.8053) | 2.6538 |
| letter_before|s0 | 0.4061 |  |  |  | -0.174 (0/16; r=0.0828) | -0.047 (3/16; r=0.537) |  |
| letter_before|s3 | 0.4165 |  |  |  | -0.205 (0/16; r=-0.0828) | +0.007 (8/16; r=0.6588) |  |
| most_frequent_letter|s0 | 0.4423 | +0.165 (16/16; r=0.1407) | +0.135 (16/16; r=-0.045) | +0.181 (16/16; r=0.484) | +0.084 (14/16; r=-0.3655) | +0.283 (16/16; r=0.723) | 1.6882 |
| most_frequent_letter|s3 | 0.4002 | +0.162 (16/16; r=0.1307) | +0.134 (16/16; r=0.0204) | +0.219 (16/16; r=0.6374) | +0.018 (10/16; r=-0.3596) | +0.357 (16/16; r=0.8219) | 2.1473 |
| word_length|s0 | 0.3626 | -0.048 (2/16; r=-0.4718) | +0.003 (7/16; r=0.0558) | -0.049 (2/16; r=-0.4895) | +0.029 (14/16; r=0.0773) | +0.193 (12/16; r=0.478) |  |
| word_length|s3 | 0.3513 | -0.018 (4/16; r=-0.1961) | +0.019 (10/16; r=-0.1112) | +0.086 (13/16; r=0.14) | +0.062 (15/16; r=0.1824) | +0.383 (16/16; r=0.5333) |  |


## E2 — retained fraction (no-question / question), above floor vs above majority

| task|shots | neutral, above floor | neutral, above majority | continuation, above floor | no-q models above floor |
|---|---|---|---|---|
| contains_letter|s0 | 0.9188 | 0.9657 | 1.2333 | 16 |
| contains_letter|s3 | 0.6355 | 0.837 | 0.866 | 16 |
| count_distinct_lm|s0 | 0.9107 | 0.9107 | 1.0819 | 16 |
| count_distinct_lm|s3 | 0.6728 | 0.6728 | 0.6097 | 16 |
| count_distinct|s0 |  | 0.7045 |  | 0 |
| count_distinct|s3 |  | 0.4218 |  | 0 |
| counting_lm|s0 | 0.5199 | 0.6984 | 0.8586 | 14 |
| counting_lm|s3 | 0.4786 | 0.7682 | 0.7826 | 16 |
| counting|s0 |  | 0.7046 |  | 0 |
| counting|s3 |  | 0.7095 |  | 0 |
| extraction|s0 | -0.0363 | 0.2276 | 0.2397 | 7 |
| extraction|s3 | -0.0002 | 0.1508 | 0.0174 | 9 |
| first_letter|s0 | 0.8114 | 0.8847 | 0.8552 | 16 |
| first_letter|s3 | 0.7583 | 0.8556 | 0.6959 | 16 |
| last_letter|s0 | 0.3237 | 0.671 | 0.473 | 16 |
| last_letter|s3 | -0.0093 | 0.5034 | 0.0342 | 6 |
| letter_before|s0 |  | 0.4848 |  | 0 |
| letter_before|s3 |  | 0.2833 |  | 0 |
| most_frequent_letter|s0 | 0.2951 | 0.542 | 0.4669 | 14 |
| most_frequent_letter|s3 | 0.0507 | 0.3235 | 0.104 | 10 |
| word_length|s0 | 0.1495 | 0.546 | 0.4391 | 14 |
| word_length|s3 | 0.1621 | 0.4157 | 0.1747 | 15 |


## E5 — question x scale interaction (slope of question − no-question per decade)

| task|shots | slope | r | p (perm) | within-family slope | family-bootstrap 95% CI | p (ans@q − word@nq, mean) | p (ans@q − word@nq, last) |
|---|---|---|---|---|---|---|---|
| contains_letter|s0 | 0.0237 | 0.4284 | 0.0937 | 0.0149 | [-0.0159, 0.0392] |  |  |
| contains_letter|s3 | 0.0505 | 0.5512 | 0.0267 | 0.0524 | [0.0199, 0.0795] |  |  |
| count_distinct_lm|s0 | 0.0008 | 0.009 | 0.9726 | -0.0046 | [-0.0398, 0.0451] | 0.4636 | 0.9297 |
| count_distinct_lm|s3 | 0.0482 | 0.4814 | 0.0574 | 0.0382 | [0.0148, 0.0931] | 0.0804 | 0.1217 |
| count_distinct|s0 | 0.096 | 0.6961 | 0.003 | 0.1184 | [0.046, 0.1812] | 0.0118 | 0.023 |
| count_distinct|s3 | 0.1049 | 0.6277 | 0.0118 | 0.1466 | [0.0584, 0.1572] | 0.0081 | 0.0021 |
| counting_lm|s0 | 0.0591 | 0.3111 | 0.2469 | 0.0782 | [-0.0135, 0.2087] |  |  |
| counting_lm|s3 | 0.0739 | 0.3996 | 0.1203 | 0.098 | [0.0304, 0.1706] |  |  |
| counting|s0 | 0.079 | 0.4258 | 0.0948 | 0.0806 | [0.0296, 0.1833] |  |  |
| counting|s3 | 0.0574 | 0.3732 | 0.1521 | 0.0788 | [-0.0063, 0.153] |  |  |
| extraction|s0 | 0.1398 | 0.8817 | 0.0001 | 0.1603 | [0.0892, 0.1834] |  |  |
| extraction|s3 | 0.1558 | 0.6236 | 0.0094 | 0.1921 | [0.093, 0.2708] |  |  |
| first_letter|s0 | 0.0755 | 0.4545 | 0.0831 | 0.0594 | [-0.0243, 0.1368] | 0.3978 | 0.4874 |
| first_letter|s3 | 0.0824 | 0.3671 | 0.1603 | 0.0639 | [-0.1007, 0.1736] | 0.2027 | 0.3876 |
| last_letter|s0 | 0.1259 | 0.8015 | 0.0005 | 0.1252 | [0.0535, 0.1587] | 0.0103 | 0.0168 |
| last_letter|s3 | 0.1324 | 0.7713 | 0.0008 | 0.1289 | [0.0555, 0.1861] | 0.0002 | 0.0007 |
| letter_before|s0 | 0.094 | 0.583 | 0.0153 | 0.1044 | [0.0212, 0.1701] |  |  |
| letter_before|s3 | 0.1348 | 0.6976 | 0.0018 | 0.1448 | [0.0793, 0.1972] |  |  |
| most_frequent_letter|s0 | 0.1915 | 0.7692 | 0.0006 | 0.1726 | [0.0835, 0.2411] | 0.0116 | 0.0109 |
| most_frequent_letter|s3 | 0.1709 | 0.7293 | 0.0011 | 0.1569 | [0.0907, 0.2664] | 0.0016 | 0.0004 |
| word_length|s0 | 0.2444 | 0.4966 | 0.052 | 0.3377 | [0.1188, 0.4105] | 0.0293 | 0.0653 |
| word_length|s3 | 0.2442 | 0.5117 | 0.0444 | 0.3265 | [0.0872, 0.5046] | 0.0201 | 0.0164 |


## E3 — representation−readout gap vs scale (r, permutation p)

| task|shots | gap | logit gap | gap, unsaturated | gap, base only |
|---|---|---|---|---|
| contains_letter|s0 | 0.1923 (p=0.48, n=16) | 0.3968 (p=0.1318, n=16) | 0.1923 (p=0.48, n=16) | -0.0118 (p=0.9732, n=10) |
| contains_letter|s3 | -0.2041 (p=0.4493, n=16) | 0.0134 (p=0.9586, n=16) | -0.1037 (p=0.7178, n=15) | -0.165 (p=0.6385, n=10) |
| count_distinct_lm|s0 | -0.626 (p=0.0094, n=16) | -0.7614 (p=0.0009, n=16) | -0.626 (p=0.0094, n=16) | -0.6135 (p=0.0656, n=10) |
| count_distinct_lm|s3 | -0.1053 (p=0.6988, n=16) | -0.3258 (p=0.2267, n=16) | -0.1053 (p=0.6988, n=16) | 0.3297 (p=0.3534, n=10) |
| count_distinct|s0 | -0.6183 (p=0.0122, n=16) | -0.6357 (p=0.0068, n=16) | -0.6183 (p=0.0122, n=16) | -0.4605 (p=0.1804, n=10) |
| count_distinct|s3 | -0.2346 (p=0.3788, n=16) | -0.2801 (p=0.2897, n=16) | -0.2346 (p=0.3788, n=16) | 0.0506 (p=0.8829, n=10) |
| counting_lm|s0 | 0.0837 (p=0.772, n=16) | 0.1472 (p=0.6347, n=16) | 0.0837 (p=0.772, n=16) | 0.1514 (p=0.7798, n=10) |
| counting_lm|s3 | 0.1301 (p=0.6241, n=16) | 0.1172 (p=0.6591, n=16) | 0.1301 (p=0.6241, n=16) | 0.2392 (p=0.4973, n=10) |
| counting|s0 | 0.0653 (p=0.8195, n=16) | 0.1309 (p=0.6606, n=16) | 0.0653 (p=0.8195, n=16) | 0.2403 (p=0.5687, n=10) |
| counting|s3 | -0.1054 (p=0.6931, n=16) | -0.0846 (p=0.7536, n=16) | -0.1054 (p=0.6931, n=16) | -0.0858 (p=0.813, n=10) |
| extraction|s0 | 0.1051 (p=0.693, n=16) | -0.1022 (p=0.6991, n=16) | 0.1051 (p=0.693, n=16) | 0.2691 (p=0.4433, n=10) |
| extraction|s3 | -0.3591 (p=0.1745, n=16) | -0.3587 (p=0.1737, n=16) | -0.3591 (p=0.1745, n=16) | -0.3904 (p=0.2692, n=10) |
| first_letter|s0 | 0.234 (p=0.3849, n=16) | 0.2937 (p=0.2648, n=16) |  | -0.1973 (p=0.582, n=10) |
| first_letter|s3 | -0.3616 (p=0.1707, n=16) | -0.24 (p=0.3819, n=16) |  | -0.5496 (p=0.1049, n=10) |
| last_letter|s0 | 0.0612 (p=0.8232, n=16) | -0.0142 (p=0.9578, n=16) | 0.4844 (p=0.1533, n=10) | 0.1717 (p=0.6248, n=10) |
| last_letter|s3 | -0.8266 (p=0.0002, n=16) | -0.4338 (p=0.0912, n=16) | -0.5852 (p=0.2258, n=6) | -0.9258 (p=0.0003, n=10) |
| letter_before|s0 | 0.4278 (p=0.0996, n=16) | 0.2044 (p=0.4532, n=16) | 0.4278 (p=0.0996, n=16) | 0.1357 (p=0.7062, n=10) |
| letter_before|s3 | 0.5389 (p=0.0327, n=16) | 0.4741 (p=0.0643, n=16) | 0.5389 (p=0.0327, n=16) | 0.821 (p=0.0034, n=10) |
| most_frequent_letter|s0 | -0.5329 (p=0.0363, n=16) | -0.5303 (p=0.039, n=16) | -0.5329 (p=0.0363, n=16) | -0.8025 (p=0.0076, n=10) |
| most_frequent_letter|s3 | -0.5063 (p=0.0471, n=16) | -0.4079 (p=0.1159, n=16) | -0.5063 (p=0.0471, n=16) | -0.2624 (p=0.4541, n=10) |
| word_length|s0 | -0.7227 (p=0.0022, n=16) | -0.5758 (p=0.0071, n=16) | -0.7227 (p=0.0022, n=16) | -0.7424 (p=0.0126, n=10) |
| word_length|s3 | -0.5558 (p=0.0189, n=16) | -0.5225 (p=0.0335, n=16) | -0.472 (p=0.0824, n=14) | -0.264 (p=0.4679, n=10) |


## E4 — estimator: selection optimism (peak − held-out-selected)

| task|shots | mean optimism | r vs depth | r vs log params | v4 − v3 representation |
|---|---|---|---|---|
| contains_letter|s0 | 0.0057 | -0.0211 | -0.1439 |  |
| contains_letter|s3 | 0.0075 | -0.1535 | -0.2808 | -0.0207 |
| count_distinct_lm|s0 | 0.0191 | -0.2623 | 0.0903 |  |
| count_distinct_lm|s3 | 0.0156 | -0.0636 | 0.1096 |  |
| count_distinct|s0 | 0.0127 | -0.2974 | -0.4069 |  |
| count_distinct|s3 | 0.014 | -0.4842 | -0.4308 | -0.012 |
| counting_lm|s0 | 0.018 | 0.1862 | 0.3425 |  |
| counting_lm|s3 | 0.0134 | -0.0776 | -0.1436 |  |
| counting|s0 | 0.0105 | -0.1145 | 0.0771 |  |
| counting|s3 | 0.0067 | -0.1855 | -0.3325 | 0.0026 |
| extraction|s0 | 0.004 | -0.1711 | -0.3367 |  |
| extraction|s3 | 0.0027 | -0.4405 | -0.5925 | -0.0035 |
| first_letter|s0 | 0.0017 | 0.1384 | 0.0049 |  |
| first_letter|s3 | 0.001 | 0.0611 | 0.1502 | -0.0006 |
| last_letter|s0 | 0.0027 | -0.2636 | -0.1742 |  |
| last_letter|s3 | 0.0008 | -0.4436 | 0.0892 | 0.0044 |
| letter_before|s0 | 0.0063 | 0.4754 | 0.1421 |  |
| letter_before|s3 | 0.0043 | -0.1343 | -0.1996 | -0.0022 |
| most_frequent_letter|s0 | 0.004 | -0.0822 | 0.0505 |  |
| most_frequent_letter|s3 | 0.0069 | -0.2876 | -0.2104 | -0.0 |
| word_length|s0 | 0.0063 | 0.226 | 0.1083 |  |
| word_length|s3 | 0.0026 | -0.5677 | -0.2005 | -0.0002 |


## E6 — above nuisance floor, every task (grid, question arm)

| task|shots | majority | floor | floor predictor | models above floor | best above floor |
|---|---|---|---|---|---|
| contains_letter|s0 | 0.52 | 0.6951 | length+letterset | 15/16 | 0.2386 |
| contains_letter|s3 | 0.4763 | 0.6776 | length+letterset | 16/16 | 0.2724 |
| count_distinct_lm|s0 | 0.2163 | 0.2163 | majority | 16/16 | 0.1474 |
| count_distinct_lm|s3 | 0.2224 | 0.2224 | majority | 16/16 | 0.192 |
| count_distinct|s0 | 0.2907 | 0.5414 | length | 0/16 | -0.0439 |
| count_distinct|s3 | 0.2895 | 0.5325 | length | 3/16 | 0.0326 |
| counting_lm|s0 | 0.2801 | 0.3694 | letterset | 16/16 | 0.3575 |
| counting_lm|s3 | 0.2712 | 0.4036 | letterset | 14/16 | 0.2585 |
| counting|s0 | 0.3221 | 0.5945 | length+letterset | 8/16 | 0.2312 |
| counting|s3 | 0.3108 | 0.6756 | length+letterset | 2/16 | 0.0938 |
| extraction|s0 | 0.1396 | 0.1999 | letterset | 16/16 | 0.2959 |
| extraction|s3 | 0.1221 | 0.1709 | length+letterset | 16/16 | 0.5035 |
| first_letter|s0 | 0.1155 | 0.4573 | length+letterset | 16/16 | 0.5414 |
| first_letter|s3 | 0.1235 | 0.475 | length+letterset | 16/16 | 0.525 |
| last_letter|s0 | 0.1706 | 0.5466 | letterset | 16/16 | 0.4394 |
| last_letter|s3 | 0.1455 | 0.5516 | length+letterset | 16/16 | 0.4471 |
| letter_before|s0 | 0.1116 | 0.4061 | letterset+q | 3/16 | 0.1904 |
| letter_before|s3 | 0.1278 | 0.4165 | length+letterset | 8/16 | 0.1873 |
| most_frequent_letter|s0 | 0.2898 | 0.4423 | length+letterset | 16/16 | 0.4397 |
| most_frequent_letter|s3 | 0.2564 | 0.4002 | length+letterset | 16/16 | 0.4828 |
| word_length|s0 | 0.1938 | 0.3626 | letterset | 12/16 | 0.5187 |
| word_length|s3 | 0.185 | 0.3513 | letterset | 16/16 | 0.635 |


## ACC — answer-token subspace vs rank-matched comparators (v3 section 8)

| shots | cells | proj − random (wins) | proj − PCA (wins) | overlap (median) | r(gap, deficit) | r, unsaturated | deficit by gap tertile | proj − random by gap tertile |
|---|---|---|---|---|---|---|---|---|
| s0 | 176 | +0.100 (159) | +0.012 (97) | 0.0278 | 0.4819 | 0.8008 | [0.0484, 0.0891, 0.1509] | [0.1019, 0.0698, 0.1288] |
| s3 | 176 | +0.119 (159) | +0.023 (113) | 0.0615 | 0.6435 | 0.5991 | [0.015, 0.0532, 0.1006] | [0.1468, 0.1056, 0.1059] |


## X — cross-task decomposition (shots=0, zero-leak cells, share of total)

| probed label | total Δ | generic frame | wrong-property Q | specific Q | cells | models |
|---|---|---|---|---|---|---|
| count_distinct | 0.0957 | 0.1385 | 0.4341 | 0.4274 | 18 | 6 |
| first_letter | 0.1366 | 0.4665 | 0.3198 | 0.2137 | 20 | 10 |
| last_letter | 0.2407 | 0.2206 | 0.0815 | 0.6979 | 32 | 16 |
| most_frequent_letter | 0.2275 | 0.2168 | 0.3826 | 0.4006 | 28 | 14 |
| word_length | 0.2962 | 0.2778 | 0.2508 | 0.4714 | 30 | 10 |


## X_term — cross-task decomposition (shots=0, zero-leak cells, share of total)

| probed label | total Δ | generic frame | wrong-property Q | specific Q | cells | models |
|---|---|---|---|---|---|---|
| extraction | 0.2209 | 0.335 | -0.0497 | 0.7147 | 32 | 16 |
| letter_before | 0.1425 | 0.2907 | 0.0907 | 0.6186 | 28 | 14 |


## Coverage

```
{
 "grid_files": 96,
 "wordpos_files": 96,
 "align_files": 96,
 "crosstask_files": 32,
 "crosstask_term_files": 32,
 "grid_models": [
  "Llama_3_1_8B",
  "Llama_3_2_1B",
  "Llama_3_2_3B",
  "OLMoE_1B_7B_0924",
  "OLMo_2_0425_1B",
  "OLMo_2_1124_13B",
  "OLMo_2_1124_7B",
  "Olmo_Hybrid_7B",
  "Phi_4_mini_instruct",
  "Qwen3_0_6B",
  "Qwen3_1_7B",
  "Qwen3_4B",
  "Qwen3_8B",
  "gemma_2_2b",
  "gemma_2_9b",
  "phi_4"
 ],
 "wordpos_models": [
  "Llama_3_1_8B",
  "Llama_3_2_1B",
  "Llama_3_2_3B",
  "OLMoE_1B_7B_0924",
  "OLMo_2_0425_1B",
  "OLMo_2_1124_13B",
  "OLMo_2_1124_7B",
  "Olmo_Hybrid_7B",
  "Phi_4_mini_instruct",
  "Qwen3_0_6B",
  "Qwen3_1_7B",
  "Qwen3_4B",
  "Qwen3_8B",
  "gemma_2_2b",
  "gemma_2_9b",
  "phi_4"
 ],
 "missing_wordpos_models": [],
 "missing_grid_models": []
}
```
