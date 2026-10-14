# v4 analysis summary (estimator = peak)

## E1 — word position vs answer position (mean above nuisance floor; models above floor; r vs log10 params)

| task|shots | floor | word@neutral (span mean) | word@neutral (last token) | word@question | answer@neutral | answer@question | routing ratio (median, span mean) |
|---|---|---|---|---|---|---|---|
| contains_letter|s0 | 0.6951 |  |  |  | +0.124 (16/16; r=0.6465) | +0.134 (16/16; r=0.7413) |  |
| contains_letter|s3 | 0.6776 |  |  |  | +0.110 (16/16; r=0.5912) | +0.170 (16/16; r=0.6498) |  |
| count_distinct_lm|s0 | 0.2163 | +0.107 (16/16; r=0.1167) | +0.097 (16/16; r=0.2487) | +0.107 (16/16; r=0.1331) | +0.091 (16/16; r=-0.0335) | +0.098 (16/16; r=0.3359) | 0.8879 |
| count_distinct_lm|s3 | 0.2224 | +0.109 (16/16; r=0.4332) | +0.113 (16/16; r=0.7586) | +0.131 (16/16; r=0.5554) | +0.080 (16/16; r=0.285) | +0.114 (16/16; r=0.6042) | 0.9054 |
| count_distinct|s0 | 0.5414 | -0.185 (0/16; r=-0.1318) | -0.156 (0/16; r=0.3208) | -0.184 (0/16; r=-0.1809) | -0.157 (0/16; r=-0.1115) | -0.130 (0/16; r=0.588) |  |
| count_distinct|s3 | 0.5325 | -0.185 (0/16; r=-0.2791) | -0.163 (0/16; r=-0.1992) | -0.129 (0/16; r=0.3782) | -0.158 (0/16; r=0.1442) | -0.062 (4/16; r=0.6108) |  |
| counting_lm|s0 | 0.3694 |  |  |  | +0.110 (16/16; r=0.3851) | +0.169 (16/16; r=0.5283) |  |
| counting_lm|s3 | 0.4036 |  |  |  | +0.069 (16/16; r=0.579) | +0.119 (16/16; r=0.5555) |  |
| counting|s0 | 0.5945 |  |  |  | -0.070 (0/16; r=0.4064) | -0.003 (8/16; r=0.5576) |  |
| counting|s3 | 0.6756 |  |  |  | -0.143 (0/16; r=0.6563) | -0.067 (2/16; r=0.6095) |  |
| extraction|s0 | 0.1999 |  |  |  | +0.002 (8/16; r=-0.2801) | +0.180 (16/16; r=0.7847) |  |
| extraction|s3 | 0.1709 |  |  |  | +0.009 (12/16; r=0.0623) | +0.278 (16/16; r=0.5804) |  |
| first_letter|s0 | 0.4573 | +0.510 (16/16; r=-0.1533) | +0.436 (16/16; r=-0.1596) | +0.516 (16/16; r=-0.0203) | +0.438 (16/16; r=-0.4282) | +0.539 (16/16; r=0.4521) | 1.0336 |
| first_letter|s3 | 0.475 | +0.493 (16/16; r=-0.2479) | +0.417 (16/16; r=-0.1836) | +0.499 (16/16; r=-0.1038) | +0.397 (16/16; r=-0.3644) | +0.523 (16/16; r=0.4766) | 1.043 |
| last_letter|s0 | 0.5466 | +0.186 (16/16; r=0.4718) | +0.319 (16/16; r=0.58) | +0.268 (16/16; r=0.601) | +0.116 (16/16; r=-0.0393) | +0.359 (16/16; r=0.7362) | 1.9635 |
| last_letter|s3 | 0.5516 | +0.161 (16/16; r=0.3698) | +0.300 (16/16; r=0.3248) | +0.247 (16/16; r=0.7182) | -0.003 (6/16; r=-0.1053) | +0.394 (16/16; r=0.8129) | 2.5528 |
| letter_before|s0 | 0.4061 |  |  |  | -0.166 (0/16; r=0.1663) | -0.041 (3/16; r=0.548) |  |
| letter_before|s3 | 0.4165 |  |  |  | -0.197 (0/16; r=-0.0198) | +0.012 (8/16; r=0.6562) |  |
| most_frequent_letter|s0 | 0.4423 | +0.169 (16/16; r=0.202) | +0.144 (16/16; r=-0.0273) | +0.187 (16/16; r=0.4998) | +0.086 (14/16; r=-0.3846) | +0.287 (16/16; r=0.7175) | 1.6978 |
| most_frequent_letter|s3 | 0.4002 | +0.166 (16/16; r=0.1519) | +0.140 (16/16; r=-0.0775) | +0.223 (16/16; r=0.653) | +0.026 (11/16; r=-0.3802) | +0.364 (16/16; r=0.829) | 2.1671 |
| word_length|s0 | 0.3626 | -0.033 (2/16; r=-0.3338) | +0.014 (9/16; r=0.0272) | -0.034 (2/16; r=-0.353) | +0.037 (14/16; r=0.1098) | +0.200 (12/16; r=0.4904) |  |
| word_length|s3 | 0.3513 | -0.009 (5/16; r=-0.1264) | +0.031 (14/16; r=-0.0757) | +0.090 (14/16; r=0.1464) | +0.069 (16/16; r=0.1704) | +0.386 (16/16; r=0.5355) |  |


## E2 — retained fraction (no-question / question), above floor vs above majority

| task|shots | neutral, above floor | neutral, above majority | continuation, above floor | no-q models above floor |
|---|---|---|---|---|
| contains_letter|s0 | 0.9315 | 0.9703 | 1.2377 | 16 |
| contains_letter|s3 | 0.6453 | 0.8374 | 0.8688 | 16 |
| count_distinct_lm|s0 | 0.9294 | 0.9294 | 1.0346 | 16 |
| count_distinct_lm|s3 | 0.7043 | 0.7043 | 0.7246 | 16 |
| count_distinct|s0 |  | 0.7734 |  | 0 |
| count_distinct|s3 |  | 0.4729 |  | 0 |
| counting_lm|s0 | 0.6516 | 0.7721 | 0.8551 | 16 |
| counting_lm|s3 | 0.5767 | 0.7993 | 0.822 | 16 |
| counting|s0 |  | 0.7523 |  | 0 |
| counting|s3 |  | 0.7441 |  | 0 |
| extraction|s0 | 0.0081 | 0.2566 | 0.2839 | 8 |
| extraction|s3 | 0.0342 | 0.1788 | 0.0443 | 12 |
| first_letter|s0 | 0.8113 | 0.8845 | 0.8553 | 16 |
| first_letter|s3 | 0.7601 | 0.8566 | 0.699 | 16 |
| last_letter|s0 | 0.3237 | 0.6697 | 0.4768 | 16 |
| last_letter|s3 | -0.0068 | 0.5041 | 0.0517 | 6 |
| letter_before|s0 |  | 0.5059 |  | 0 |
| letter_before|s3 |  | 0.3066 |  | 0 |
| most_frequent_letter|s0 | 0.3014 | 0.5438 | 0.4652 | 14 |
| most_frequent_letter|s3 | 0.0705 | 0.3339 | 0.1123 | 11 |
| word_length|s0 | 0.1849 | 0.5584 | 0.4679 | 14 |
| word_length|s3 | 0.1788 | 0.4261 | 0.1965 | 16 |


## E5 — question x scale interaction (slope of question − no-question per decade)

| task|shots | slope | r | p (perm) | within-family slope | family-bootstrap 95% CI | p (ans@q − word@nq, mean) | p (ans@q − word@nq, last) |
|---|---|---|---|---|---|---|---|
| contains_letter|s0 | 0.026 | 0.5736 | 0.0198 | 0.0216 | [-0.005, 0.0387] |  |  |
| contains_letter|s3 | 0.0497 | 0.5724 | 0.0194 | 0.0522 | [0.0204, 0.0802] |  |  |
| count_distinct_lm|s0 | 0.0174 | 0.2929 | 0.2723 | 0.0114 | [-0.0079, 0.0366] | 0.4221 | 0.3961 |
| count_distinct_lm|s3 | 0.052 | 0.6008 | 0.0128 | 0.055 | [0.0264, 0.0792] | 0.0467 | 0.0576 |
| count_distinct|s0 | 0.0799 | 0.7317 | 0.001 | 0.0956 | [0.047, 0.1334] | 0.0072 | 0.0159 |
| count_distinct|s3 | 0.0943 | 0.6125 | 0.0135 | 0.1299 | [0.0537, 0.1464] | 0.0057 | 0.0034 |
| counting_lm|s0 | 0.0778 | 0.4929 | 0.0491 | 0.0877 | [0.0281, 0.1716] |  |  |
| counting_lm|s3 | 0.0602 | 0.3947 | 0.128 | 0.0713 | [0.0214, 0.124] |  |  |
| counting|s0 | 0.0999 | 0.5699 | 0.0196 | 0.1032 | [0.0553, 0.2013] |  |  |
| counting|s3 | 0.0629 | 0.4762 | 0.0601 | 0.079 | [0.0162, 0.1336] |  |  |
| extraction|s0 | 0.138 | 0.8734 | 0.0001 | 0.158 | [0.0767, 0.1751] |  |  |
| extraction|s3 | 0.1502 | 0.6031 | 0.0127 | 0.1833 | [0.0809, 0.2766] |  |  |
| first_letter|s0 | 0.0732 | 0.4427 | 0.0907 | 0.0566 | [-0.0239, 0.1341] | 0.4433 | 0.4896 |
| first_letter|s3 | 0.0842 | 0.3796 | 0.1458 | 0.0659 | [-0.0946, 0.1737] | 0.2452 | 0.4182 |
| last_letter|s0 | 0.1251 | 0.8119 | 0.0005 | 0.1257 | [0.0639, 0.1545] | 0.0091 | 0.0212 |
| last_letter|s3 | 0.1314 | 0.7819 | 0.0007 | 0.1268 | [0.0566, 0.185] | 0.0004 | 0.0002 |
| letter_before|s0 | 0.0931 | 0.5924 | 0.0127 | 0.1039 | [0.0291, 0.176] |  |  |
| letter_before|s3 | 0.127 | 0.7064 | 0.0015 | 0.1331 | [0.07, 0.1926] |  |  |
| most_frequent_letter|s0 | 0.1916 | 0.7752 | 0.0005 | 0.168 | [0.083, 0.24] | 0.0107 | 0.0106 |
| most_frequent_letter|s3 | 0.1658 | 0.7405 | 0.0012 | 0.1516 | [0.096, 0.2533] | 0.0006 | 0.0001 |
| word_length|s0 | 0.2451 | 0.5214 | 0.0397 | 0.3317 | [0.1197, 0.4161] | 0.0293 | 0.0498 |
| word_length|s3 | 0.2451 | 0.5271 | 0.0347 | 0.3292 | [0.0959, 0.4846] | 0.0188 | 0.0175 |


## E3 — representation−readout gap vs scale (r, permutation p)

| task|shots | gap | logit gap | gap, unsaturated | gap, base only |
|---|---|---|---|---|
| contains_letter|s0 | 0.1882 (p=0.4885, n=16) | 0.4053 (p=0.1237, n=16) | 0.1882 (p=0.4885, n=16) | -0.0135 (p=0.9714, n=10) |
| contains_letter|s3 | -0.2273 (p=0.3934, n=16) | 0.0333 (p=0.9028, n=16) | -0.1284 (p=0.6559, n=15) | -0.1961 (p=0.5805, n=10) |
| count_distinct_lm|s0 | -0.6325 (p=0.0092, n=16) | -0.7648 (p=0.0007, n=16) | -0.6325 (p=0.0092, n=16) | -0.645 (p=0.0536, n=10) |
| count_distinct_lm|s3 | -0.0846 (p=0.7619, n=16) | -0.3231 (p=0.2327, n=16) | -0.0846 (p=0.7619, n=16) | 0.4192 (p=0.2255, n=10) |
| count_distinct|s0 | -0.6425 (p=0.0084, n=16) | -0.6429 (p=0.0061, n=16) | -0.6425 (p=0.0084, n=16) | -0.4782 (p=0.1626, n=10) |
| count_distinct|s3 | -0.3097 (p=0.2387, n=16) | -0.3548 (p=0.1749, n=16) | -0.3097 (p=0.2387, n=16) | -0.0295 (p=0.9311, n=10) |
| counting_lm|s0 | 0.1514 (p=0.5915, n=16) | 0.179 (p=0.5639, n=16) | 0.1514 (p=0.5915, n=16) | 0.2376 (p=0.5693, n=10) |
| counting_lm|s3 | 0.1024 (p=0.6987, n=16) | 0.0872 (p=0.7455, n=16) | 0.1024 (p=0.6987, n=16) | 0.3198 (p=0.3775, n=10) |
| counting|s0 | 0.071 (p=0.8055, n=16) | 0.1355 (p=0.6497, n=16) | 0.071 (p=0.8055, n=16) | 0.2642 (p=0.4929, n=10) |
| counting|s3 | -0.1528 (p=0.5712, n=16) | -0.1273 (p=0.6351, n=16) | -0.1528 (p=0.5712, n=16) | -0.1383 (p=0.6916, n=10) |
| extraction|s0 | 0.0818 (p=0.7557, n=16) | -0.1204 (p=0.6479, n=16) | 0.0818 (p=0.7557, n=16) | 0.2182 (p=0.5403, n=10) |
| extraction|s3 | -0.4369 (p=0.0928, n=16) | -0.4373 (p=0.0936, n=16) | -0.4369 (p=0.0928, n=16) | -0.4645 (p=0.1776, n=10) |
| first_letter|s0 | 0.2338 (p=0.3853, n=16) | 0.3844 (p=0.1423, n=16) |  | -0.1996 (p=0.5786, n=10) |
| first_letter|s3 | -0.3413 (p=0.1984, n=16) | -0.1351 (p=0.6351, n=16) |  | -0.5547 (p=0.1034, n=10) |
| last_letter|s0 | 0.0513 (p=0.8505, n=16) | 0.0007 (p=0.9973, n=16) | 0.4795 (p=0.1573, n=10) | 0.1483 (p=0.6719, n=10) |
| last_letter|s3 | -0.8242 (p=0.0002, n=16) | -0.4299 (p=0.0947, n=16) | -0.5708 (p=0.2447, n=6) | -0.9226 (p=0.0005, n=10) |
| letter_before|s0 | 0.445 (p=0.0847, n=16) | 0.2231 (p=0.4113, n=16) | 0.445 (p=0.0847, n=16) | 0.2777 (p=0.4303, n=10) |
| letter_before|s3 | 0.518 (p=0.0412, n=16) | 0.4432 (p=0.0859, n=16) | 0.518 (p=0.0412, n=16) | 0.7931 (p=0.0064, n=10) |
| most_frequent_letter|s0 | -0.5331 (p=0.036, n=16) | -0.5284 (p=0.0393, n=16) | -0.5331 (p=0.036, n=16) | -0.8048 (p=0.0076, n=10) |
| most_frequent_letter|s3 | -0.5383 (p=0.0324, n=16) | -0.4345 (p=0.0938, n=16) | -0.5383 (p=0.0324, n=16) | -0.312 (p=0.3677, n=10) |
| word_length|s0 | -0.7199 (p=0.0024, n=16) | -0.573 (p=0.0075, n=16) | -0.7199 (p=0.0024, n=16) | -0.7401 (p=0.0106, n=10) |
| word_length|s3 | -0.5646 (p=0.0163, n=16) | -0.5285 (p=0.0306, n=16) | -0.48 (p=0.0771, n=14) | -0.3041 (p=0.4019, n=10) |


## E4 — estimator: selection optimism (peak − held-out-selected)

| task|shots | mean optimism | r vs depth | r vs log params | v4 − v3 representation |
|---|---|---|---|---|
| contains_letter|s0 | 0.0057 | -0.0211 | -0.1439 |  |
| contains_letter|s3 | 0.0075 | -0.1535 | -0.2808 | -0.0132 |
| count_distinct_lm|s0 | 0.0191 | -0.2623 | 0.0903 |  |
| count_distinct_lm|s3 | 0.0156 | -0.0636 | 0.1096 |  |
| count_distinct|s0 | 0.0127 | -0.2974 | -0.4069 |  |
| count_distinct|s3 | 0.014 | -0.4842 | -0.4308 | 0.002 |
| counting_lm|s0 | 0.018 | 0.1862 | 0.3425 |  |
| counting_lm|s3 | 0.0134 | -0.0776 | -0.1436 |  |
| counting|s0 | 0.0105 | -0.1145 | 0.0771 |  |
| counting|s3 | 0.0067 | -0.1855 | -0.3325 | 0.0093 |
| extraction|s0 | 0.004 | -0.1711 | -0.3367 |  |
| extraction|s3 | 0.0027 | -0.4405 | -0.5925 | -0.0009 |
| first_letter|s0 | 0.0017 | 0.1384 | 0.0049 |  |
| first_letter|s3 | 0.001 | 0.0611 | 0.1502 | 0.0003 |
| last_letter|s0 | 0.0027 | -0.2636 | -0.1742 |  |
| last_letter|s3 | 0.0008 | -0.4436 | 0.0892 | 0.0052 |
| letter_before|s0 | 0.0063 | 0.4754 | 0.1421 |  |
| letter_before|s3 | 0.0043 | -0.1343 | -0.1996 | 0.002 |
| most_frequent_letter|s0 | 0.004 | -0.0822 | 0.0505 |  |
| most_frequent_letter|s3 | 0.0069 | -0.2876 | -0.2104 | 0.0069 |
| word_length|s0 | 0.0063 | 0.226 | 0.1083 |  |
| word_length|s3 | 0.0026 | -0.5677 | -0.2005 | 0.0024 |


## E6 — above nuisance floor, every task (grid, question arm)

| task|shots | majority | floor | floor predictor | models above floor | best above floor |
|---|---|---|---|---|---|
| contains_letter|s0 | 0.52 | 0.6951 | length+letterset | 16/16 | 0.2449 |
| contains_letter|s3 | 0.4763 | 0.6776 | length+letterset | 16/16 | 0.2799 |
| count_distinct_lm|s0 | 0.2163 | 0.2163 | majority | 16/16 | 0.1474 |
| count_distinct_lm|s3 | 0.2224 | 0.2224 | majority | 16/16 | 0.1999 |
| count_distinct|s0 | 0.2907 | 0.5414 | length | 0/16 | -0.0427 |
| count_distinct|s3 | 0.2895 | 0.5325 | length | 4/16 | 0.0338 |
| counting_lm|s0 | 0.2801 | 0.3694 | letterset | 16/16 | 0.3823 |
| counting_lm|s3 | 0.2712 | 0.4036 | letterset | 16/16 | 0.2585 |
| counting|s0 | 0.3221 | 0.5945 | length+letterset | 8/16 | 0.2351 |
| counting|s3 | 0.3108 | 0.6756 | length+letterset | 2/16 | 0.0938 |
| extraction|s0 | 0.1396 | 0.1999 | letterset | 16/16 | 0.2959 |
| extraction|s3 | 0.1221 | 0.1709 | length+letterset | 16/16 | 0.5035 |
| first_letter|s0 | 0.1155 | 0.4573 | length+letterset | 16/16 | 0.5427 |
| first_letter|s3 | 0.1235 | 0.475 | length+letterset | 16/16 | 0.525 |
| last_letter|s0 | 0.1706 | 0.5466 | letterset | 16/16 | 0.4394 |
| last_letter|s3 | 0.1455 | 0.5516 | length+letterset | 16/16 | 0.4471 |
| letter_before|s0 | 0.1116 | 0.4061 | letterset+q | 3/16 | 0.1993 |
| letter_before|s3 | 0.1278 | 0.4165 | length+letterset | 8/16 | 0.1911 |
| most_frequent_letter|s0 | 0.2898 | 0.4423 | length+letterset | 16/16 | 0.4473 |
| most_frequent_letter|s3 | 0.2564 | 0.4002 | length+letterset | 16/16 | 0.4828 |
| word_length|s0 | 0.1938 | 0.3626 | letterset | 12/16 | 0.5249 |
| word_length|s3 | 0.185 | 0.3513 | letterset | 16/16 | 0.635 |


## ACC — answer-token subspace vs rank-matched comparators (v3 section 8)

| shots | cells | proj − random (wins) | proj − PCA (wins) | overlap (median) | r(gap, deficit) | r, unsaturated | deficit by gap tertile | proj − random by gap tertile |
|---|---|---|---|---|---|---|---|---|
| s0 | 176 | +0.100 (159) | +0.012 (97) | 0.0278 | 0.4819 | 0.8008 | [0.0484, 0.0891, 0.1509] | [0.1019, 0.0698, 0.1288] |
| s3 | 176 | +0.119 (159) | +0.023 (113) | 0.0615 | 0.6435 | 0.5991 | [0.015, 0.0532, 0.1006] | [0.1468, 0.1056, 0.1059] |


## X — cross-task decomposition (shots=0, zero-leak cells, share of total)

| probed label | total Δ | generic frame | wrong-property Q | specific Q | cells | models |
|---|---|---|---|---|---|---|
| count_distinct | 0.0884 | 0.2753 | 0.5246 | 0.2001 | 21 | 7 |
| first_letter | 0.129 | 0.4633 | 0.3044 | 0.2323 | 22 | 11 |
| last_letter | 0.2426 | 0.2262 | 0.0782 | 0.6956 | 32 | 16 |
| most_frequent_letter | 0.2287 | 0.2077 | 0.389 | 0.4032 | 28 | 14 |
| word_length | 0.3476 | 0.2155 | 0.2619 | 0.5225 | 24 | 8 |


## X_term — cross-task decomposition (shots=0, zero-leak cells, share of total)

| probed label | total Δ | generic frame | wrong-property Q | specific Q | cells | models |
|---|---|---|---|---|---|---|
| extraction | 0.2173 | 0.3369 | -0.0528 | 0.7159 | 32 | 16 |
| letter_before | 0.1368 | 0.282 | 0.1255 | 0.5925 | 28 | 14 |


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
