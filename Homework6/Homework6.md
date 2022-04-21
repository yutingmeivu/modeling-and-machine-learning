Homework 6
================
YutingMei
April 21, 2022

``` r
library('randomForest')  ## fit random forest
```

    ## randomForest 4.7-1

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library('dplyr')    ## data manipulation
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     combine

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library('caret') ## 'createFolds'
```

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

    ## Loading required package: lattice

-   Using the “vowel.train” data, and the “randomForest” function in the
    R package “randomForest”. Develop a random forest classifier for the
    vowel data by doing the following:

-   Convert the response variable in the “vowel.train” data frame to a
    factor variable prior to training, so that “randomForest” does
    classification rather than regression.

``` r
# get vowel.train
vowel_train <- 
  read.table(url(
    'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train'), header=TRUE, sep = ',')
```

``` r
vowel_train = vowel_train[,-1]
# vowel_train
```

``` r
vowel_train$y = factor(vowel_train$y)
```

-   Review the documentation for the “randomForest” function.

``` r
?randomForest()
```

-   Fit the random forest model to the vowel data using all of the 11
    features using the default values of the tuning parameters.

``` r
fit <- randomForest(y ~ ., data=vowel_train, 
                    proximity=TRUE)
```

``` r
print(fit)
```

    ## 
    ## Call:
    ##  randomForest(formula = y ~ ., data = vowel_train, proximity = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 2.84%
    ## Confusion matrix:
    ##     1  2  3  4  5  6  7  8  9 10 11 class.error
    ## 1  48  0  0  0  0  0  0  0  0  0  0  0.00000000
    ## 2   1 47  0  0  0  0  0  0  0  0  0  0.02083333
    ## 3   0  0 48  0  0  0  0  0  0  0  0  0.00000000
    ## 4   0  0  0 47  0  1  0  0  0  0  0  0.02083333
    ## 5   0  0  0  0 46  1  0  0  0  0  1  0.04166667
    ## 6   0  0  0  0  0 44  0  0  0  0  4  0.08333333
    ## 7   0  0  0  0  1  0 46  1  0  0  0  0.04166667
    ## 8   0  0  0  0  0  0  0 48  0  0  0  0.00000000
    ## 9   0  0  0  0  0  0  1  1 45  1  0  0.06250000
    ## 10  0  0  0  0  0  0  1  0  0 47  0  0.02083333
    ## 11  0  0  0  0  0  1  0  0  0  0 47  0.02083333

-   Use 5-fold CV and tune the model by performing a grid search for the
    following tuning parameters: 1) the number of variables randomly
    sampled as candidates at each split; consider values 3, 4, and 5,
    and 2) the minimum size of terminal nodes; consider a sequence (1,
    5, 10, 20, 40, and 80).

``` r
set.seed(1985)
vow_flds  <- createFolds(vowel_train$y, k=5)
print(vow_flds)
```

    ## $Fold1
    ##   [1]   2   8  13  18  20  24  27  28  32  38  39  51  52  56  69  84  87  88
    ##  [19]  96  97 100 102 108 110 115 118 119 134 143 146 150 151 156 164 182 190
    ##  [37] 191 193 199 202 204 209 231 236 240 243 247 264 268 270 275 279 281 282
    ##  [55] 287 289 299 307 309 311 314 325 345 346 348 349 351 354 356 361 362 370
    ##  [73] 379 381 382 395 399 406 408 415 416 417 418 422 435 436 438 452 453 454
    ##  [91] 455 460 463 466 473 478 481 487 489 492 497 504 505 513 517
    ## 
    ## $Fold2
    ##   [1]   3   4   7   9  19  23  43  46  49  60  62  63  89  90  92  93 103 105
    ##  [19] 120 122 125 126 130 132 135 137 142 148 149 154 155 158 161 167 170 174
    ##  [37] 177 205 206 207 213 215 218 219 220 222 225 229 230 233 239 242 249 256
    ##  [55] 258 259 265 272 274 277 278 283 285 290 293 294 300 312 316 327 328 331
    ##  [73] 332 334 352 353 358 359 363 366 371 373 377 380 394 405 413 442 444 451
    ##  [91] 458 461 464 469 479 482 484 496 498 507 509 516 523 528
    ## 
    ## $Fold3
    ##   [1]  11  12  16  17  21  37  40  42  48  50  54  61  73  74  76  78  79  82
    ##  [19]  85  94 109 112 121 129 136 138 141 144 152 159 165 166 168 172 173 178
    ##  [37] 179 183 185 192 195 197 201 212 228 234 237 253 255 266 267 280 286 288
    ##  [55] 296 301 303 313 315 317 320 323 326 329 330 333 335 339 342 344 350 364
    ##  [73] 365 368 374 378 386 391 393 398 411 412 414 419 420 431 432 441 443 445
    ##  [91] 448 449 462 471 477 480 483 488 490 491 495 514 524 525 526 527
    ## 
    ## $Fold4
    ##   [1]   1   5   6  15  26  29  33  34  35  36  45  47  53  55  65  66  68  71
    ##  [19]  80  91 111 113 114 123 124 128 131 139 140 145 153 157 163 169 171 186
    ##  [37] 187 198 200 203 216 217 221 223 224 226 235 246 250 251 254 260 261 262
    ##  [55] 263 284 291 297 304 305 308 310 318 319 321 337 338 340 341 347 357 360
    ##  [73] 372 383 392 400 401 402 404 409 410 424 426 427 428 430 434 440 446 447
    ##  [91] 450 456 457 467 468 472 474 476 485 499 502 508 510 512 515 519 522
    ## 
    ## $Fold5
    ##   [1]  10  14  22  25  30  31  41  44  57  58  59  64  67  70  72  75  77  81
    ##  [19]  83  86  95  98  99 101 104 106 107 116 117 127 133 147 160 162 175 176
    ##  [37] 180 181 184 188 189 194 196 208 210 211 214 227 232 238 241 244 245 248
    ##  [55] 252 257 269 271 273 276 292 295 298 302 306 322 324 336 343 355 367 369
    ##  [73] 375 376 384 385 387 388 389 390 396 397 403 407 421 423 425 429 433 437
    ##  [91] 439 459 465 470 475 486 493 494 500 501 503 506 511 518 520 521

``` r
cvrandf <- function(nodesize, mtry) {
  cverr <- rep(NA, length(vow_flds))
  for(tst_idx in 1:length(vow_flds)) { ## for each fold
    
    ## get training and testing data
    vow_trn <- vowel_train[-vow_flds[[tst_idx]],]
    vow_tst <- vowel_train[ vow_flds[[tst_idx]],]

    
    ## fit rf model to training data
    radf_fit = randomForest(y ~ ., data=vow_trn, 
                    proximity=TRUE, nodesize = nodesize, mtry = mtry)
    
    ## compute test error on testing data
    pre_tst <- predict(radf_fit, vow_tst)
    cverr[tst_idx] <- table(mapply(`%in%`, pre_tst, vow_tst)) / length(pre_tst)
  }
  return(cverr)
}
```

``` r
nodel = c(1, 5, 10, 20, 40, 80)
mtryl = c(3, 4, 5)
df_all = NULL
nl = c()
ml = c()
for (i in nodel)
  for (j in mtryl){
    df_all$cv_error = cvrandf(i, j)
    names(df_all) = paste0('nodel_', i, 'mtry_', j)
    nl = append(nl, i)
    ml = append(ml, j)
  }
```

``` r
df_all = df_all %>% data.frame()
cnames = paste0('nodel_', nl, '_mtry_', ml)
colnames(df_all) = cnames
df_all
```

    ##   nodel_1_mtry_3 nodel_1_mtry_4 nodel_1_mtry_5 nodel_5_mtry_3 nodel_5_mtry_4
    ## 1      0.9047619      0.9047619      0.9047619      0.9047619      0.9047619
    ## 2      0.9038462      0.9038462      0.9038462      0.9038462      0.9038462
    ## 3      0.9056604      0.9056604      0.9056604      0.9056604      0.9056604
    ## 4      0.9065421      0.9065421      0.9065421      0.9065421      0.9065421
    ## 5      0.9056604      0.9056604      0.9056604      0.9056604      0.9056604
    ##   nodel_5_mtry_5 nodel_10_mtry_3 nodel_10_mtry_4 nodel_10_mtry_5
    ## 1      0.9047619       0.9047619       0.9047619       0.9047619
    ## 2      0.9038462       0.9038462       0.9038462       0.9038462
    ## 3      0.9056604       0.9056604       0.9056604       0.9056604
    ## 4      0.9065421       0.9065421       0.9065421       0.9065421
    ## 5      0.9056604       0.9056604       0.9056604       0.9056604
    ##   nodel_20_mtry_3 nodel_20_mtry_4 nodel_20_mtry_5 nodel_40_mtry_3
    ## 1       0.9047619       0.9047619       0.9047619       0.9047619
    ## 2       0.9038462       0.9038462       0.9038462       0.9038462
    ## 3       0.9056604       0.9056604       0.9056604       0.9056604
    ## 4       0.9065421       0.9065421       0.9065421       0.9065421
    ## 5       0.9056604       0.9056604       0.9056604       0.9056604
    ##   nodel_40_mtry_4 nodel_40_mtry_5 nodel_80_mtry_3 nodel_80_mtry_4
    ## 1       0.9047619       0.9047619       0.9047619       0.9047619
    ## 2       0.9038462       0.9038462       0.9038462       0.9038462
    ## 3       0.9056604       0.9056604       0.9056604       0.9056604
    ## 4       0.9065421       0.9065421       0.9065421       0.9065421
    ## 5       0.9056604       0.9056604       0.9056604       0.9056604
    ##   nodel_80_mtry_5
    ## 1       0.9047619
    ## 2       0.9038462
    ## 3       0.9056604
    ## 4       0.9065421
    ## 5       0.9056604

``` r
sapply(df_all, mean)
```

    ##  nodel_1_mtry_3  nodel_1_mtry_4  nodel_1_mtry_5  nodel_5_mtry_3  nodel_5_mtry_4 
    ##       0.9052942       0.9052942       0.9052942       0.9052942       0.9052942 
    ##  nodel_5_mtry_5 nodel_10_mtry_3 nodel_10_mtry_4 nodel_10_mtry_5 nodel_20_mtry_3 
    ##       0.9052942       0.9052942       0.9052942       0.9052942       0.9052942 
    ## nodel_20_mtry_4 nodel_20_mtry_5 nodel_40_mtry_3 nodel_40_mtry_4 nodel_40_mtry_5 
    ##       0.9052942       0.9052942       0.9052942       0.9052942       0.9052942 
    ## nodel_80_mtry_3 nodel_80_mtry_4 nodel_80_mtry_5 
    ##       0.9052942       0.9052942       0.9052942

``` r
sapply(df_all, sd)
```

    ##  nodel_1_mtry_3  nodel_1_mtry_4  nodel_1_mtry_5  nodel_5_mtry_3  nodel_5_mtry_4 
    ##     0.001025365     0.001025365     0.001025365     0.001025365     0.001025365 
    ##  nodel_5_mtry_5 nodel_10_mtry_3 nodel_10_mtry_4 nodel_10_mtry_5 nodel_20_mtry_3 
    ##     0.001025365     0.001025365     0.001025365     0.001025365     0.001025365 
    ## nodel_20_mtry_4 nodel_20_mtry_5 nodel_40_mtry_3 nodel_40_mtry_4 nodel_40_mtry_5 
    ##     0.001025365     0.001025365     0.001025365     0.001025365     0.001025365 
    ## nodel_80_mtry_3 nodel_80_mtry_4 nodel_80_mtry_5 
    ##     0.001025365     0.001025365     0.001025365

-   With the tuned model, make predictions using the majority vote
    method, and compute the misclassification rate using the
    ‘vowel.test’ data.

``` r
vowel_test <- 
  read.table(url(
    'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test'), header=TRUE, sep = ',')
```

``` r
which.min(sapply(df_all, mean))
```

    ## nodel_1_mtry_3 
    ##              1

``` r
# FALSE is the rate of model make classification correctly
# the misclassification rate is 0.1688312
ft_tuned = randomForest(y ~ ., data=vowel_train, 
                    proximity=TRUE, nodesize = 1, mtry = 3)
pre_tuned <- predict(ft_tuned, vowel_test)
mis_error <- table(mapply(`%in%`, pre_tuned, vowel_test)) / length(pre_tuned)
mis_error
```

    ## 
    ##     FALSE      TRUE 
    ## 0.8311688 0.1688312
