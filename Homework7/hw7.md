Homework 7
================
YutingMei
April 21, 2022

``` r
library('rgl')
library('ElemStatLearn')
library('nnet')
library('dplyr')
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library('keras')
```

``` r
library(reticulate)
# 
# # create a new environment 
# conda_create("r-reticulate")
```

``` r
# unlink("/Users/pritom/Library/r-miniconda", recursive = TRUE)
```

``` r
# remotes::install_github("rstudio/reticulate")
# remotes::install_github("rstudio/tensorflow")
# remotes::install_github("rstudio/keras")
# reticulate::miniconda_uninstall()
# reticulate::install_miniconda()
# keras::install_keras()
```

``` r
keras:::keras_version()
```

    ## Loaded Tensorflow version 2.8.0

    ## [1] '2.8.0'

-   Use the Keras library to re-implement the simple neural network
    discussed during lecture for the mixture data (see nnet.R). Use a
    single 10-node hidden layer; fully connected.

``` r
data("mixture.example")
dat <- mixture.example
```

``` r
plot_mixture_data <- function(dat=mixture.example, showtruth=FALSE) {
  ## create 3D graphic, rotate to view 2D x1/x2 projection
  par3d(FOV=1,userMatrix=diag(4))
  plot3d(dat$xnew[,1], dat$xnew[,2], dat$prob, type="n",
         xlab="x1", ylab="x2", zlab="",
         axes=FALSE, box=TRUE, aspect=1)
  ## plot points and bounding box
  x1r <- range(dat$px1)
  x2r <- range(dat$px2)
  pts <- plot3d(dat$x[,1], dat$x[,2], 1,
                type="p", radius=0.5, add=TRUE,
                col=ifelse(dat$y, "orange", "blue"))
  lns <- lines3d(x1r[c(1,2,2,1,1)], x2r[c(1,1,2,2,1)], 1)
  
  if(showtruth) {
    ## draw Bayes (True) classification boundary
    probm <- matrix(dat$prob, length(dat$px1), length(dat$px2))
    cls <- contourLines(dat$px1, dat$px2, probm, levels=0.5)
    pls <- lapply(cls, function(p) 
      lines3d(p$x, p$y, z=1, col='purple', lwd=3))
    ## plot marginal probability surface and decision plane
    sfc <- surface3d(dat$px1, dat$px2, dat$prob, alpha=1.0,
      color="gray", specular="gray")
    qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
      color="gray", lit=FALSE)
  }
}

## compute and plot predictions
plot_nnet_predictions <- function(fit, dat=mixture.example) {
  
  ## create figure
  plot_mixture_data()

  ## compute predictions from nnet
  preds <- predict(fit, dat$xnew, type="class")
  probs <- predict(fit, dat$xnew, type="raw")[,1]
  probm <- matrix(probs, length(dat$px1), length(dat$px2))
  cls <- contourLines(dat$px1, dat$px2, probm, levels=0.5)

  ## plot classification boundary
  pls <- lapply(cls, function(p) 
    lines3d(p$x, p$y, z=1, col='purple', lwd=2))
  
  ## plot probability surface and decision plane
  sfc <- surface3d(dat$px1, dat$px2, probs, alpha=1.0,
                   color="gray", specular="gray")
  qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
                 color="gray", lit=FALSE)
}

## plot data and 'true' probability surface
plot_mixture_data(showtruth=TRUE)
```

``` r
## plot data and 'true' probability surface
plot_mixture_data(showtruth=TRUE)

## fit single hidden layer, fully connected NN 
## 10 hidden nodes
# fit <- nnet(x=dat$x, y=dat$y, size=10, entropy=TRUE, decay=0) 
model = keras_model_sequential()
model %>%
  layer_flatten() %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
# plot_nnet_predictions(fit)
```

``` r
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
```

``` r
model %>% fit(dat$x, dat$y, epochs = 5, verbose = 2)
```

``` r
predictions <- model %>% predict(dat$xnew, type = "class")
```

``` r
predictions %>% length
```

    ## [1] 13662

-   Create a figure to illustrate that the predictions are (or are not)
    similar using the ‘nnet’ function versus the Keras model.

``` r
fit <- nnet(x=dat$x, y=dat$y, size=10, entropy=TRUE, decay=0) 
```

    ## # weights:  41
    ## initial  value 159.026281 
    ## iter  10 value 99.222814
    ## iter  20 value 89.520567
    ## iter  30 value 76.378936
    ## iter  40 value 70.842254
    ## iter  50 value 66.455862
    ## iter  60 value 62.424360
    ## iter  70 value 61.671760
    ## iter  80 value 61.494303
    ## final  value 61.492770 
    ## converged

``` r
# result using nnet
predict_nn = predict(fit, dat$xnew, type = 'class')
```

``` r
# result of keras
predictions_keras = apply(predictions, 1, function(x) ifelse(x[2] > x[1], 1, 0))
```

``` r
plot_mix_data <- expression({
  plot(dat$xnew[,1], dat$xnew[,2],
       col=ifelse(predictions_keras==0, 'blue', 'orange'),
       pch=20,
       xlab=expression(x[1]),
       ylab=expression(x[2]),
       main = 'predictions from Keras')
})

eval(plot_mix_data)
```

![](hw7_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
plot_mix_data <- expression({
  plot(dat$xnew[,1], dat$xnew[,2],
       col=ifelse(predict_nn=='0', 'blue', 'orange'),
       pch=20,
       xlab=expression(x[1]),
       ylab=expression(x[2]),
       main = 'predictions from nnet')
})

eval(plot_mix_data)
```

![](hw7_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

-   (optional extra credit) Convert the neural network described in the
    “Image Classification” tutorial to a network that is similar to one
    of the convolutional networks described during lecture on 4/15
    (i.e., Net-3, Net-4, or Net-5) and also described in the ESL book
    section 11.7. See the !ConvNet tutorial on the RStudio Keras
    website.

``` r
# convert to net-3 with local connectivity
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_locally_connected_2d(filters = 64, kernel_size = c(3,3), activation = "relu")
```

``` r
summary(model)
```

    ## Model: "sequential_1"
    ## ________________________________________________________________________________
    ##  Layer (type)                       Output Shape                    Param #     
    ## ================================================================================
    ##  conv2d_2 (Conv2D)                  (None, 30, 30, 32)              896         
    ##  max_pooling2d_1 (MaxPooling2D)     (None, 15, 15, 32)              0           
    ##  conv2d_1 (Conv2D)                  (None, 13, 13, 64)              18496       
    ##  max_pooling2d (MaxPooling2D)       (None, 6, 6, 64)                0           
    ##  conv2d (Conv2D)                    (None, 4, 4, 64)                36928       
    ##  locally_connected2d (LocallyConnec  (None, 2, 2, 64)               147712      
    ##  ted2D)                                                                         
    ## ================================================================================
    ## Total params: 204,032
    ## Trainable params: 204,032
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

``` r
model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")
```

``` r
summary(model)
```

    ## Model: "sequential_1"
    ## ________________________________________________________________________________
    ##  Layer (type)                       Output Shape                    Param #     
    ## ================================================================================
    ##  conv2d_2 (Conv2D)                  (None, 30, 30, 32)              896         
    ##  max_pooling2d_1 (MaxPooling2D)     (None, 15, 15, 32)              0           
    ##  conv2d_1 (Conv2D)                  (None, 13, 13, 64)              18496       
    ##  max_pooling2d (MaxPooling2D)       (None, 6, 6, 64)                0           
    ##  conv2d (Conv2D)                    (None, 4, 4, 64)                36928       
    ##  locally_connected2d (LocallyConnec  (None, 2, 2, 64)               147712      
    ##  ted2D)                                                                         
    ##  flatten_1 (Flatten)                (None, 256)                     0           
    ##  dense_3 (Dense)                    (None, 64)                      16448       
    ##  dense_2 (Dense)                    (None, 10)                      650         
    ## ================================================================================
    ## Total params: 221,130
    ## Trainable params: 221,130
    ## Non-trainable params: 0
    ## ________________________________________________________________________________
