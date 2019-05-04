#' Title
#'
#' @param X.mat 
#' @param y.vec 
#' @param fold.vec 
#' @param n.folds 
#' @param penalty.vec 
#' @param step.size 
#'
#' @return
#' @export
#'
#' @examples
#' penalty.vec<- c(0.02,0.03,0.04,0.05,0.1)
#' data(SAheart, package = "ElemStatLearn") 
#' y.vec <- as.vector(SAheart[, 10])
#' X.mat <- as.matrix(SAheart[,c(-10,-5)])
#' X.sc <- scale(X.mat)
#' fold.vec <- sample(rep(1:5, l=nrow(X.mat)))
#' n.folds <- 5
#' LinearModelL1CV(X.mat, y.vec,fold.vec,n.folds,penalty.vec,0.5)
LinearModelL1CV <- function(
  X.mat,
  y.vec,
  #(the following arguments should have sensible defaults)
  fold.vec=sample(rep(1:4, l=nrow(X.mat))), 
  n.folds=4,
  penalty.vec=c(0.05,0.1,0.2,0.3,0.4),
  step.size=0.5) {
  if(!is.matrix(X.mat))
  {
    stop("Feature matrix is not a matrix")
  }
  
  if(nrow(X.mat) <= 0 | ncol(X.mat) <= 0)
  {
    stop("Feature matrix has unexpected dimensions")
  }
  
  if(length(y.vec) <= 0)
  {
    stop("Output matrix has unexpected dimensions")
  }
  
  if(is.null(fold.vec))
  {
    fold.vec <- sample(rep(1:5, l=nrow(X.mat)))
    n.folds <- 5
  }
  
  is.binary <- all(y.vec %in% c(0, 1))
  
  if(is.binary) {
    y.tilde <- ifelse(y.vec == 1, 1, -1)
  }
  
  #logistic loss 
  log.loss <- function(pred,y.tilde.vec){
    log(1+exp(-y.tilde.vec * pred))
  }
  
  X.mat <- cbind(1,X.mat)
  train.loss.mat <- matrix(, n.folds, length(penalty.vec))
  validation.loss.mat <- matrix(, n.folds, length(penalty.vec))
  
  n.folds <- max(fold.vec)
  for(fold.i in 1:n.folds)
  {
    fold_data <- which(fold.vec %in% c(fold.i))
    
    X.train <- X.mat[fold_data ,]
    X.valid <- X.mat[-fold_data ,]
    
    y.tilde.train <- y.tilde[fold_data]
    y.tilde.validation <- y.tilde[fold_data]
    
    Y.train <- y.vec[fold_data]
    Y.valid <- y.vec[-fold_data]
    
    # n_rows_validation_set <- nrow(validation_set)
    # n_rows_train_set <- nrow(train_set)
    
    for(prediction.set.name in c("train", "validation")){
      if(identical(prediction.set.name, "train")){
        W <- LinearModelL1penalties(X.train, Y.train,penalty.vec,step.size)
        pred.mat <- X.train %*% W
        if(is.binary)
        {
          loss.mat <- log.loss(pred.mat,y.tilde.train)
          train.loss.mat[fold.i, ] <- colMeans(loss.mat)
        }
        else
        {
          train.loss.mat[fold.i, ] = colMeans((t(pred.mat) - Y.train)^2)
        }
      }
      else{
        W <- LinearModelL1penalties(X.train, Y.train,penalty.vec,step.size)
        pred.mat <- X.train %*% W
        if(is.binary)
        {
          loss.mat <- log.loss(pred.mat,y.tilde.validation)
          validation.loss.mat[fold.i,] = colMeans(loss.mat)
        }
        else{
          validation.loss.mat[fold.i,] = colMeans((t(pred.mat) - Y.valid)^2)
        }
      }
    }
  }
  print(validation.loss.mat)
  mean.validation.loss.vec <- colMeans(validation.loss.mat)
  mean.train.loss.vec <- colMeans(train.loss.mat)
  selected.steps = which.min(mean.validation.loss.vec)
  best_model <- LinearModelL1penalties(X.mat, y.vec, penalty.vec = c(selected.steps),step.size)
  weight_vec <- best_model
  
  list(
    mean.validation.loss = mean.validation.loss.vec,
    mean.train.loss.vec =  mean.train.loss.vec,
    selected.steps = selected.steps,
    w.vec=weight_vec
   )
  
}
  