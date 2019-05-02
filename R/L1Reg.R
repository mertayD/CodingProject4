LinearModelL1 <- function(X.mat, y.vec, penalty, opt.thresh=0.001, intial.weight.vec, step.size=0.1) {

  y.tilde <- ifelse(y.vec == 1, 1, -1)

  X.sc <- scale(X.mat)
  X.filtered <- X.sc[, attr(X.sc, "scaled:scale") != 0]

  # Sigmoid Function
  sigmoid <- function(z){
    1/(1+exp(-z))
  }

  ## First element is bias/intercept
  if(is.null(intial.weight.vec)) {
    w.vec <- rep(0, l=ncol(X.filtered) + 1)
  }

  X.int <- cbind(1, X.filtered)
  pred.vec <- X.int %*% w.vec
  prob.vec <- sigmoid(-pred.vec * y.tilde)
  grad.vec <- -t(X.int) %*% (y.tilde * prob.vec)
  d.vec <- -grad.vec
  u.vec <- w.vec + step.size * d.vec

  lambda <- penalty

  positive.part <- function(x) {
    ifelse(x > 0, x, 0)
  }

  soft <- function(x, lambda=10) {
    sign(x) * positive.part(abs(x) - lambda)
  }

  w.vec <- c(u.vec[1], soft(u.vec[-1], step.size*lambda))

}

LinearModelL1penalties <- function(X.mat, y.vec, penalty.vec, step.size=0.05) {
  i = 0
  for(penalty in penalty.vec) {
    pred.mat[, i] <- LinearModelL1(X.mat=X.mat, y.vec=y.vec, penalty=penalty, opt.thresh=0.001,intial.weight.vec=NULL, step.size=step.size)
    i = i + 1
  }

  min(pred.mat)
}

LinearModelL1CV <- function(X.mat, y.vec, fold.vec, n.folds=5, penalty.vec, step.size=0.1) {

  set.seed(1)
  n.folds <- 5
  fold.vec <- sample(rep(1:n.folds, l=nrow(X.mat)))
  validation.fold <- 1

  is.train <- fold.vec != validation.fold
  is.validation <- fold.vec == validation.fold

  X.train <- X.mat[is.train ,]
  y.train <- y.vec[is.train]

  if(is.null(fold.vec)) {
    fold.vec <- sample(rep(1:n.folds, l=nrow(X.mat)))
  }

  train.loss.mat <- matrix(0, nrow = n.folds, ncol = max.neighbors)
  validation.loss.mat <- matrix(0, nrow = n.folds, ncol = max.neighbors)

  for(fold.i in 1:n.folds){
    test.i <- which(fold.vec == fold.i, arr.ind=TRUE)
    train.features <- matrix(X.mat[-test.i, ], nrow = nrow(X.mat) - length(test.i), ncol(X.mat))
    nrow(train.features)
    ncol(train.features)
    is.matrix(train.features)
    train.labels <- y.vec[-test.i]
    y.vec[-test.i]
    validation.features <- matrix(X.mat[test.i], nrow = length(test.i), ncol = ncol(X.mat))
    validation.labels <- y.vec[test.i]
    pred.mat <- NN1toMaxPredictMatrix_func(
      train.features, train.labels,
      max.neighbors, validation.features)
    is.matrix(X.mat)
    pred.mat

    set.list <- list(train=train.features, validation=!train.features)

    for(set.name in names(set.list)){
      # is.set <- set.list[[set.name]]
      # print(is.set)
      # set.pred.mat <- pred.mat[is.set,]
      # set.label.vec <- data.test[is.set]

      if(all(y == 1 || y == 0)) {
        loss.mat <- ifelse(pred.mat > 0.5, 1, 0) != validation.labels #zero-one loss for binary classification.
      } else {
        loss.mat <- (pred.mat - validation.labels)^2 #square loss for regression.
      }

      train.loss.mat[fold.i, ] <- colMeans(as.matrix(loss.mat))
    }
  }
  min(train.loss.mat)
}

# data(zip.train, package = "ElemStatLearn")
# all.y.vec <- zip.train[, 1]
# is.01 <- all.y.vec %in% c(0,1)
# y.vec <- all.y.vec[is.01]
# X.mat <- zip.train[is.01, -1]
#LinearModelL1(X.mat=X.mat, y.vec=y.vec, penalty=10, opt.thresh=0.001,intial.weight.vec =  NULL, step.size = 0.1)
#LinearModelL1penalties(X.mat, y.vec, )
