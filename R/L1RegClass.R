LinearModelL1 <- function(X.scaled.mat, y.vec, penalty=10, opt.thresh=0.001, initial.weight.vec=NULL, step.size=0.4) {

  # X.filtered <- X.sc[, attr(X.sc, "scaled:scale") != 0]
  # X.int <- cbind(1, X.filtered)
  X.int <- X.scaled.mat
  y.tilde <- ifelse(y.vec == 1, 1, -1)


  print(penalty)
  ## First element is bias/intercept
  if(is.null(initial.weight.vec)) {
    w.vec <- rep(0, l=ncol(X.int))
  }

  else{
    w.vec <- initial.weight.vec
  }
  # Sigmoid Function
  sigmoid <- function(z){
    1/(1+exp(-z))
  }

  # Postive Part Function
  positive.part <- function(x) {
    ifelse(x < 0, 0, x)
  }

  # Soft function for optimality
  soft <- function(x, lambda) {
    sign(x) * positive.part(abs(x) - lambda)
  }

  # Optimal criterion
  l1.opt <- function(w.vec, d) {
    ifelse(
      w.vec==0,
      positive.part(abs(d) - lambda),
      abs(d - sign(w.vec)*lambda)
    )
  }

  curve(soft(x,1), -5, 5)

  lambda <- penalty
  it <- 1

  pred.vec <- X.int %*% w.vec
  prob.vec <- sigmoid(-pred.vec * y.tilde)
  grad.vec <- -t(X.int) %*% (y.tilde * prob.vec) / nrow(X.int)
  d.vec <- -grad.vec

  crit.vec <- c(
    abs(grad.vec[1]),
    l1.opt(w.vec[-1], d.vec[-1])
  )

  cost.weight <- function(w.vec) {
    pred.vec <- X.int %*% w.vec
    loss.vec <- log(1 + exp(-pred.vec * y.tilde))
    mean(loss.vec) + lambda * sum(abs(w.vec[-1]))
  }

  # Cost of a given step size
  cost.step <- function(step) {
    new.w.vec <- w.step(step)
    cost.weight(new.w.vec)
  }

  w.step <- function(step) {
    u.vec <- w.vec + step * d.vec
    c(u.vec[1], soft(u.vec[-1], step*lambda))
  }

  # Max value of lambda e.g. the optimal solution should be less than this
  curve(sapply(x, cost.step), 0, 10)

  is.optimal = FALSE

  while(is.optimal == FALSE) {

    pred.vec <- X.int %*% w.vec
    prob.vec <- sigmoid(-pred.vec * y.tilde)
    grad.vec <- -t(X.int) %*% (y.tilde * prob.vec) / nrow(X.int)
    d.vec <- -grad.vec

    crit.vec <- c(
      abs(grad.vec[1]),
      l1.opt(w.vec[-1], d.vec[-1])
    )

    lambda.max <- max(abs(grad.vec[-1]))
    cat(sprintf("it=%d crit=%f lambda.max=%f\n", it, max(crit.vec), lambda.max))

    it <- it + 1

    # Runs line search
    while(cost.step(step.size/2) < cost.step(step.size)) {
      step.size <- step.size/2
    }

    while(cost.step(step.size*2) < cost.step(step.size)) {
      step.size <- step.size*2
    }

    points(step.size, cost.step(step.size))
    w.vec <- w.step(step.size)

    # Check if sub-optimal
    b = w.vec[1]

    if(b < opt.thresh) {
      w.vec[-1] = 0
    }

    l1.opt.vec <- l1.opt(w.vec[-1], d.vec)
    optimal.vec <- l1.opt.vec < opt.thresh
    is.optimal = all(optimal.vec)
  }

  return(w.vec)
}

LinearModelL1penalties <- function(X.mat, y.vec, penalty.vec=c(10), step.size=0.4) {
  X.sc <- scale(X.mat)
  y.tilde <- ifelse(y.vec == 1, 1, -1)
  X.filtered <- X.sc[, attr(X.sc, "scaled:scale") != 0]
  X.int <- cbind(1, X.filtered)

  # Init matrix
  W.mat <- matrix(0, nrow = length(penalty.vec), ncol = ncol(X.int) + 1)

  new.w.vec=NULL

  for(lambda in 1:length(penalty.vec)) {
    new.w.vec = LinearModelL1(X.int, y.tilde, penalty=penalty.vec[lambda], initial.weight.vec=new.w.vec, step.size=step.size)
    W.mat[, lambda] <- new.w.vec
    new.intial.weight <- new.w.vec
  }

  w.unsc <- W.mat/attr(X.sc, "scaled:scale")
  # b.orig <- -t(W.mat[, 1]/attr(X.sc, "scaled:scale")) %*% attr(X.sc, "scaled:center")
  # w.with.intercept <- rbind(intercept=as.numeric(b.orig), w.unsc)
  return(w.unsc)
}

LinearModelL1CV <- function(
  X.mat,
  y.vec,
  fold.vec=sample(rep(1:5, l=nrow(X.mat))),
  n.folds=5,
  penalty.vec=c(5,4,3,2,1),
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
  }

  is.binary <- all(y.vec %in% c(0, 1))

  if(is.binary) {
    y.tilde <- ifelse(y.vec == 1, 1, -1)
  }

  train.loss.mat <- matrix(, ncol(X.mat), n.folds)
  validation.loss.mat <- matrix(, ncol(X.mat), n.folds)

  # n.folds <- max(fold.vec)
  for(fold.i in 1:n.folds)
  {
    fold_data <- which(fold.vec %in% c(fold.i))

    X.train <- X.mat[-fold_data ,]
    X.valid <- X.mat[fold_data ,]

    Y.train <- y.vec[-fold_data]
    Y.valid <- y.vec[fold_data]

    # n_rows_validation_set <- nrow(validation_set)
    # n_rows_train_set <- nrow(train_set)

    for(prediction.set.name in c("train", "validation")){
      if(identical(prediction.set.name, "train")){

        W <- LinearModelL1penalties(X.train, Y.train)
        print(ncol(W))
        print(ncol(X.train))
        pred.mat <- X.train %*% t(W)


        if(is.binary)
        {
          loss.mat <- ifelse(pred.mat > 0.5 , 1, 0) != Y.train
          print(nrow(loss.mat))
          print(ncol(loss.mat))
          print("COLUMN")
          print(ncol(X.mat))
          train.loss.mat[, fold.i] <- colMeans(loss.mat)
        }
        else
        {
          train.loss.mat[, fold.i] = colMeans((pred.mat - Y.train)^2)
        }
      }
      else{
        W <- LinearModelL1penalties(X.valid, Y.valid)
        pred.mat <- X.valid %*% t(W)
        if(is.binary)
        {
          loss.mat <-  ifelse(pred.mat > 0.5 , 1, 0) != Y.valid
          validation.loss.mat[,fold.i] = colMeans(loss.mat)
        }
        validation.loss.mat[,fold.i] = colMeans((pred.mat - Y.valid)^2)
      }
    }
  }
  mean.validation.loss.vec <- rowMeans(validation.loss.mat)
  mean.train.loss.vec <- rowMeans(train.loss.mat)
  selected.steps = which.min(mean.validation.loss.vec)
  best_model <- LinearModelL1penalties(X.train, y.train, penalty.vec = c(selected.steps))
  weight_vec <- best_model[, selected.steps]

  list(
    mean.validation.loss = mean.validation.loss.vec,
    mean.train.loss.vec =  mean.train.loss.vec,
    selected.steps = selected.steps,
    pred.mat=best_model$pred.mat,
    V.mat= best_model$V.mat,
    w.vec=weight_vec,
    predict=function(testX.mat) {
      str(cbind(1, testX.mat))
      A.mat <- testX.mat %*% best_model$V.mat
      Z.mat <- sigmoid(A.mat)
      pred.vec <- Z.mat %*% weight_vec
      return(pred.vec)
    })
}

data(zip.train, package = "ElemStatLearn")
table(zip.train[, 1])
table(zip.train[, 2])

all.y.vec <- zip.train[, 1]
is.01 <- all.y.vec %in% c(0,1)

y.vec <- all.y.vec[is.01]
y.vec <- y.vec[0:50]
X.mat <- zip.train[is.01, -1]
X.mat <- X.mat[0:50, ]
X.sc <- scale(X.mat)

LinearModelL1CV(X.mat, y.vec)
