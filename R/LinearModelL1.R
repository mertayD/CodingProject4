#' Title
#'
#' @param X.scaled.mat 
#' @param y.vec 
#' @param penalty 
#' @param opt.thresh 
#' @param initial.weight.vec 
#' @param step.size 
#'
#' @return
#' @export
#'
#' @examples
#' data(SAheart, package = "ElemStatLearn") 
#' y.vec <- SAheart[, 10]
#' X.mat <- SAheart[,c(-10,-5)]
#' X.sc <- scale(X.mat)
#' initial.weight.vec <- w.vec <- rep(0, l=ncol(X.sc))
#' LinearModelL1(X.sc, y.vec, 0.1,0.01,initial.weight.vec,0.4)
LinearModelL1 <- function(
  X.scaled.mat, 
  y.vec, 
  penalty = 1, #non-negative numeric scalar 
  opt.thresh = 0.01, #positive numeric scalar 
  initial.weight.vec, 
  step.size = 0.1
)
{
  #sigmoid
  sigmoid <- function(z){
    1/(1+exp(-z))
  }
  #positive part
  ppart <- function(x){
    ifelse(x<0, 0, x)
  }
  #soft
  soft <- function(x, l){
    sign(x) * ppart(abs(x)-l)
  }
  #proximality 
  prox <- function(x, l){
    c(x[1], soft(x[-1], l))
  }
  #w step
  wstep <- function(size){
    prox(w.vec+size*d.vec, penalty*size)
  }
  #logistic loss 
  log.loss <- function(pred){
    log(1+exp(-y.tilde * pred))
  }
  #cost of step
  cstep <- function(size){
    mean(log.loss(X.int %*% wstep(size)))
  }
  #criterion
  subdiff.crit <- function(w,d){
    ifelse(
      w==0,
      ppart(abs(d)-penalty),
      abs(d-sign(w)*penalty))
  }
  is.optimal <- FALSE
  X.int <- cbind(1, X.scaled.mat)
  w.vec <- rep(runif(ncol(X.int)), l=ncol(X.int))
  y.tilde <- ifelse(y.vec==1, 1, -1)
  it<- 0 
  while(is.optimal == FALSE )
  {
    pred.vec <- X.int %*% w.vec
    prob.vec <- sigmoid(-pred.vec * y.tilde)
    grad.vec <- -t(X.int) %*% (y.tilde * prob.vec) / nrow(X.int)
    d.vec <- -grad.vec
    optimality <- c(abs(d.vec[1]), subdiff.crit(w.vec[-1], d.vec[-1]))
    cat(sprintf("crit=%f cost=%f step=%f\n bias=%f\n", head(optimality[2]), cstep(step.size), step.size, w.vec[1]))
    is.optimal <- max(optimality) < opt.thresh
    is.optimal = all(is.optimal)
    w.vec <- wstep(step.size)
    #print(is.optimal)
    print(it)
    it <- it + 1
  }
  return(w.vec)
}








