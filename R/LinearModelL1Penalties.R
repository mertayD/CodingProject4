#' Title
#'
#' @param X.mat 
#' @param y.vec 
#' @param penalty.vec 
#' @param step.size 
#'
#' @return
#' @export
#'
#' @examples
#' penalty.vec<- c(0,02,0.03,0.04,0.05,0.1)
#' data(SAheart, package = "ElemStatLearn") 
#' y.vec <- SAheart[, 10]
#' X.mat <- SAheart[,c(-10,-5)]
#' X.sc <- scale(X.mat)
#' LinearModelL1penalties(X.mat, y.vec, penalty.vec,0.5)
#' 
LinearModelL1penalties <- function(
  X.mat, 
  y.vec, 
  penalty.vec =  c(0.05,0.1,0.2,0.3,0.4), #vector of decreasing penalty values
  step.size
)
{
  X.sc <- scale(X.mat)
  y.tilde <- ifelse(y.vec == 1, 1, -1)
  X.filtered <- X.sc[, attr(X.sc, "scaled:scale") != 0]
  X.int <- X.filtered
  
  W.mat <- matrix(0, nrow = length(penalty.vec), ncol = ncol(X.int) + 1)
  new.w.vec=NULL
  for(lambda in 1:length(penalty.vec)) {
    new.w.vec <- LinearModelL1(X.int, y.vec, penalty=penalty.vec[lambda], opt.thresh = 0.01,initial.weight.vec=new.w.vec, step.size=step.size)
    W.mat[lambda, ] <- new.w.vec
  }
  return(W.mat)
}