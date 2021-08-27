get_mixed_p_val_vec <- function(quantile_vec, mu_vec, pi_vec, lambdatilde_p){
	#First compute the composite pvalues
	p <- length(mu_vec)
	
	#For pgamma, we need to specify the shape and scale parameters
	#shape alpha = 1/lambda
	alpha <- 1/lambdatilde_p
	#NB: scale beta = mu/alpha, as per formulation in Lindsay paper
	beta_vec <- mu_vec/alpha
	
	#we could probablu vectorise this, but this is simpler
	#we use the pgamma to compute a vector of pvalues from the vector of quantiles, for a given distribution
	#we then scale this by the appropriate pi_vec value, and add this vector to a 0 vector, and repeat
	#finally, each component of the vector is a pi_vec-scaled sum of pvalues
	partial_pval_vec <- rep(0, length(quantile_vec))
    print(partial_pval_vec)
	for (i in 1:p){		
		partial_pval_vec <- partial_pval_vec + pi_vec[i] * pgamma(quantile_vec, shape=alpha, scale = beta_vec[i])		
        cat(i, ": ")
        print(partial_pval_vec)
	}
	return(partial_pval_vec)
}


temp <- function(){
    pi_vec <- c(0.3699345, 0.4827818,  0.1472837)
    mu_roots <- c(6.103128, 12.037864, 17.860640)
    lambdatilde_p <- 0.5583305
    x = c(0.627, 10.203)
	mixed_p_val_vec <- get_mixed_p_val_vec(x, mu_roots, pi_vec, lambdatilde_p)
    print(mixed_p_val_vec)
}

