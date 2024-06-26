// 
functions{
  real normal_lb_rng(real mu, real sigma, real lb) { // normal distribution with a lower bound
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
  int<lower=0> N;
  array[N] real SecondRating; //What we are predicting, our y 
  array[N] real FirstRating; // The participants first rating
  array[N] real GroupRating; // The group rating the participant gets
}

transformed data {
  array[N] real l_FirstRating; // Creating an array for logit transformed data 
  array[N] real l_GroupRating;
  array[N] real l_SecondRating;
  
  for (n in 1:N){
    l_FirstRating[n] = logit(FirstRating[n]/9); // Doing the logit transformation
    l_GroupRating[n] = logit(GroupRating[n]/9);
    l_SecondRating[n] = logit(SecondRating[n]/9); // Consider adding the grouprating = 0 on the logit scale? 
  }
}

parameters {
  real bias; 
  real st_d; 
  real<lower = 0.5, upper = 1> weight1; 
  real<lower = 0.5, upper = 1> weight2;
}

transformed parameters {
  real<lower = 0, upper = 1> t_weight1;
  real<lower = 0, upper = 1> t_weight2;

  t_weight1 = (weight1 - 0.5) * 2;  
  t_weight2 = (weight2 - 0.5) * 2;
}

model {
  target += normal_lpdf(bias | 0, 1); //prior for bias, as this is the only thing we estimate in this model 
  target += normal_lpdf(st_d | 0, .3) - normal_lccdf(0 | 0, .3);
  target += beta_lpdf(t_weight1 | 1, 1); //prior for transformed parameter weights 1 and 2
  target += beta_lpdf(t_weight2 | 1, 1);
  
  for (n in 1:N) {
    target += normal_lpdf(l_SecondRating[n] | bias + l_FirstRating[n]*t_weight1 + l_GroupRating[n]*t_weight2, st_d); 
  }
  
}

generated quantities{
  real bias_prior;
  real inv_bias;
  real w1_prior;
  real w2_prior;
  array[N] real log_lik;
  
  bias_prior = inv_logit(normal_rng(0, 1));
  inv_bias = inv_logit(bias);
  
  w1_prior = 0.5 + inv_logit(normal_rng(0, 1))/2;
  w2_prior = 0.5 + inv_logit(normal_rng(0, 1))/2;
  
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(l_SecondRating[n] | bias + l_FirstRating[n]*t_weight1 + l_GroupRating[n]*t_weight2, st_d);
  }
  
    
  //predicted_choice = round(inv_logit_scaled(log_lik[n])*9)
}

