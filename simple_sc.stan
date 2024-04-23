// 
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

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real bias; 
  real st_d; 
}

model {
  target += normal_lpdf(bias | 0, 1); //prior for bias, as this is the only thing we estimate in this model 
  target += normal_lpdf(st_d | 1, 0.1);
  target += normal_lpdf(to_vector(l_SecondRating) | bias + to_vector(l_FirstRating)*0.5 + to_vector(l_GroupRating)*0.5, st_d); 
}

generated quantities{
  real bias_prior;
  real sd_prior;
  array[N] real log_lik;
  
  bias_prior = normal_rng(0, 1);
  sd_prior = normal_rng(1, 0.1);
  
  for (n in 1:N){  
    log_lik[n] = normal_lpdf(l_SecondRating[n] | bias_prior + l_FirstRating[n]*0.5 + l_GroupRating[n]*0.5, sd_prior);
  }
  
}

