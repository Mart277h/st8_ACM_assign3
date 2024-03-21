
// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  vector[N] y;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real mu;
  real<lower=0> sigma;
}

transformed parameters {
  real logit_transformed_source1;
  real logit_transformed_source1;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  target += ; //priors 
  
  target += normal_lpmf(y | bias + to_vector(Source1) + to_vector(Source2)); //beta distribution if transforming, normal distribution if you do logit of transformed parameters
}

