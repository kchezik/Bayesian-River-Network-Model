data{
   // Data for fixed effects
   int<lower=1> N;   // Number of observations
   int<lower=1> K;   // Number of predictors
   matrix[N,K] X;    // Predictors matrix
   vector[N] y;      // Response vector
   
   // Data for group-level effects of ID 1
   int<lower=1> n_r1;    // Number of group levels
   int<lower=1> x_r1[N]; // An array of group level effect values
   matrix[N,N] re1;      // Group effect covariance design matrix
   
   // Data for group-level effects of ID 2
   int<lower=1> n_r2;    // Number of group levels
   int<lower=1> x_r2[N]; // An array of group level effect values
   matrix[N,N] re2;      // Group effect covariance design matrix
   
   // Data for river network effects
   matrix[N,N] dist; // River network distance matrix between observations
   
   // Prediction Data
   int<lower=1> Np;        // Number of prediction points
   matrix[Np,K-1] X_p;     // Predictor matrix
   matrix[Np,Np] dist_p;   // River network distance matrix between prediction points
   int<lower=1> x_r1_p[Np];// Array of group level effect values
   matrix[Np,Np] re1_p;    // Group effect covariance design matrix
   int<lower=1> x_r2_p[Np];// Array of group level effect values
   matrix[Np,Np] re2_p;    // Group effect covariance design matrix
}
transformed data {
  int Kc = K - 1;        // Number of predictors without the intercept
  matrix[N, K - 1] Xc;   // Predictor design matrix without an intercept
  vector[K - 1] means_X; // Vector to hold preditor means
  
  // For each column in X subtract the mean value
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}
parameters{
   vector[Kc] beta;                       // Coefficients approximating predictor mean effects
   positive_ordered[2] sigma;             // I.I.D. (Nugget) and Partial-sill sigma
   real<lower=0, upper=max(dist)> range;  // Range parameter
   real temp_Intercept;                   // Centered intercept
   vector<lower=0>[2] sigma_r;            // Random effect variance
   vector[n_r1] w_r1[1];                  // Group level effect 1
   vector[n_r2] w_r2[1];                  // Group level effect 2
}
transformed parameters{
   // Group-level effects
   vector[n_r1] r1 = (sigma_r[1]*(w_r1[1]));
   vector[n_r2] r2 = (sigma_r[2]*(w_r2[1]));
}
model{
   matrix[N,N] Z;  // Covariance matrix
   matrix[N,N] L;  // Lower diagnal of Z
   
   // Estimate the mean given fixed effects and random effects
   vector[N] mu = temp_Intercept + Xc*beta;
   
   // Calculate the group effects on the mean.
   for (n in 1:N) {
    mu[n] += r1[x_r1[n]] + r2[x_r2[n]];
   }
   
   // Fit the covariance matrix by row and colum using the river distance matrix
   for(i in 1:N){
      for(j in i:N){
         // Exponential tail-down model + group effect variance
         Z[i,j] = sigma[2]*exp(-3*dist[i,j]/range) + sigma_r[1]*re1[i,j] + sigma_r[2]*re2[i,j];
         // Make symetrical
         Z[j,i] = Z[i,j];
      }
      // Add the nugget effect (i.i.d remainder) and random effect variance to the diagonal
      Z[i,i] += sigma[1];
   }
   
   // Decompose for speed boost
   L = cholesky_decompose(Z);
   
   // priors including all constants
   target += student_t_lpdf(temp_Intercept | 3, 11, 10);
   target += student_t_lpdf(sigma | 3, 0, 10)
     - 1 * student_t_lccdf(0 | 3, 0, 10);
   target += student_t_lpdf(sigma_r | 3, 0, 10)
     - 1 * student_t_lccdf(0 | 3, 0, 10);
   target += normal_lpdf(w_r1[1] | 0, 1);
   target += normal_lpdf(w_r2[1] | 0, 1);
   target += student_t_lpdf(range |3, 5, 10);
   
   // Fit observations to the mean with covariance matrix L.
   target += multi_normal_cholesky_lpdf(y | mu, L);
}
generated quantities {
  // predcition estimates
  vector[Np] y_new;
  // prediction covariance matrix
  matrix[Np,Np] Z_new;
  matrix[Np,Np] L_new;
  // actual population-level intercept
  real alpha = temp_Intercept - dot_product(means_X, beta);
  // predict fixed and group effects
  vector[Np] mu_new;
  for(n in 1:Np){
     mu_new[n] = alpha + X_p[n]*beta + r1[x_r1_p[n]] + r2[x_r2_p[n]];
  }
  // predict covariance
  for(i in 1:Np){
     for(j in i:Np){
        Z_new[i,j] = sigma[2]*exp(-3*dist_p[i,j]/range) + sigma_r[1]*re1_p[i,j] + sigma_r[2]*re2_p[i,j];
        Z_new[j,i] = Z_new[i,j];
     }
     Z_new[i,i] += sigma[1];
  }
  
  L_new = cholesky_decompose(Z_new);
  
  y_new = multi_normal_cholesky_rng(mu_new, L_new);
}
