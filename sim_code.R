library(SSN)
library(rstan)
library(tidyverse)
library(brms)

# Build a simulated river network.
set.seed(4)
tree = createSSN(10,
          obsDesign = binomialDesign(c(40)),
          predDesign = poissonDesign(2),
          importToR = TRUE,
          path = "./Sim.ssn",
          treeFunction = iterativeTreeLayout)

# Plot Network and Sample Points
plot(tree, lwdLineCol = "addfunccol", lwdLineEx = 8,
     lineCol = "blue", cex = 2, xlab = "x-coordinate",
     ylab = "y-coordinate", pch = 1)
plot(tree, PredPointsID = "preds", add = TRUE, cex = .5, pch = 19,
     col = "green")

# Create Distance Matrix and Simulate Data
createDistMat(tree, "preds", o.write=TRUE, amongpred = TRUE)

# Get Observation points
DFobs = getSSNdata.frame(tree, "Obs")
DFpred = getSSNdata.frame(tree, "preds")
# Create covariates
DFobs[,"X1"] = rnorm(length(DFobs[,1]))
DFpred[,"X1"] = rnorm(length(DFpred[,1]))
DFobs[,"X2"] = rnorm(length(DFobs[,1]))
DFpred[,"X2"] = rnorm(length(DFpred[,1]))
# Create factor covariates
DFobs[,"F1"] = as.factor(sample.int(4,length(DFobs[,1]), replace = TRUE))
DFpred[,"F1"] = as.factor(sample.int(4,length(DFpred[,1]), replace = TRUE))
# Create random effects
DFobs[,"RE1"] = as.factor(sample(1:3,length(DFobs[,1]), replace = TRUE))
DFobs[,"RE2"] = as.factor(sample(1:4,length(DFobs[,1]), replace = TRUE))
DFpred[,"RE1"] = as.factor(sample(1:3,length(DFpred[,1]), replace = TRUE))
DFpred[,"RE2"] = as.factor(sample(1:4,length(DFpred[,1]), replace = TRUE))

# Fit to GLM model in SSN
sim.out = SimulateOnSSN(ssn.object = tree, ObsSimDF = DFobs, 
                        PredSimDF = DFpred, PredID = "preds",
                        formula = ~ X1 + X2 + F1, coefficients = c(10,1,0,-2,0,2),
                        CorModels = c("Exponential.taildown","RE1","RE2"),
                        use.nugget = T, CorParms = c(2, 10, 1, .5, .1),
                        addfunccol = "addfunccol")

Torg <- Torgegram(sim.out$ssn.object, "Sim_Values")
plot(Torg)

# Fit Simulated SSN
glmssn.out <- glmssn(Sim_Values ~ X1 + X2 + F1, sim.out$ssn.object,
                     CorModels = c("Exponential.taildown","RE1","RE2"),
                     addfunccol = "addfunccol", use.nugget = T)
summary(glmssn.out)

# Exract the junction distance matrix. 
# Distance to common junction downstream matrix. (see page 13 of vignette)
distObs = getStreamDistMat(sim.out$ssn.object)$dist.net1
dist.H.o = distObs + t(distObs) # Total Distance between points.
#dwnStrm_O = 1*(distObs == 0) # Flow connected design matrix

distPreds = getStreamDistMat(sim.out$ssn.object, "preds")$dist.net1 
dist.H.p = distPreds + t(distPreds) # Total Distance between points.
#dwnStrm_P = 1*(distPreds == 0) # Flow connected design matrix

# Get model data with factored columns.
df = sim.out$ssn.object@obspoints@SSNPoints[[1]]@point.data # Get Data
x = model.matrix(Sim_Values ~ X1 + X2 + F1, data = df) # Make Matrix
re1 = model.matrix(~RE1-1, data = df); re1 = re1%*%t(re1)
re2 = model.matrix(~RE2-1, data = df); re2 = re2%*%t(re2)

# Get prediction data and matrices
df_pred = sim.out$ssn.object@predpoints@SSNPoints[[1]]@point.data # Get prediction data
xp = model.matrix(Sim_Values ~ X1 + X2 + F1, data = df_pred); xp = xp[,-1] # Make Matrix
re1_p = model.matrix(~RE1-1, data = df_pred); re1_p = re1_p%*%t(re1_p)
re2_p = model.matrix(~RE2-1, data = df_pred); re2_p = re2_p%*%t(re2_p)

# Set up multiple cores.
options(mc.cores = parallel::detectCores(), auto_write = TRUE)
# Compile Model.
compile = stan_model("./sim_ssn.stan")
#Run model.
mod = sampling(compile, data = list(N = nrow(df),
                                    K = ncol(x),
                                    X = x,
                                    y = df$Sim_Values,
                                    n_r1 = length(unique(df$RE1)),
                                    x_r1 = as.numeric(df$RE1),
                                    re1 = re1,
                                    n_r2 = length(unique(df$RE2)),
                                    x_r2 = as.numeric(df$RE2),
                                    re2 = re2,
                                    dist = dist.H.o,
                                    Np = nrow(df_pred),
                                    X_p = xp,
                                    dist_p = dist.H.p,
                                    x_r1_p = as.numeric(df_pred$RE1),
                                    re1_p = re1_p,
                                    x_r2_p = as.numeric(df_pred$RE2),
                                    re2_p = re2_p),
               iter = 500, chains = 4, thin = 1, 
               pars = c("y_new","beta","range","alpha","sigma","sigma_r"), include = T,
               control = list(adapt_delta = .8, max_treedepth = 10), save_warmup = T)

shinystan::launch_shinystan(mod)

posterior = data.frame(as.matrix(mod))[,1:24]
tdy_post = gather(posterior, "pred","est")
tdy_post$pred = as.numeric(str_extract(tdy_post$pred, '\\d+'))
med = tdy_post %>% group_by(pred) %>% summarise(med = median(est))
med$obs = df_pred$Sim_Values



library(ggridges)
ggplot() + 
  stat_density_ridges(data = tdy_post,
                      aes(x=est, y=as.factor(pred), fill=factor(..quantile..)),
                      geom = "density_ridges_gradient", 
                      calc_ecdf = TRUE, quantiles = c(0.1, 0.9)) + 
  scale_fill_manual(name = "Probability", values = c("#FF0000A0", "#A0A0A0A0", "#0000FFA0"),
    labels = c("(0, 0.1]", "(0.1, 0.9]", "(0.9, 1]")) +
  geom_segment(data = med, aes(x = obs, xend = obs, y = pred, yend = pred + .9), color = "red") +
  xlim(c(-2,15)) + theme_ridges(grid = F, center = T)


fit = brm(brmsformula(Sim_Values ~ X1 + X2 + F1 + (1 | RE1) + (1 | RE2), center = T), data = df)
stancode(fit)
get_prior(brmsformula(Sim_Values ~ X1 + X2 + F1 + (1 | RE1) + (1 | RE2), center = T), data = df)
