# Optimization with Genetic Algorithms

This repository holds code and demonstrative notebooks for optimization with genetic algorithms (GA). Much of this code is adapted from something I originally wrote in MATLAB back in 2007 or so. The [lecture notebook](./notebooks/lecture.ipynb) acts as sort of lecture / demonstration notes. This notebook covers

- some mathematical optimization concepts
- the GA
	- as a metaphor
	- in detail
	- its operators / parameters
- frequentist statistics
- the three demonstrative use cases listed below.

## Feature Selection for Machine Learning
The [GA Feature Selection Notebook](./notebooks/GA_FeatureSelection.ipynb) demonstrates feature selection with the GA for a given dataset. The objective function fits a machine learning model - either regression or classification - then computes the error to be minimized. The data is simulated using a known dependence structure, so it is possible to assess the accuracy of the GA result.

Statistical modelers have been trying models on subsets of features for almost as long as statistical modeling (most of what we call "machine learning" is actually statistical modeling) has been around. Perhaps unimaginably, we call the process of selecting a subset of available features [feature selection](https://en.wikipedia.org/wiki/Feature_selection). In feature selection, we use some procedure to generate subsets of the existing features, fit a model to them, and evaluate that model to find an optimal subset. The goal of feature selection is usually to balance two considerations: model performance and model complexity. It is generally beneficial for a model to be simpler - to use fewer features, for example. Practitioners often prefer a simpler model, even if it performs slightly worse than a more complex model. This follows the principle of [Occam's Razor](https://en.wikipedia.org/wiki/Occam%27s_razor).

A simple way to perform feature selection, that guarantees finding the most optimal subset of features, is combinatorial enumeration - a.k.a. brute force. Combinatorial enumeration does exactly what it sounds like - the model is evauated on the complete enumeration of all possible combinations of features. This is no mean feat, as the number of ways to combine `p` features is exponential in `p`; there are `2^p-1` possible nontrivial subsets.

## Real-valued Optimization: Best-fitting Distribution Selection
The [GA Real Optimization - Statistical Distribution Fitting](./notebooks/GA_RealOptimization_DistFit.ipynb) demonstrates using the GA to find the best fitting distribution for a given dataset. The data is simulated from a distribution with known parameters, so it is possible to assess the accuracy of the GA result.

There are three major perceptions of data in statistics:

- [Frequentist](https://en.wikipedia.org/wiki/Frequentist_inference) - considers observed data to be a random sample from an unknown population generated by a "real" probability distribution
- [Bayesian](https://en.wikipedia.org/wiki/Bayesian_inference) - considers observed data to be "real", which can be represented by a hypothesized probability distribution
- [Information Theoretic](https://en.wikipedia.org/wiki/Information_theory) - focuses on determining the maximal amount of information in (or that can be gleaned from) some data

The Frequentist perspective underlies the majority of statistical thinking, and gives us hypothesis testing and confidence intervals. An exercise commonly performed in statistics - whether Frequentist or Bayesian - is that of determining a statistical probability distribution which fits a dataset best, given a vector of parameters (the length of which depends on the distribution).

Frequentists will pick the distribution and it's parameters by maximizing the likelihood function (the product of the probability densities for each observed datapoint), or perhaps the log likelihood (the sum of the log of the probability densities for each observed datapoint). The [maximum likelihood estimate](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), or MLE, is the parameter vector which has the highest probability of generating the sample data observed. Instead of finding the parameters which maximize the log likelihood, Bayesians will use the [highest posterior density interval (or credibility interval)](https://en.wikipedia.org/wiki/Credible_interval). MLE's for some statistical distributions, such as the Gaussian, can be found analytically, computed as a function of the observed sample data. For most other probability distributions, the likelihood function must be numerically optimized. If the MLE for a distribution fit to a dataset gives the parameters most likely to have generated the observed sample data, then we can pick the distribution most likely to have generated the sample data as the distribution associated with the maximum likelihood, when all are evaluated at their MLE's.

To use the GA to find the MLE's for a distribution with `n` parameters, using `q_i` bits to encode each, the binary words on which the GA operates should be of length equal to the sum `q_0+q_1+...+q_n`. There is no requirement for `q_i = q_j` (`i!=j`).


## Real-valued Optimization: Multivariate Function Minimization
The [GA Real Optimization - Multivariate Minimzation](./notebooks/GA_RealOptimization_MultivarMin.ipynb) notebook demonstrates the performance of the GA on 13 test functions. Since the minima are known, we can assess the accuracy of the GA result.

Numerical optimization of mathematical functions is an important topic, and has attracted a lot of research by some of the most brilliant mathematicians and computer scientists over the years. As researchers develop and test novel optimization algorithms, it is important that they can evaluate and compare their strengths and weaknesses. To this end, there are many benchmark functions with known optima that present different challenges to algorithms and allow characterisation of

- accuracy & precision
- rate of convergence
- robustness wrt noise and / or initialization
- performance

Many of them are listed [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization). The Sphere function, for example, should be relatively easy to minimize, as there is only a single minimum and constant gradient everywhere. Others, such as the Rastrigin or Ackley functions, have several local minima, and can be difficult for gradient-following functions to minimize.