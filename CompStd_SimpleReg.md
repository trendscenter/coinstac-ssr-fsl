# COINSTAC: Non-Interactive Linear Regression

**Synopsis:** This is a COINSTAC computation which approximates a linear regression analysis run on data from multiple sites using a single round of communication between sites and the aggregator.

**Analytical Description:** 
Suppose there are $S$ sites. Each site $s$ has a a collection of $n_s$ covariate-response pairs $\{ (x^{(s)}_j, y^{(s)}_j) : j = 1, 2, \ldots, n_s\}$ where each $x^{(s)}_j$ is a $d$-dimensional vector of covariates (real numbers) stored as a $d \times 1$ array and $y^{(s)}_j$ is a scalar response. Define the total number of samples as $N = \sum_{s=1}^{S} n_s$.

A linear regression at a single site would compute the least squares solution:

$$\beta_s = \mathop{\mathrm{argmin}}_{\beta} \sum_{j=1}^{n_s} (y^{(s)}_j - \beta^{\top} x^{(s)}_j )^2$$

This function has each site $s$ compute $\beta_s$ and then performs as weighted average of the $\beta_s$:

$$\beta_{\mathrm{agg}} = \sum_{s=1}^{S} \frac{n_s}{N} \beta_s$$

## Required Preprocessing

In a linear regression model, we are given covariate-response pairs $\{ (v_j, y_j) : j = 1, 2, \ldots, n\}$ and try to fit a model

$$y = b_0 + b_1 v(1) + b_2 v(2) + \ldots b_{d-1} v(d-1)$$

using least squares. To simplify the problem, we append a 1 to the covariate vector and define
	$$\begin{bmatrix} x_j(1) \\ x_j(2) \\ \vdots \\ x_j(d) \end{bmatrix} = \begin{bmatrix} v_j(1) \\ v_j(2) \\ \vdots \\ v_j(d-1) \\ 1 \end{bmatrix}$$

so that the model is 

$$y = \beta^{\top} x$$

where $\beta = [b, b_0]$. This computation assumes that this preprocessing step has been done already.


## Local and aggregator computations

### Local script

The local script at site $s$ does the following:

1. Reads a local data set $\{ (x^{(s)}_j, y^{(s)}_j) : j = 1, 2, \ldots, n_s\}$ from disk.
2. Computes $\beta_s$, the least-squares regression coefficients
3. Sends $\beta_s$ and $n_s$ to the aggregator.

### Aggregator script

The aggregator does the following:

1. Receives $\{ \beta_s : s =1, 2, \ldots, S\}$ from the $S$ sites.
2. Computes the average $\beta_{\mathrm{agg}} = \sum_{s=1}^{S} \frac{n_s}{N} \beta_s$.
3. Sends $\beta_{\mathrm{agg}}$ to all sites.
4. Deletes the data $\{ (n_s,\beta_s) : s =1, 2, \ldots, S\}$.

## Communication and storage specification

**What data must sites provide?**

* Each site $s$ needs to provide access to their covariate-response pairs $\{ (x^{(s)}_j, y^{(s)}_j) : j = 1, 2, \ldots, n_s\}$.

**What is shared from the sites to the aggregator?**

* The site ID
* The number of samples $n_s$ at that site.
* The locally computed regression coefficients $\beta_s$.

**What intermediate resultes are stored locally at the sites?**

* The sites do not receive any intermediate results.

**What intermediate results are stored at the aggregator?**

* The aggregator deletes $\{ (n_s, \beta_s) : s = 1, 2, \ldots S\}$ after computing $\beta_{\mathrm{agg}}$.

**what is the output from the computation?**

This computation produces a single output file:

* *format*: CSV
* *content*: 
  * $d$-dimensional vector $\beta_{\mathrm{agg}}$
  * ...



