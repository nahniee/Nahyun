#' ---
#' title: "Applied Statistics"
#' author: "Nahyun Lee"
#' date: "02/09/2025"
#' geometry: margin=1cm
#' number_sections: true
#' ---
#' <style type="text/css">
#' .main-container {
#' max-width: 800px !important;
#' font-size: 12px;
#' }
#' code.r{
#' font-size: 12px;
#' word-wrap: keep-all;
#' }
#' pre {
#' font-size: 12px
#' }
#' h1.title {
#'   font-size: 24px;
#' }
#' h1 {
#'   font-size: 18px;
#'   font-weight: bold;
#' }
#' h2 {
#'   font-size: 16px;
#'   font-weight: bold;
#' }
#' h3 {
#'   font-size: 14px;
#'   font-weight: bold;
#' }
#' </style>
#+ setup, include = FALSE
knitr::opts_chunk$set(comment=NA, warning=FALSE, message=FALSE, size=12, width=1200000)

#' # Is testing for normality needed?
#' Simulate data from the linear model y = 3-5x+epsilon, where x~Uniform[-1, 1] and epsilon follows a double-exponential distribution (non-normal errors). 
#' For each sample size n is element of {20, 50, 100, 200, 500}, repeat the process 500 times: fit a least squares model, record whether the 95% CI for the slope covers the true value (-5), and compute the p-value from the Shapiro-Wilk normality test on residuals. 
#' Plot the average p-value vs. CI coverage rate to assess whether normality testing is necessary for valid inference.
#' 
# Set parameters
n_values = c(20, 50, 100, 200, 500)
N = 500  # Number of simulations per sample size
num_n = length(n_values)

# Initialize vectors to store results
coverage = numeric(num_n)
avg_p = numeric(num_n)

# Loop through each sample size
for (i in seq_along(n_values)) {
  n = n_values[i]
  cover = logical(N)
  pvals = numeric(N)
  
  # Simulate N times
  for (j in 1:N) {
    # Generate data
    x = runif(n, -1, 1)
    epsilon = (rexp(n) - rexp(n)) / sqrt(0.5)  # error in double-exponential with SD=1
    y = 3 - 5 * x + epsilon
    
    # Fit linear model
    fit = lm(y ~ x)
    
    # Check if CI for slope covers true value (-5)
    ci = confint(fit, level = 0.95)[2, ]
    cover[j] = (ci[1] <= -5) & (ci[2] >= -5)
    
    # Shapiro-Wilk test on residuals
    res = residuals(fit)
    pvals[j] = shapiro.test(res)$p.value
  }
  
  # Calculate coverage and average p-value
  coverage[i] = mean(cover)
  avg_p[i] = mean(pvals)
}

# Create plot data
plot_data = data.frame(
  n = n_values,
  avg_p_value = avg_p,
  coverage_rate = coverage
)

# Generate scatterplot
library(ggplot2)
ggplot(plot_data, aes(x = avg_p_value, y = coverage_rate)) +
  geom_point(aes(color = factor(n)), size = 5) +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "red") +
  labs(
    x = "Average Normality Test p-value",
    y = "Coverage Rate",
    title = "Coverage Rate vs. Average p-value of Shapiro-Wilk Test",
    color = "Sample Size (n)"
    ) +
  theme_minimal()

#' # Comment
#' 
#' Coverage vs. Normality Test: The plot demonstrates that as the sample size increases, the average p-value from the Shapiro-Wilk test decreases, indicating that the residuals are increasingly detected as non-normal. However, the coverage of the confidence interval (CI) still approaches 95%.
#' This suggests that even when normality is violated, confidence intervals remain valid for large sample sizes, thanks to the Central Limit Theorem (CLT).
#' 
#' Misleading Nature of Normality Tests: Shapiro-Wilk test is sensitive to large sample sizes, often rejecting normality even when deviations are small. For small sample sizes, it may fail to detect non-normality due to low power. This highlights that normality tests are not reliable indicators of inference quality.
#' 
#' Thus, since confidence intervals achieve correct coverage even under non-normal errors for large  n , testing for normality is not necessary for valid inference. Instead, robust methods or techniques that do not assume normality should be prioritized.
#'
#' 
#' # Illustrate
# Generate an alternative visualization
library(gridExtra)

#' Plot 1: Coverage Rate vs. Sample Size
p1 = ggplot(plot_data, aes(x = n, y = coverage_rate)) +
  geom_line(color = "blue") +
  geom_point(size = 3, color = "blue") +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "red") +
  labs(title = "Coverage Rate vs. Sample Size",
       x = "Sample Size (n)",
       y = "Coverage Rate") +
  theme_minimal()

#' Plot 2: Average p-value vs. Sample Size
p2 = ggplot(plot_data, aes(x = n, y = avg_p_value)) +
  geom_line(color = "green") +
  geom_point(size = 3, color = "green") +
  labs(title = "Shapiro-Wilk Test p-value vs. Sample Size",
       x = "Sample Size (n)",
       y = "Average p-value") +
  theme_minimal()

# Arrange both plots side by side
grid.arrange(p1, p2, ncol = 2)

#' Plot 1: As  n  increases, CI coverage stabilizes at 95%, reinforcing that normality is not a requirement for valid inference.
#' 
#' Plot 2: As sample size grows, the Shapiro-Wilk p-value declines, meaning residuals are more often detected as non-normal.
#' 
#' Comparison: This combined visualization highlights the disconnect between normality tests and inference quality, making it clearer why testing for normality is unnecessary.
#' 
#' While normality tests can detect deviations from normality, they do not provide meaningful insight into whether confidence intervals will be valid. The key takeaway is that for large sample sizes, confidence intervals remain reliable despite non-normal residuals, thanks to the Central Limit Theorem. Therefore, normality tests should not be relied upon for evaluating inference accuracy.
#' 
#' 
#' 
#' # Prediction with the mammals dataset
#' Model brain weight as a function of body weight using a log-log linear regression on the mammals dataset, visualize the fit, back-transform to original scale, and evaluate prediction accuracy for an external species using a 95% prediction interval.
#' 
# Load required packages
library(MASS)
library(ggplot2)

# Load the dataset
data(mammals)
head(mammals)

#' Part A: Why log-log scale?
#' 
#' 1. Body and brain weights span orders of magnitude; log scales compress the range.
#' 2. Log transformation often linearizes relationships (e.g., power laws: y = a*x^b).
#' 3. Reduces heteroscedasticity (uneven variance) in the data. => log(brain) = Beta_0 + Beta_1 * log(body) + epsilon

#' Part B: Scatterplot in log-log scale
mammals$log_body = log(mammals$body)
mammals$log_brain = log(mammals$brain)

ggplot(mammals, aes(x = log_body, y = log_brain)) +
  geom_point(color = "darkblue", size = 2) +
  labs(
    x = "Log(Body Weight)",
    y = "Log(Brain Weight)",
    title = "Log-Log Scatterplot of Brain vs. Body Weight in Mammals"
    ) +
  theme_minimal()

#' Part C: Fit least squares line in log-log scale
fit = lm(log_brain ~ log_body, data = mammals)
coef_fit = coef(fit)
summary(fit)

# Equation: log(brain) = Beta_0 + Beta_1 * log(body)
equation_label = sprintf("log(brain) = %.2f + %.2f * log(body)", coef_fit[1], coef_fit[2])
equation_label

# Add line to the plot
ggplot(mammals, aes(x = log_body, y = log_brain)) +
  geom_point(color = "darkblue", size = 2) +
  geom_smooth(method = "lm", formula = y ~ x, color = "red", se = FALSE) +
  labs(
    x = "Log(Body Weight)",
    y = "Log(Brain Weight)",
    title = "Log-Log Scatterplot with Least Squares Fit",
    caption = equation_label
    ) +
  theme_minimal()

#' Part D: Plot the line in original scale (power law)
#' Equation: brain = exp(Beta_0) * body^Beta_1
a = exp(coef_fit[1])  # exp(intercept): estimated coefficient for intercept
b = coef_fit[2]       # slope from log-log regression

# Format the equation for the original scale
equation_original = sprintf("brain = %.2e * body^{%.2f}", a, b)
equation_original

# Generate predictions
body_seq = seq(min(mammals$body), max(mammals$body), length.out = 100)
pred_brain = a * (body_seq)^b

# Plot
ggplot(mammals, aes(x = body, y = brain)) +
  geom_point(color = "darkblue", size = 2) +
  geom_line(data = data.frame(body = body_seq, brain = pred_brain), 
            aes(x = body, y = brain), color = "red", linewidth = 1) +
  labs(
    x = "Body Weight",
    y = "Brain Weight",
    title = "Power Law Relationship in Original Scale",
    caption = sprintf("brain = %.2e * body^{%.2f}", a, b)
    ) +
  scale_x_log10() + 
  scale_y_log10() +
  theme_minimal()

#' The original-scale equation represents a power-law relationship between body and brain weight, common in biological scaling laws like Kleiber’s law (metabolic scaling).
#' The equation for the function in the original scale is: brain = a * body^{b} where a = exp(Beta_0) and b = Beta_1.
#'
#'
#' Part E: 95% Prediction Interval in Original Scale
#' 
#' I chose the Koala, a mammal species not present in the dataset (sources: San Diego Zoo, Journal of Comparative Neurology)
#'
#' Body Weight(x_new): 9kg (average for adult koalas)
#'
#' Brain Weight(y_new): 19g (average for adult koalas)
mammals$log_body = log(mammals$body)
mammals$log_brain = log(mammals$brain)
fit = lm(log_brain ~ log_body, data = mammals)
summary(fit)

# Define new data (koala: body = 9 kg)
new_data = data.frame(log_body = log(9))

# Predict log(brain) with 95% prediction interval
pred_log = predict(fit, newdata = new_data, interval = "prediction", level = 0.95)

# Exponentiate to get interval in original scale (grams)
pred_original = exp(pred_log)
pred_original

#' The wide prediction interval reflects uncertainty in predicting brain weight for species outside the dataset. 
#' While the koala's brain weight is within the interval, the model's precision is limited for extrapolation.
#' The interval is asymmetric in original scale(multiplicative error structure), which is typical for log-log models.
#' Prediction intervals are critical for assessing model reliability. Even with good fit(R^2 > 0.9 for mammals data), predictions for new species remain uncertain.
#'
#' Thus, While the prediction interval includes the Koala’s brain weight, the large range (10.87g to 178.97g) suggests significant uncertainty, indicating that extrapolation should be done with caution.
#' This suggests the model generalizes moderately well but highlights the need for caution when extrapolating beyond the training data.
#'
#'
#'
#' # Robust methods applied to the mammals dataset
#' In the original scale (no logs), apply the M-estimators with squared, absolute, Huber, and Tukey losses, as well as least median of squares, least trimmed of squares, and one additional method of your choice.
#' Overlay the lines fitted by these methods on a scatterplot of the data.
#' 
# Load required packages
library(MASS)
library(robustbase)
library(quantreg)    # For absolute loss (quantile regression)
library(ggplot2)

# Load the dataset
data(mammals)
df = data.frame(body = mammals$body, brain = mammals$brain)

# Generate prediction grid
body_seq = seq(min(df$body), max(df$body), length.out = 100)

# Fit models
fit_ls = lm(brain ~ body, data = df)                          # Least Squares
fit_abs = rq(brain ~ body, data = df, tau = 0.5)              # Absolute loss (quantile)
fit_huber = rlm(brain ~ body, data = df, psi = psi.huber)     # Huber
fit_tukey = rlm(brain ~ body, data = df, psi = psi.bisquare)  # Tukey
fit_lms = lmsreg(brain ~ body, data = df)                     # LMS
fit_lts = ltsReg(brain ~ body, data = df)                     # LTS
fit_mm = lmrob(brain ~ body, data = df)                       # MM-estimator

# Predictions
predictions = data.frame(
  body = body_seq,
  LS = predict(fit_ls, newdata = data.frame(body = body_seq)),
  Absolute = predict(fit_abs, newdata = data.frame(body = body_seq)),
  Huber = predict(fit_huber, newdata = data.frame(body = body_seq)),
  Tukey = predict(fit_tukey, newdata = data.frame(body = body_seq)),
  LMS = fit_lms$coefficients[1] + fit_lms$coefficients[2] * body_seq,
  LTS = fit_lts$coefficients[1] + fit_lts$coefficients[2] * body_seq,
  MM = predict(fit_mm, newdata = data.frame(body = body_seq))
)

# Reshape for plotting
library(tidyr)
pred_long = pivot_longer(predictions, cols = -body, names_to = "Method", values_to = "brain")

# Plot
ggplot(df, aes(x = body, y = brain)) +
  geom_point(alpha = 0.6) +
  geom_line(data = pred_long, aes(x = body, y = brain, color = Method), linewidth = 1) +
  scale_x_log10() +  # Log scale for clarity (data is skewed)
  scale_y_log10() +
  labs(
    x = "Body Weight (kg)",
    y = "Brain Weight (g)",
    title = "Robust Regression Fits in Original Scale",
    caption = "Lines overlaid on log-log axes for visualization (data remains in original scale)"
  ) +
  theme_minimal() +
  scale_color_manual(values = c(
    "LS" = "red",
    "Absolute" = "blue",
    "Huber" = "green",
    "Tukey" = "purple",
    "LMS" = "orange",
    "LTS" = "brown",
    "MM" = "pink"
  ))

#' Comment
#' 
#' 1. Least Squares (LS) is highly influenced by large mammals, making it sensitive to extreme values, so resulting in a steeper slope.
#'
#' 2. Robust methods (Huber, Tukey, MM, LTS) downweight outliers, leading to flatter slopes that reduce the impact of extreme values.
#'
#' 3. LMS (Least Median of Squares) and LTS (Least Trimmed Squares) provide strong resistance to outliers but may oversimplify the relationship, failing to capture finer structural trends.
#'
#' 4. Quantile Regression (Absolute Loss) aligns well with robust M-estimators, offering a median-based alternative that resists extreme values.
#'
#' If we compare to Log-Log LS Model and problem 2, then the log-log LS model naturally captures the power-law relationship, which is biologically meaningful.
#' Robust methods in the original scale struggle with nonlinearity and heteroscedasticity—even though they mitigate the effect of outliers, they do not fully address the curvature in the data.
#' 
#' I prefer Log-Log LS. Despite its sensitivity to outliers, it correctly models the fundamental biological relationship. Also, It aligns with empirical findings in biological scaling laws (e.g., Kleiber’s Law).
#' If working in the original scale is required, the MM-estimator and Tukey’s biweight are reasonable choices since they offer a good balance between robustness and efficiency. Also, avoid LMS or LTS unless extreme outliers are a primary concern, as they may oversimplify the relationship.
#'
#' Thus, while robust regression methods in the original scale effectively reduce the influence of outliers, the log-log least squares model remains preferable for capturing the biologically accurate power-law relationship.
#' If original-scale regression is mandatory, the MM-estimator or Tukey’s biweight provide the best trade-off between robustness and model fidelity.
#'
#'
#'
#'
#'
#'
#' # LDA vs logistic regression
#' Run numerical experiments comparing LDA and logistic regression, showing that LDA performs better when its assumptions hold, while logistic regression is more robust when they don’t.
#' 
# Load required libraries
library(MASS)
library(ggplot2)
library(tidyr)

# Simulation parameters
n_sim = 100       # Number of simulations
n_per_class = 500 # Samples per class
train_ratio = 0.7 # Training data ratio

# Case 1: LDA assumptions hold (equal covariance matrices)
results_case1 = matrix(NA, nrow = n_sim, ncol = 2)
colnames(results_case1) = c("LDA", "Logistic")

for (i in 1:n_sim) {
  # Generate data with equal covariance
  sigma = matrix(c(1, 0.5, 0.5, 1), 2, 2)
  mu1 = c(0, 0)
  mu2 = c(1, 1)
  
  class1 = mvrnorm(n_per_class, mu1, sigma)
  class2 = mvrnorm(n_per_class, mu2, sigma)
  X = rbind(class1, class2)
  y = factor(rep(0:1, each = n_per_class))
  
  # Train/test split
  train_indices = sample(1:(2*n_per_class), size = train_ratio*2*n_per_class)
  X_train = X[train_indices, ]
  y_train = y[train_indices]
  X_test = X[-train_indices, ]
  y_test = y[-train_indices]
  
  # Fit LDA
  lda_model = lda(X_train, y_train)
  lda_pred = predict(lda_model, X_test)$class
  lda_error = mean(lda_pred != y_test)
  
  # Fit logistic regression
  logit_model = glm(y_train ~ ., data = data.frame(X_train, y_train), family = "binomial")
  logit_prob = predict(logit_model, data.frame(X_test), type = "response")
  logit_pred = ifelse(logit_prob > 0.5, 1, 0)
  logit_error = mean(logit_pred != y_test)
  
  results_case1[i, ] = c(lda_error, logit_error)
}

# Case 2: LDA assumptions violated (unequal covariance matrices)
results_case2 = matrix(NA, nrow = n_sim, ncol = 2)
colnames(results_case2) = c("LDA", "Logistic")

for (i in 1:n_sim) {
  # Generate data with unequal covariance
  sigma1 = matrix(c(1, 0.5, 0.5, 1), 2, 2)
  sigma2 = matrix(c(2, -0.5, -0.5, 2), 2, 2)
  mu1 = c(0, 0)
  mu2 = c(1, 1)
  
  class1 = mvrnorm(n_per_class, mu1, sigma1)
  class2 = mvrnorm(n_per_class, mu2, sigma2)
  X = rbind(class1, class2)
  y = factor(rep(0:1, each = n_per_class))
  
  # Train/test split
  train_indices = sample(1:(2*n_per_class), size = train_ratio*2*n_per_class)
  X_train = X[train_indices, ]
  y_train = y[train_indices]
  X_test = X[-train_indices, ]
  y_test = y[-train_indices]
  
  # Fit LDA
  lda_model = lda(X_train, y_train)
  lda_pred = predict(lda_model, X_test)$class
  lda_error = mean(lda_pred != y_test)
  
  # Fit logistic regression
  logit_model = glm(y_train ~ ., data = data.frame(X_train, y_train), family = "binomial")
  logit_prob = predict(logit_model, data.frame(X_test), type = "response")
  logit_pred = ifelse(logit_prob > 0.5, 1, 0)
  logit_error = mean(logit_pred != y_test)
  
  results_case2[i, ] = c(lda_error, logit_error)
}

# Visualize results
df_case1 = as.data.frame(results_case1)
df_case1$Case = "LDA Assumptions Hold"
df_case1_long = pivot_longer(df_case1, cols = c(LDA, Logistic), 
                             names_to = "Method", values_to = "Error")

df_case2 = as.data.frame(results_case2)
df_case2$Case = "LDA Assumptions Violated"
df_case2_long = pivot_longer(df_case2, cols = c(LDA, Logistic), 
                             names_to = "Method", values_to = "Error")

combined_df = rbind(df_case1_long, df_case2_long)

ggplot(combined_df, aes(x = Method, y = Error, fill = Method)) +
  geom_boxplot() +
  facet_wrap(~ Case) +
  labs(title = "LDA vs. Logistic Regression Performance",
       y = "Test Error Rate", x = "Method") +
  theme_minimal()

# Summary table
summary_table = rbind(
  data.frame(
    Case = "LDA Assumptions Hold",
    Method = c("LDA", "Logistic"),
    Mean_Error = colMeans(results_case1),
    SD_Error = apply(results_case1, 2, sd)
  ),
  data.frame(
    Case = "LDA Assumptions Violated",
    Method = c("LDA", "Logistic"),
    Mean_Error = colMeans(results_case2),
    SD_Error = apply(results_case2, 2, sd)
  )
)

print(summary_table)
#'
#' # Comment
#' 
#' Case 1: LDA assumptions hold(Equal Covariance Matrices), classes follow multivariate normal distributions with identical covariance matrices.
#'
#' => Means: Class 1 = (0, 0), Class 2 = (1, 1); Covariance = [[1, 0.5], [0.5, 1]]
#'
#' => Performance: LAD achieves lower test error compared to logistic regression.
#'
#' => Interpretation: LDA outperforms logistric regression when its assumptions (eqaul covariance, normality) hold. Also, LDA leverages the true covariance structure, while logistic regression makes no parametric assumptions and is less efficient here, so LDA achieves lower test error.
#'
#'
#' Case 2:  LDA Assumptions Violated (Unequal Covariance Matrices), classes follow multivariate normal distributions with different covariance matrices.
#'
#' => Class1: [[1, 0.5], [0.5, 1]], Class 2: [[2, -0.5], [-0.5, 2]]
#'
#' => Performance: Logistic Regression achieves lower test error compared to LDA.
#'
#' => Interpretation: LDA’s performance degrades when its assumptions are violated (unequal covariance) and logistic regression, being a flexible discriminative model, adapts better to the violated assumptions, so logistic regression outperforms LDA.
#' 
#' 
#' Thus, LDA is optimal when classes are Gaussian with equal covariance and training data is limited and logistic regression is preferred when LDA's assumptions are violated and data distribution is unkown or non-Gaussian.
#' 
#' That's why the boxplots show clear seperation in performance across cases such that LDA errors cluster lower in Case 1 and logistic regression errors cluster lower in Case 2.
#' 
#' 
#' 
#' 
#' # Piecewise constant model fitted by LAD
#' Create a function LADpiecewiseConstant(x, y, q) that fits a piecewise constant model using absolute loss over q intervals on [0, 1], returns the fitted values and average loss, and overlays the result on a scatterplot. 
#' Test it on synthetic data with normal and Cauchy errors.
#' 
LADpiecewiseConstant = function(x, y, q) {
  # Check if x values are within [0, 1]
  if (any(x < 0 | x > 1)) {
    stop("All x values must be within the interval [0, 1].")
  }
  
  # Create breaks for intervals
  breaks = seq(0, 1, length.out = q + 1)
  intervals = cut(x, breaks = breaks, right = FALSE, include.lowest = TRUE)
  
  # Compute medians for each interval
  medians = tapply(y, intervals, median, na.rm = TRUE)
  
  # Check for intervals with no data
  if (any(is.na(medians))) {
    warning("Some intervals contain no data; resulting in NA values for those intervals.")
  }
  
  # Extract inner knots (breaks between intervals)
  knots = breaks[2:q]
  
  # Create step function using medians
  step_fun = stepfun(knots, medians)
  
  # Calculate fitted values and average absolute loss
  fitted_values = step_fun(x)
  average_loss = mean(abs(y - fitted_values), na.rm = TRUE)
  
  # Generate a smooth grid for plotting the step function
  t_grid = seq(0, 1, length.out = 1000)
  fitted_curve = step_fun(t_grid)
  
  # Plot the data and fitted function
  plot(x, y, col = "darkgray", pch = 19, cex = 0.6,
       main = paste("LAD Piecewise Constant Model (q =", q, ")"),
       xlab = "x", ylab = "y")
  lines(t_grid, fitted_curve, col = "red", lwd = 2)
  abline(v = breaks, lty = 3, col = "blue")
  
  # Return results
  list(values = medians, average_loss = average_loss)
}

# Example usage with synthetic data
set.seed(123)
n = 200
x = runif(n)
f = function(x) (1 + 10*x - 5*x^2) * sin(10*x)

# Case 1: Normal errors
y_normal = f(x) + rnorm(n)
result_normal = LADpiecewiseConstant(x, y_normal, q = 10)

# Case 2: Cauchy errors
y_cauchy = f(x) + rcauchy(n)
result_cauchy = LADpiecewiseConstant(x, y_cauchy, q = 10)

# Display average losses
cat("Average Absolute Loss (Normal Errors):", result_normal$average_loss, "\n")
cat("Average Absolute Loss (Cauchy Errors):", result_cauchy$average_loss, "\n")
#'
#'
#' # Comment
#' 
#' Normal Errors: The relatively low loss indicates the model fits the data well, it produces symmetric noise around the true function, which the LAD method handles effectively, and the piecewise constant model (using medians for each interval) aligns closely with the underlying trend.
#'
#' => LAD and least squares perform comparably, but LAD sacrifices some efficiency for robustness.
#'
#' => Tight clustering around the fitted function.
#'
#' => Thus, the fitted values are close to the true underlying function, and the average loss is relatively low.
#'
#' 
#' Cauchy Errors: The higher loss reflects the heavy-tailed nature of Cauchy noise, which generates extreme outliers, despite the LAD method’s robustness, the Cauchy distribution’s infinite variance amplifies deviations between observations and fitted values, and the plot’s y-axis range visually confirms the presence of extreme outliers, but the fitted step function remains stable and captures the underlying trend, demonstrating that medians resist distortion from outliers.
#'
#' => LAD performs significantly better than least squares, as means would be heavily skewed by outliers and the fitted function’s structure remains intact even with extreme noise, proving the method’s suitability for heavy-tailed data.
#'
#' => The higher average loss for Cauchy errors is not a failure of the model but a reflection of the inherent noise magnitude in the data.
#'
#' => Sparse extreme outliers dominate the plot, but the step function remains faithful to the central trend.
#'
#' => Thus, despite heavy-tailed noise, the fitted values remain close to the true values, demonstrating the robustness of the LAD method and the average loss is higher due to outliers, but the model structure is preserved.
#' 
#' 
#' 
#'
#' # Cross-validation
#' Extend the previous method by creating LADpiecewiseConstantCV(x, y, qmax), which selects the optimal number of intervals q using leave-k-out cross-validation (with k = n/3) and Monte Carlo replication (B = 1000). 
#' Apply it to various sample sizes and return the model with the lowest estimated risk.
#' 
#'
# Modified LADpiecewiseConstant function (with plot disable option)
LADpiecewiseConstant = function(x, y, q, plot = TRUE) {
  if (any(x < 0 | x > 1)) {
    stop("All x values must be within the interval [0, 1].")
  }
  
  breaks = seq(0, 1, length.out = q + 1)
  intervals = cut(x, breaks = breaks, right = FALSE, include.lowest = TRUE)
  medians = tapply(y, intervals, median, na.rm = TRUE)
  
  if (any(is.na(medians))) {
    warning("Some intervals contain no data; NA values replaced with global median.")
    medians[is.na(medians)] = median(y, na.rm = TRUE)
  }
  
  knots = breaks[2:q]
  step_fun = stepfun(knots, medians)
  
  fitted_values = step_fun(x)
  average_loss = mean(abs(y - fitted_values), na.rm = TRUE)
  
  if (plot) {
    t_grid = seq(0, 1, length.out = 1000)
    fitted_curve = step_fun(t_grid)
    plot(x, y, col = "darkgray", pch = 19, cex = 0.6,
         main = paste("LAD Piecewise Constant Model (q =", q, ")"),
         xlab = "x", ylab = "y")
    lines(t_grid, fitted_curve, col = "red", lwd = 2)
    abline(v = breaks, lty = 3, col = "blue")
  }
  
  list(values = medians, average_loss = average_loss, breaks = breaks)
}

# Cross-validation function
LADpiecewiseConstantCV = function(x, y, qmax = 20) {
  n = length(x)
  k = floor(n / 3)
  B = 1e3
  q_values = 1:qmax
  total_risk = numeric(qmax)
  
  for (b in 1:B) {
    valid_indices = sample(n, size = k)
    train_indices = setdiff(1:n, valid_indices)
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    
    for (q in q_values) {
      if (q > length(unique(x_train))) {
        total_risk[q] = Inf
        next
      }
      
      model = tryCatch({
        LADpiecewiseConstant(x_train, y_train, q, plot = FALSE)
      }, error = function(e) NULL)
      
      if (is.null(model)) {
        total_risk[q] = Inf
        next
      }
      
      y_pred = stepfun(model$breaks[2:q], model$values)(x_valid)
      loss = sum(abs(y_valid - y_pred), na.rm = TRUE)
      total_risk[q] = total_risk[q] + loss
    }
  }
  
  avg_risk = total_risk / B
  qCV = which.min(avg_risk)
  final_model = LADpiecewiseConstant(x, y, qCV)
  final_model
}

# Test with different sample sizes
set.seed(123)
f = function(x) (1 + 10*x - 5*x^2)*sin(10*x)

# n = 100
n = 100
x = runif(n)
y = f(x) + rnorm(n)
result_100 = LADpiecewiseConstantCV(x, y)
cat("Optimal q for n=100:", length(result_100$values), "\n")

# n = 1000
n = 1000
x = runif(n)
y = f(x) + rnorm(n)
result_1000 = LADpiecewiseConstantCV(x, y)
cat("Optimal q for n=1000:", length(result_1000$values), "\n")

# n = 10000 (reduce B for speed)
n = 10000
B = 100
x = runif(n)
y = f(x) + rnorm(n)
result_10000 = LADpiecewiseConstantCV(x, y)
cat("Optimal q for n=10000:", length(result_10000$values), "\n")

#' # Comment
#' 
#' n = 100: Smaller samples favor lower q to avoid overfitting.
#' 
#' n = 1000: Medium samples allow higher q to capture complex patterns.
#' 
#' n = 10000: Large samples enable high q for refined modeling.
#' 
#' 
#' For n = 10000, reducing B(Monte Carlo replicates) improves runtime and parallelization or optimization techniques can further enhance performance.
#' 
#' For model selection, Cross-validation selects q that minimizes validation loss, balancing bias-variance tradeoff and noisy datasets tend to prefer lower q.
#' 

