import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

white_wine_dataset = np.loadtxt("winequality-white.csv", delimiter=";",skiprows=1)
white_wine_features = ("Fixed acidity", "Volatile acidity", "Citric acid", 
	                   "Residual sugar", "Chlorides", "Free sulfur dioxide", 
	                   "Total sulfur dioxide", "Density", "pH", 
	                   "Sulphates", "Alcohol")

white_wine_quality = white_wine_dataset[:, -1]
white_wine_dataset = white_wine_dataset[:, 0:-1]

# Solve the the least-squares regression problem
beta1 = cvx.Variable(np.shape(white_wine_dataset)[1])
obj1 = cvx.Minimize(cvx.norm(white_wine_dataset @ beta1 - white_wine_quality, 2))
prob1 = cvx.Problem(obj1)
prob1.solve(verbose=False)

print("Optimal value for l2-regression:  ", prob1.value)

# Solve the l1-regression problem
beta2 = cvx.Variable(np.shape(white_wine_dataset[1]))
obj2 = cvx.Minimize(cvx.norm(white_wine_dataset @ beta2 - white_wine_quality, 1))
prob2 = cvx.Problem(obj2)
prob2.solve(verbose=False)

print("Problem value for l1-regression:  ", prob2.value)

# Compute the infinity-norm of the difference between the l2-regression 
# coefficients and the l1-regression coefficients
inf_norm_regression_coeffs = np.linalg.norm(beta1.value - beta2.value, ord=np.inf)
print("Infinity norm between l2- and l1-regression coefficients: ", inf_norm_regression_coeffs)

# Make a bar plot of the l1- and l2-regression coefficients
width = 0.3
labels = np.arange(1, np.shape(white_wine_dataset)[1] + 1)

plt.figure(1)
plt.bar(labels, beta1.value, width, label="l2-regression coeffs")
plt.bar(labels + width, beta2.value, width, label="l1-regression coeffs")

plt.xticks(labels + width/2, white_wine_features)
plt.xlabel("Features for white wine quality")
plt.ylabel("Regression coefficient values")
plt.title("l2- and l1-regression coefficients")
plt.legend(loc="best")

# We remove the outliers by taking the infinity norm of each row of the dataset
# and using the IQR to remove rows whose infinity norms are below the 25th and
# are above the 75th percentiles
inf_norms_white_wine_dataset = np.linalg.norm(white_wine_dataset, ord=np.inf, axis=1)
num_rows = np.size(inf_norms_white_wine_dataset)
lower_quartile = np.median(np.sort(inf_norms_white_wine_dataset)[0:int(num_rows/2)])
upper_quartile = np.median(np.sort(inf_norms_white_wine_dataset)[int(num_rows/2):])

non_outlier_rows = np.logical_and(inf_norms_white_wine_dataset > lower_quartile,
	                              inf_norms_white_wine_dataset < upper_quartile)



white_wine_dataset_no_outliers = white_wine_dataset[non_outlier_rows, :]
white_wine_quality_no_outliers = white_wine_quality[non_outlier_rows]

# Solve the least-squares regression problem with outliers removed
beta3 = cvx.Variable(np.shape(white_wine_dataset_no_outliers)[1])
obj3 = cvx.Minimize(cvx.norm(white_wine_dataset_no_outliers @ beta3 - white_wine_quality_no_outliers, 2))
prob3 = cvx.Problem(obj3)
prob3.solve(verbose=False)

print("Optimal value for l2-regression  (outliers removed): ", prob3.value)

# Solve the l1-regression problem with outliers removed
beta4 = cvx.Variable(np.shape(white_wine_dataset_no_outliers)[1])
obj4 = cvx.Minimize(cvx.norm(white_wine_dataset_no_outliers @ beta4 - white_wine_quality_no_outliers, 1))
prob4 = cvx.Problem(obj4)
prob4.solve(verbose=False)

print("Problem status for l1-regression (outliers removed): ", prob4.status)
print("Optimal value for l1-regression  (outliers removed): ", prob4.value)

# Make a bar plot of the l1- and l2- regression coefficients, after removing
# the outlier rows
plt.figure(2)
plt.bar(labels, beta3.value, width, label="l2-regression coefficients")
plt.bar(labels + width, beta4.value, width, label="l1-regression coefficients")

plt.xticks(labels + width/2, white_wine_features)
plt.xlabel("Features for white wine quality")
plt.ylabel("Regression coefficient values")
plt.title("l2- and l1-regression coefficient values (outliers removed)")
plt.legend(loc="best")
plt.show()