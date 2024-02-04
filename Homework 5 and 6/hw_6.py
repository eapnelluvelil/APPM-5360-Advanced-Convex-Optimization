import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import timeit

def gradientCheck(f, grad_f, x0, title):
	# Determine the number of entries in x0 to determine the number
	# of unit vectors to use
	x0 = np.array(x0)
	m = x0.size
	I = np.eye(m)

	# Implement the first and second checks in the appendix
	stepsizes = 1.5 * np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11])
	gradient_errors_fd = np.zeros(stepsizes.size)
	gradient_errors_cd = np.zeros(stepsizes.size)

	for i in np.arange(stepsizes.size):
		h = stepsizes[i]
		approx_grad_fd = np.zeros((m, ))
		approx_grad_cd = np.zeros((m, ))
		for j in np.arange(m):
			e_i = I[:, j]

			# Compute a first-order forward difference approximation to
			# the gradient
			approx_grad_fd[j] = (f(x0 + h*e_i) - f(x0))/h
			# Compute a second-order centered difference approximation to
			# the gradient
			approx_grad_cd[j] = (f(x0 + h*e_i) - f(x0 - h*e_i))/(2*h);

		gradient_errors_fd[i] = np.linalg.norm(grad_f(x0) - approx_grad_fd, 2)
		gradient_errors_cd[i] = np.linalg.norm(grad_f(x0) - approx_grad_cd, 2)

	plt.figure(1)
	# Plot the forward difference and centered difference errors
	# in the gradient approximations
	plt.plot(stepsizes, gradient_errors_fd, label="Forward difference error")
	plt.plot(stepsizes, gradient_errors_cd, label="Centered difference error")
	# Plot O(h^(-1)) and O(h^(-2)) reference error lines
	plt.plot(stepsizes, stepsizes, label="O(h) reference line")
	plt.plot(stepsizes, np.power(stepsizes, 2), label="O(h^(2)) reference line")
	plt.xscale("log", base=10)
	plt.yscale("log", base=10)
	plt.xlabel("Stepsize h (log-scale)")
	plt.ylabel("2-norm error of gradient approximation (log-scale)")
	plt.title(title)
	plt.legend()
	plt.show()

def gradientDescent(f, grad_f, x0, x_star, title, max_iters=100, stepsize=1, tol=1e-7, compute_error=False):
	num_iters   = 0
	errors      = []
	xn          = np.zeros(x0.size)

	while (num_iters < max_iters):
		num_iters += 1
		xn = x0 - stepsize*grad_f(x0)

		# Compute the error between the new iterate and true solution,
		# if provided
		if compute_error:
			errors.append(np.linalg.norm(xn - x_star, 2))

		if (np.linalg.norm(xn - x0, 2) <= tol):
			break

		x0 = xn

	errors = np.array(errors)

	# Make a plot of error between each iterate and the true solution
	# as a function of the iteration count
	if compute_error:
		plt.figure()
		plt.plot(np.arange(num_iters), errors, label="Error between iterate and true solution")
		# plt.xscale("log", base=10)
		plt.yscale("log", base=10)
		plt.xlabel("Iteration count")
		plt.ylabel("2-norm of error between iteration and true solution (log-scale)")
		plt.legend()
		plt.title(title)
		plt.show()

	return xn

def gradientDescentLS(f, grad_f, x0, x_star, error_title, eval_title, max_iters=100, stepsize=1, tol=1e-7, compute_error=False):
	num_iters  = 0
	func_evals = []
	errors     = []
	xn         = np.zeros(x0.size)

	# Typical line search parameters
	c = 1e-4
	rho = 0.9
	t = stepsize

	while (num_iters < max_iters):
		num_iters += 1
		
		# Perform a line search to refine the stepsize
		# Our descent direction will be the negative gradient at the current iterate
		armijo_cond_iters = 0
		pn = -grad_f(x0)
		while (f(x0 + t*pn) > f(x0) - c*t*np.vdot(grad_f(x0), pn)):
			t *= rho
			armijo_cond_iters += 1

		xn = x0 - t*grad_f(x0)
		func_evals.append(f(xn))

		# Compute the error between the new iterate and the true solution,
		# if provided
		if compute_error:
			errors.append(np.linalg.norm(xn - x_star, 2))

		if (np.linalg.norm(xn - x0, 2) < tol):
			break

		x0 = xn

		# Increase the stepsize by a factor of two 
		# if we decrease the stepsize only once
		if armijo_cond_iters == 1:
			t *= 2

	# Make a plot of error between each iterate and the true solution
	# as a function of the iteration count
	if compute_error:
		plt.figure()
		plt.plot(np.arange(num_iters), errors, label="Error between iterate and true solution")
		# plt.xscale("log", base=10)
		plt.yscale("log", base=10)
		plt.xlabel("Iteration")
		plt.ylabel("2-norm of error between iterate and true solution (log-scale)")
		plt.title(error_title)
		plt.show()

	# Make a plot of the function evaluations at each iterate as a 
	# function of the iteration count
	plt.figure()
	plt.plot(np.arange(num_iters), func_evals)
	plt.xlabel("Iteration")
	plt.ylabel("Function value at each iterate")
	plt.title(eval_title)
	plt.show()

	return xn


def main():
	# Problem 1, part (b)
	# Apply the gradientCheck function to pairs of functions with known gradients
	f = lambda x: np.exp(x)
	grad_f = lambda x: np.exp(x)
	x0 = 0

	gradientCheck(f, grad_f, x0, "Error in approximating the gradient of e^(x) at x = 0")

	g = lambda x: np.sin(x)
	grad_g = lambda x: np.cos(x)
	x0_g = 1

	gradientCheck(g, grad_g, x0_g, "Error in approximating the gradient of sin(x) at x = 1")

	h = lambda x: np.cos(x)
	grad_h = lambda x: (-1)*np.sin(x)
	x0_h = 2

	gradientCheck(h, grad_h, x0_h, "Error in approximating the gradient of cos(x) at x = 2")

	# Apply the gradientCheck function to a large dimensional quadratic 
	# to make sure it scales
	# k = lambda x: np.exp(x[0]**2 + x[1]**2 + x[2]**2)
	# grad_k = lambda x: 2*np.exp(x[0]**2 + x[1]**2 + x[2]**2)*x
	# x0_k   = np.array([1, 2, 3])

	k = lambda x: (1/2)*np.vdot(x, x)
	grad_k = lambda x: x
	x0_k = np.ones(50, )

	gradientCheck(k, grad_k, x0_k, "Error in approximating the gradient of a large dimensional function")

	# Test if the gradientCheck function can reliably determine if an incorrect gradient
	# has been passed in
	j = lambda x: x*x
	grad_j = lambda x: 2.5*x
	x0_j = 3

	gradientCheck(j, grad_j, x0_j, "Error in approximating the gradient of a function with a slightly incorrect gradient")

	# Problem 2:
	# Apply the gradient descent function to the quadratic
	# f(x) = (1/2)*||Ax - b||^(2), where A is over-determined
	# and has full column rank
	m = 10
	n = 5
	A = np.random.rand(m, n)
	b = np.random.rand(m)

	print("Does A have full column rank? ", np.linalg.matrix_rank(A) == n)

	f = lambda x: (1/2) * np.linalg.norm(A @ x - b, 2)**2
	grad_f = lambda x: A.T @ (A @ x - b)

	# Compute the least-squares solution to the above problem
	x_star, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

	# Use the zero vectors as our initial guess 
	x0 = np.zeros((n, ))

	# Compute an approximate solution using our gradient descent function
	stepsize = 1/(np.linalg.norm(A, ord=2)**2)
	error_title = "Error between gradient descent iterates and true solution to (1/2) ||Ax - b||^(2)"
	xn = gradientDescent(f, grad_f, x0, x_star, error_title, max_iters=200, stepsize=stepsize, 
		                 tol=1e-7, compute_error=True)

	# Apply the gradient descent function to the same quadratic as above, but 
	# now use a line search
	error_title_LS = "Error between gradient descent (LS) iterates and true solution to (1/2) ||Ax - b||^(2)"
	eval_title     = "Evaluation of (1/2) ||Ax - b||^(2) at each iterate"
	xn_LS = gradientDescentLS(f, grad_f, x0, x_star, error_title_LS, eval_title, max_iters=200, stepsize=stepsize,
		                      tol=1e-7, compute_error=True)

	# Problem 3
	# Compute the 2-norm of the difference between the approximate solutions 
	# computed via gradient descent and gradient descent with line search
	print("2-norm of difference between GD and GD with linesearch solutions: ",
		  np.linalg.norm(xn - xn_LS, 2))

	# Problem 4
	# Load the spam data
	spamData = sio.loadmat("spamData")

	# Pre-process the training and test data
	Xtrain = np.log(spamData["Xtrain"] + 0.1)
	Xtest  = np.log(spamData["Xtest"]  + 0.1)

	print(Xtrain[0, :])
	print(Xtest[0, :])

	# Load the training and test labels, and change any 0's to -1's
	ytrain = np.array(np.reshape(spamData["ytrain"], newshape=(spamData["ytrain"].size, )), dtype=np.int32)
	ytrain[ytrain == 0] = -1 

	ytest  = np.array(np.reshape(spamData["ytest"], newshape=(spamData["ytest"].size, )),   dtype=np.int32)
	ytest[ytest   == 0] = -1

	def lr(w):
		return np.sum(np.log(1 + np.exp(-ytrain * np.matmul(Xtrain, w))))

	def grad_lr(w):
		mu = 1/(1 + np.exp(-ytrain * np.matmul(Xtrain, w)))
		return (np.matmul(-Xtrain.T, ytrain*(1 - mu)))

	sigmoid = lambda a: np.exp(a)/(1 + np.exp(a))

	# Problem 4, part (a)
	# Time the fast implementation of the gradient of the negative log-likelihood function
	# and make sure it takes less than 1 second
	w = np.random.rand(Xtrain.shape[1])
	print("Time taken to evaluate the fast implementation of the gradient of loss function: ",
		  timeit.timeit('"grad_lr(w)"', number=1000000))

	# Problem 4, part (b)
	# Run the gradient descent solver with line search on the logistic function
	# on the training data to obtain a classified w
	w0 = (1/1000)*np.ones(Xtrain.shape[1])
	eval_title_lr = "Evaluation of logistic-loss function trained on spam data at each iterate"
	w_GDLS = gradientDescentLS(lr, grad_lr, w0, w0, "", eval_title_lr, max_iters=5000, stepsize=1,
		                       tol=1e-7, compute_error=False)

	print(lr(w_GDLS))

	# Perform classification on the training and testing data
	training_computed_labels = np.array([sigmoid(np.vdot(w_GDLS, Xtrain[i, :])) for i in range(Xtrain.shape[0])])
	for i in np.arange(training_computed_labels.size):
		if training_computed_labels[i] > 0.5:
			training_computed_labels[i] = 1;
		else:
			training_computed_labels[i] = -1;

	testing_computed_labels  = np.array([sigmoid(np.vdot(w_GDLS, Xtest[i, :])) for i in range(Xtest.shape[0])])
	for i in np.arange(testing_computed_labels.size):
		if testing_computed_labels[i] > 0.5:
			testing_computed_labels[i] = 1;
		else:
			testing_computed_labels[i] = -1;

	print("Training data misclassification rate: ", 1 - (np.sum(ytrain == training_computed_labels)/ ytrain.size))
	print("Testing data misclassification rate: ",  1 - (np.sum(ytest  == testing_computed_labels) / ytest.size))

	# Problem 1
	# Run the gradient check on the logistic function/gradient pair,
	# but use random data instead of the spam data
	X_random = np.random.rand(m, n)
	y_random  = np.array([1, 1, -1, -1, 1, 1, 1, -1, 1, -1])

	def lr_random(w):
		return np.sum(np.log(1 + np.exp(-y_random * np.matmul(X_random, w))))

	def grad_lr_random(w):
		mu = 1/(1 + np.exp(-y_random * np.matmul(X_random, w)))
		return (np.matmul(-X_random.T, y_random*(1 - mu)))

	w0_grad_check = (1/1000)*np.ones((n, ))
	gradientCheck(lr_random, grad_lr_random, w0_grad_check, "Error in approximating the gradient of logistic-loss")

main()