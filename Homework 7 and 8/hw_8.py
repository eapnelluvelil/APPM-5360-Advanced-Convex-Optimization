import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import cvxpy as cvx

def gradientDescentLS(f, grad_f, x0, max_iters=100, stepsize=1, tol=1e-7):
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

		if (np.linalg.norm(xn - x0, 2) < tol):
			break

		x0 = xn

		# Increase the stepsize by a factor of two 
		# if we decrease the stepsize only once
		if armijo_cond_iters == 1:
			t *= 2

	func_evals = np.array(func_evals)
	return func_evals, xn

def NAG(f, grad_f, x0, max_iters=100, stepsize=1, tol=1e-7):
	func_evals = []
	# lambda_prev = 0
	x_prev = x0
	y_prev = x0
	y_curr = np.zeros(x0.shape)
	x_curr = np.zeros(x0.shape)
	
	num_iters = 0
	while (num_iters < max_iters):
		x_curr = y_curr - stepsize*grad_f(y_curr)
		y_curr = x_curr + (num_iters/(num_iters + 3))*(x_curr - x_prev)
		num_iters += 1
	
		func_evals.append(f(x_curr))
	
		if (np.linalg.norm(x_curr - x_prev, 2) < tol):
			break

		x_prev = x_curr
		y_prev = y_curr

	func_evals = np.array(func_evals)
	return func_evals, x_curr

def l1_norm_prox(l, t, y):
	return np.sign(y)*np.maximum(np.abs(y) - t*l, np.zeros(y.shape))

def prox_GD(f, grad_f, l, prox_g, x0, use_g=False, max_iters=100, stepsize=1, tol=1e-7):
	# If g is the zero function, then use Nesterov's accelerated gradient descent
	if not use_g:
		_, w_NAG = NAG(f, grad_f, x0, max_iters=max_iters,
			                        stepsize=stepsize, tol=tol)
		return w_NAG
	# If g is not the zero function, then use proximal gradient descent
	else:
		x_prev = x0
		y_prev = x0

		x_curr = np.zeros(x0.shape)
		y_curr = np.zeros(x0.shape)
		
		num_iters = 0
		while (num_iters < max_iters):
			x_curr = prox_g(l, stepsize, y_prev - stepsize*grad_f(y_prev))
			y_curr = x_curr + ((num_iters)/(num_iters + 3))*(x_curr - x_prev)

			num_iters += 1

			if (np.linalg.norm(x_curr - x_prev, 2) < tol):
				break

			x_prev = x_curr
			y_prev = y_curr

		return x_curr

def main():
	# Problem 1
	# Load the spam data
	spamData = sio.loadmat("spamData")
		
	# Pre-process and training and testing data
	Xtrain = np.log(spamData["Xtrain"] + 0.1)
	Xtest  = np.log(spamData["Xtest"] + 0.1)

	# Load the training and testing labels, and change any 0's to 1's
	ytrain = np.array(np.reshape(spamData["ytrain"], newshape=(spamData["ytrain"].size, )), dtype=np.int32)
	ytrain[ytrain == 0] = -1
		
	ytest  = np.array(np.reshape(spamData["ytest"],  newshape=(spamData["ytest"].size,  )), dtype=np.int32)
	ytest[ytest == 0] = -1

	Xtrain_norm = np.linalg.norm(Xtrain, 2)
	stepsize = 4/(Xtrain_norm**2)
	
	# Define the logistic loss function and its gradient
	def lr(w):
		return np.sum(np.log(1 + np.exp(-ytrain * np.matmul(Xtrain, w))))
	
	def grad_lr(w):
		mu = 1/(1 + np.exp(-ytrain * np.matmul(Xtrain, w)))
		return (np.matmul(-Xtrain.T, ytrain*(1 - mu)))
	
	sigmoid = lambda a: np.exp(a)/(1 + np.exp(a))

	# w0 = (1/1000)*np.ones(Xtrain.shape[1])
	w0 = np.random.rand(Xtrain.shape[1])/1000

	# Solve the logistic regression problem using gradient descent
	# with line search
	func_evals_GDLS, w_GDLS = gradientDescentLS(lr, grad_lr, w0, max_iters=15000, 
					                            stepsize=1, tol=1e-10) 

	# Solve the logistic regression problem using a variant of
	# Nesterov accelerated gradient descent
	func_evals_NAG, w_NAG = NAG(lr, grad_lr, w0, max_iters=15000, 
			                    stepsize=stepsize, tol=1e-10)
		
	# Solve the logistic regression problem using the Nelder-Mead method
	minimum_NM = scipy.optimize.fmin(lr, w0, xtol=1e-7, maxiter=60000, full_output=1,
				                     retall=1)
	# func_evals_NM = lr(np.array(minimum_NM[5]))
		
	# Plot the objective values on a semilogy plot
	plt.figure(1)
	plt.semilogy(np.arange(0, func_evals_GDLS.size), func_evals_GDLS)
	plt.semilogy(np.arange(0, func_evals_NAG.size), func_evals_NAG)
	# plt.semilogy(np.arange(0, func_evals_NM.size), func_evals_NM)

	# Problem 2
	# Re-run logistic regression on the spam data, but this time,
	# add a l1-penalty term
	l = 5
	# def lr_l1_penalty(w):
	# 	return (np.sum(np.log(1 + np.exp(-ytrain * np.matmul(Xtrain, w)))) + \
	# 	        l*np.linalg.norm(w, 1))
	
	w_PGD = prox_GD(lr, grad_lr, l, l1_norm_prox, w0, use_g=True, max_iters=15000, 
		            stepsize=stepsize, tol=1e-10)
	
	# Print out the training and testing data classication accuracies corresponding
	# to the non-regularized and regularized weights

	# Perform classification on the training and testing data with 
	# non-regularized weights
	training_computed_labels_nr = sigmoid(Xtrain @ w_NAG)
	training_computed_labels_nr[training_computed_labels_nr > 0.5] = 1
	training_computed_labels_nr[training_computed_labels_nr <= 0.5] = -1

	testing_computed_labels_nr  = sigmoid(Xtest @ w_NAG)
	testing_computed_labels_nr[testing_computed_labels_nr > 0.5] = 1
	testing_computed_labels_nr[testing_computed_labels_nr <= 0.5] = -1

	# Perform classification on the training and testing data with
	# regularized weights
	training_computed_labels_r = sigmoid(Xtrain @ w_PGD)
	training_computed_labels_r[training_computed_labels_r > 0.5] = 1
	training_computed_labels_r[training_computed_labels_r <= 0.5] = -1

	testing_computed_labels_r   = sigmoid(Xtest @ w_PGD)
	testing_computed_labels_r[testing_computed_labels_r > 0.5] = 1
	testing_computed_labels_r[testing_computed_labels_r <= 0.5] = -1

	print("Training data misclassification rate (non-regularized): ", 
           (1 - (np.sum(ytrain == training_computed_labels_nr)/ ytrain.size)))
	print("Testing data misclassification rate (non-regularized):  ",
           (1 - (np.sum(ytest  == testing_computed_labels_nr) / ytest.size)))
	
	print("Training data misclassification rate (regularized):     ", 
           (1 - (np.sum(ytrain == training_computed_labels_r)/ ytrain.size)))
	print("Testing data misclassification rate (regularized):      ",
           (1 - (np.sum(ytest  == testing_computed_labels_r) / ytest.size)))
	
	plt.figure(2)
	plt.plot(np.arange(0, w_NAG.size), w_NAG)
	plt.plot(np.arange(0, w_PGD.size), w_PGD)

	# Problem 3
	Y = pickle.load(open("SheppLogan_150x150.pkl", "rb"))
	
	# Select roughly 10% of the pixel values in the Shepp Logan phantom to add
	# random noise to
	num_random_rows_cols = int(Y.size * 0.10)
	random_rows_Y = np.random.choice(Y.shape[0], num_random_rows_cols, replace=True)
	random_cols_Y = np.random.choice(Y.shape[1], num_random_rows_cols, replace=True)

	Y_noisy = np.array(Y)
	Y_noisy[random_rows_Y, random_cols_Y] += np.random.uniform(low=0.0, high=1.0, 
							                                   size=(num_random_rows_cols, ))
	
	plt.imshow(Y)
	plt.imshow(Y_noisy)

	# Form the discrete gradient operator
	n1 = Y.shape[0]
	n2 = Y.shape[1]

	D_n1 = scipy.sparse.spdiags((-1)*np.ones(n1), 0, m=n1, n=n1) + \
	       scipy.sparse.spdiags(np.ones(n1 - 1), 1, m=n1, n=n1)
	D_n2 = scipy.sparse.spdiags((-1)*np.ones(n2), 0, m=n2, n=n2) + \
	       scipy.sparse.spdiags(np.ones(n2 - 1), 1, m=n2, n=n2)
	
	I_n1 = scipy.sparse.spdiags(np.ones(n1), 0)
	I_n2 = scipy.sparse.spdiags(np.ones(n2), 0)
	
	L_h_tilde = scipy.sparse.kron(D_n2, I_n1)
	L_v_tilde = scipy.sparse.kron(I_n2, D_n1)

	L = np.concatenate((L_h_tilde, L_v_tilde))

	def phi(y_1, y_2):
		return np.sqrt(y_1**2 + y_2**2)

	def g(y):
		result = 0
		for i in np.arange(y.shape[0]):
			result += phi(y[i, 0], y[i, 1])
		return result
	
	def TV(X):
		return g(L*np.concatenate((np.flatten(X, "F"), np.flatten(X, "F"))))

	# Solve the TV de-noising problem via CVXPY
	tau = (1/4)*TV(Y_noisy)
	print(tau)
	
	# plt.show()

main()