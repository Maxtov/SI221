import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random, math
import tiilab
from sklearn.datasets import make_blobs

def generate_data(sigma):
	X,y = make_blobs(n_samples=200, centers=[[-1,0],[1,0]], n_features=2, cluster_std=sigma)
	return add_bias(X),y

def add_bias(X):
	newX = []
	for x in X:
		newX.append((x[0],x[1],1))
	X=np.array(newX)
	return X

def predict(x, w):
	if(np.dot(w,x) > 0):
		return 1
	else:
		return 0

def train(data, y,eta, T,w):
	for i in range(T):
		for i in range(len(data)):
			prediction = predict(data[i], w)
			if(prediction != y[i]):
				if(prediction):
					w = w - eta * data[i]
				else:
					w = w + eta * data[i]
	return w

def do_N_perceptron(N,eta,T,sigmas):
	erreurs = np.zeros((4,T))
	e=np.zeros(4)
	s = np.zeros(4)
	for sigma in range(len(sigmas)):
		w = np.zeros(3)
		for n in range(N):
			X,y = generate_data(sigmas[sigma])
			#X = add_bias(X_no_bias)
			w = train(X,y,eta,T,w)
			erreurs[sigma][n] = calcul_erreur(X,y,w)
		e[sigma] = np.array(erreurs[sigma]).mean()
		s[sigma] = calcul_deviation(erreurs[sigma],e[sigma],N)
	return w, e,s

def calcul_erreur(data,y,w):
	err = 0
	for i in range(len(data)):
		if(predict(data[i],w)!=y[i]):
			err += 1
	return err

def calcul_deviation(erreurs,e,N):
	sum=0
	for i in range(N):
		sum += (erreurs[i]-e)**2
	s = np.sqrt(sum/50)
	return s

def question1_1():
	T = 200
	sigmas = [0.05,0.25,0.5,0.75]
	w, err, s = do_N_perceptron(50,0.1,T,sigmas)
	print(w)
	print("taux d'erreurs et deviation : ",err)
	print("deviation : ",s)

	r = range(len(err))
	plt.title("Erreur et Deviation en fonction de sigma")
	plt.bar(r, err, width = 0.4, color = 'red',
	           edgecolor = 'black', linewidth = 1, label='erreur')
	plt.bar([x + 0.4 for x in r], s, width = 0.4, color = 'blue',
	           edgecolor = 'black', linewidth = 1, label='deviation')
	plt.legend()
	plt.xticks([r1 + 0.4 / 2 for r1 in r], ['0.05', '0.25', '0.5', '0.75'])
	plt.show()

def do_N_perceptron_with_p(N,eta,T,p):
	erreurs = np.zeros((4,N))
	e=np.zeros(4)
	s = np.zeros(4)
	for k in range(len(p)):
		w = np.zeros(3)
		for n in range(N):
			X,y = generate_data(0.15)
			for i in range(len(X)):
				rd = random.random()
				if(rd<p[k]):
					y[i] = 1 - y[i]
			w = train(X,y,eta,T,w)
			erreurs[k][n] = calcul_erreur(X,y,w)
		e[k] = np.array(erreurs[k]).mean()
		s[k] = calcul_deviation(erreurs[k],e[k],N)
	return w, e,s

def question1_2():
	T = 200
	N=50
	p = [0,0.05,0.10,0.20]
	w, err, s = do_N_perceptron_with_p(N,0.1,T,p)
		
	print(w)
	print("taux d'erreurs et deviation : ",err)
	print("deviation : ",s)


	r = range(len(err))
	plt.title("Erreur et Deviation en fonction de sigma")
	plt.bar(r, err, width = 0.4, color = 'red',
	           edgecolor = 'black', linewidth = 1, label='erreur')
	plt.bar([x + 0.4 for x in r], s, width = 0.4, color = 'blue',
	           edgecolor = 'black', linewidth = 1, label='deviation')
	plt.legend()
	plt.xticks([r1 + 0.4 / 2 for r1 in r], ['0%', '5%', '10%', '20%'])
	plt.show()

def evaluate_pixels(image,value):
	nb_rows = len(image[0])
	nb_cols = len(image[0][0])
	labels=np.zeros((nb_rows,nb_cols))
	for row in range(nb_rows):
		for col in range(nb_cols):
			if(image[0][row][col] < value):
				labels[row][col] = 0
			else:
				labels[row][col] = 1
	return labels

def add_bias_image(image):
	newImg = []
	for row in image:
		newCol = []
		for pixel in row:
			newCol.append((pixel,1))
		newImg.append(newCol)
	img=np.array(newImg)
	return img

def error_correction(image, labels, eta, weights):
	erreurs = []
	for row in range(len(image)):
		weights = train(image[row],labels[row],eta,1,weights)
		erreurs.append(calcul_erreur(image[row],labels[row],weights))
	erreur = np.sum(erreurs)
	return weights,erreur



#img[0] contient les pixels
img=tiilab.imz2mat("data/landsattarasconC4.ima")
lab = evaluate_pixels(img,30)

dataset = add_bias_image(img[0])

print(dataset[:][0][:,0])

plt.scatter(lab[0],dataset[:][0][:,0])
plt.show()

err=-1
eta = 1
w = np.zeros(2)

epochs = 0
while err != 0:
	w,err = error_correction(dataset,lab,eta,w)
	epochs += 1
	print(err)

print(w)
print(epochs)

'''
plt.imshow(lab)
plt.show(block = False)
tiilab.visusar(img[0])
'''

''' Afficher un set de donnée et la droite à partir de w
X,y = make_blobs(n_samples=200, centers=[[-1,0],[1,0]], n_features=2, cluster_std=sigma)
plt.scatter(X[:,0],X[:,1],c=y)
abs = np.arange(-1,1,0.01)
f = -(w[0]/w[1]) * abs - (w[2]/w[1])
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.plot(abs,f)
plt.show()
'''




'''
for it in range(0,1):
	X=[]
	y=[]
	for i in range(0,200):
		#rd = random.randint(0,1)
		if(i<100):
			rd=0
			mu = (-1,0)
		else:
			rd=1
			mu = (1,0)
		X.append(np.random.multivariate_normal(mu,var))
		y.append(rd)

	for i in range(0,200):
		if(y[i]==0):
			if( y[i]*np.dot(w,X[i]) <= 0):
				print("<=")
				loss[it] += 1
				w = w + y[i]*X[i]
		elif(y[i]==1):
			if( y[i]*np.dot(w,X[i]) > 0):
				print(">")
				loss[it] += 1
				w = w - y[i]*X[i]


print(sum(loss))
print(w)




 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights
 
# Calculate weights
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

I = np.array([[1, 0], [0, 1]])
sigma = 0.95
var = sigma**2 * I
dataset = []
y=[]
for i in range(0,200):
		#rd = random.randint(0,1)
		if(i<=100):
			rd=0
			mu = (-1,0)
		else:
			rd=1
			mu = (1,0)
		x = np.random.multivariate_normal(mu,var)
		x = (x[0],x[1],rd)
		dataset.append(x)


l_rate = 0.1

n_epoch = 150

weights = train_weights(dataset, l_rate, n_epoch)
print(weights)
'''