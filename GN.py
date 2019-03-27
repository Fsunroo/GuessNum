import numpy as np 
#X--> input , Y-->Out put
X=np.array(([2,4,6,6],[3,6,4,7]),dtype=float)
Y=np.array(([41],[52]),dtype=float)
X=X/np.amax(X,axis=0)
Y=Y/1000
Q=np.array(([4,7,6,7]),dtype=float)
Q=Q/np.amax(Q,axis=0)
class scorecal(object):
	"""docstring for scorecal"""
	def __init__(self):
		#declearing sizes
		self.inputSize=4
		self.hiddenSize_1=5
		self.hiddenSize_2=5
		self.outputSize=1
		#declearing random parameters
		self.w1=np.random.randn(self.inputSize, self.hiddenSize_1) # an 2*3 array   3*1 . 2*3 = 3*3
		self.w2=np.random.randn(self.hiddenSize_1, self.hiddenSize_2) # an 3*1 array    3*3 . 3.1 = 3.1
		self.w3=np.random.randn(self.hiddenSize_2, self.outputSize)
	def sigmoid(self,s):
		return 1/(1+np.exp(-s))
	def Forward(self,X):                                  #HERE I AM
		self.z=np.dot(X, self.w1)
		self.z2=self.sigmoid(self.z)
		self.z3=np.dot(self.z2, self.w2)
		self.z4=self.sigmoid(self.z3)
		self.z5=np.dot(self.z4, self.w3)
		o=self.sigmoid(self.z5)
		return o
	def outputa(self,a):
		return a*(1-a)
	def Backward(self,X,Y,o):
		"""ajustmant = eror*input*output*(1-output)"""
		self.eror_o=Y-o
		self.delta_o= self.eror_o*self.outputa(o)
		self.w3+=self.z4.T.dot(self.delta_o)

		self.eror_z4=self.delta_o.dot(self.w3.T)
		self.delta_z4=self.eror_z4*self.outputa(self.z4)
		self.w2+=self.z2.T.dot(self.delta_z4)

		self.eror_z2=self.delta_z4.dot(self.w2.T)
		self.delta_z2=self.eror_z2*self.outputa(self.z2)
		self.w1+=X.T.dot(self.delta_z2)
	def train(self,X,Y):
		o=self.Forward(X)
		self.Backward(X,Y,o)
SC=scorecal()
for i in range(100000):
	SC.train(X,Y)
#print Y
#print Q
print "result is \n" +str(SC.Forward(X))
print "result is \n" +str(SC.Forward(Q))
		