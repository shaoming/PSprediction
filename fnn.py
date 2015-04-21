#!/bin/bash/python
import numpy, math
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import *
from pybrain.datasets.supervised import *
from pybrain.supervised.trainers import BackpropTrainer
class myBackpropTrainer(BackpropTrainer):
    def setPen(self, pen):
		self.pen = pen
    def _calcDerivs(self, seq):
        """Calculate error function and backpropagate output errors to yield 
        the gradient."""
        self.module.reset()        
        for sample in seq:
            self.module.activate(sample[0])
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            # need to make a distinction here between datasets containing
            # importance, and others
            target = sample[1]
            outerr = target - self.module.outputbuffer[offset]
	    if outerr > 0:
		outerr = outerr 
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                self.module.backActivate(outerr * importance)                
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                # FIXME: the next line keeps arac from producing NaNs. I don't
                # know why that is, but somehow the __str__ method of the 
                # ndarray class fixes something,
                str(outerr)
                self.module.backActivate(outerr)
  
        return error, ponderation 
class fnn:
	def __init__(self,input_data,n_ahead,n_input,pen=3):
		inputs = input_data[:]
		#from pybrain.structure import FeedForwardNetwork
		#self.net = FeedForwardNetwork()
		#from pybrain.structure import LinearLayer, SigmoidLayer
		#inLayer = LinearLayer(n_input)
		#hiddenLayer1 = SigmoidLayer(8)
		#hiddenLayer2 = SigmoidLayer(4)	
		#outLayer = SigmoidLayer(1)
		#self.net.addInputModule(inLayer)
		#self.net.addModule(hiddenLayer1)
		#self.net.addModule(hiddenLayer2)
		#self.net.addOutputModule(outLayer)	
		#from pybrain.structure import FullConnection
		#in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
		#hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2)
		#hidden2_to_out = FullConnection(hiddenLayer2, outLayer)
		#self.net.addConnection(in_to_hidden1)
		#self.net.addConnection(hidden1_to_hidden2)
		#self.net.addConnection(hidden2_to_out)
		#self.net.sortModules()
		self.net = buildNetwork(n_input,
                   80, # number of hidden units
                   1,
                   #bias = True,
                   #hiddenclass = TanhLayer,
                   hiddenclass = SigmoidLayer,
                   #outclass = Si,
                   outclass = SigmoidLayer,
		   recurrent=True
                   )
		for i in range(len(inputs)):
			inputs[i] = float(inputs[i])/1024.0
		ds = SupervisedDataSet(n_input, 1)
		for i in range(len(inputs)-n_ahead-n_input-1):
			ind = numpy.array(inputs[i:i+n_input])
			ds.addSample((ind), float(max(inputs[n_input+i+1:1+n_input+i+n_ahead])))
		self.trainer = myBackpropTrainer(self.net, ds, learningrate = 0.01, momentum=0.05, verbose = True)
		self.trainer.setPen(pen)
		print "training"
		self.trainer.trainUntilConvergence(maxEpochs = 40)
		#self.trainer.testOnData(verbose=True)
	def predict(self,input_data):
		inputs = input_data[:]
		for i in range(len(inputs)):
                        inputs[i] = float(inputs[i])/1024.0
		ind = numpy.array(inputs[:])		
		rel=int(self.net.activate(ind)*1024.0)
		print rel
		if rel > 1024:
			return 1024
		if rel < 1:
			return 1
		return rel

if __name__=="__main__":
	inputs = [i*2 for i in range(500)]
	inputs1 = [i*2+1 for i in range(500)]
	inputs += inputs1
	fnn11=fnn(inputs,1,2,10)
	print "test"
	print fnn11.predict([20,22])
	print fnn11.predict([798,800])
	

		
