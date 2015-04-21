#!bin/usr/python
from fnn import *
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
	def __init__(self,dataset,ninput):
		#self.name = dataset[0]
		from pybrain.structure import FeedForwardNetwork
                self.net = FeedForwardNetwork()
                from pybrain.structure import LinearLayer, SigmoidLayer
                inLayer = LinearLayer(ninput)
                hiddenLayer1 = SigmoidLayer(12)
                hiddenLayer2 = SigmoidLayer(8)
                outLayer = SigmoidLayer(1)
                self.net.addInputModule(inLayer)
                self.net.addModule(hiddenLayer1)
                self.net.addModule(hiddenLayer2)
                self.net.addOutputModule(outLayer)
                from pybrain.structure import FullConnection
                in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
                hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2)
                hidden2_to_out = FullConnection(hiddenLayer2, outLayer)
                self.net.addConnection(in_to_hidden1)
                self.net.addConnection(hidden1_to_hidden2)
                self.net.addConnection(hidden2_to_out)
                self.net.sortModules()

		#self.net = buildNetwork(ninput,
                #  20, # number of hidden units
                #   1,
                   #bias = True,
                #   hiddenclass = SigmoidLayer,
                #   outclass = SigmoidLayer,
		#   recurrent=True
                #)
		ds = SupervisedDataSet(ninput, 1)
		for i in range(len(dataset[4])):
			ind = numpy.array(dataset[4][i])
			ds.addSample((ind), dataset[2][i]/2.0)
		self.trainer = BackpropTrainer(self.net, ds, learningrate = 0.05, momentum=0.4, verbose = True)
		self.trainer.trainUntilConvergence(maxEpochs = 400)
	def predict(self,data):
                ind = numpy.array(data[:])
                return (self.net.activate(ind)*2.0)[0]
	def predictN(self,dataset):
		dataset[3]=[0.0 for i in range(len(dataset[4]))]
		for i in range(len(dataset[4])):
			dataset[3][i] = self.predict(dataset[4][i])
		for i in range(len(dataset[2])):
			print dataset[1][i]+"\t"+str(dataset[2][i])+"\t"+str(dataset[3][i])
			#pass
		
def parseInput(filename,name1,name2):
	inputs = open(filename)
	dataset1 = [[name1],[],[],[],[]]
	dataset2 = [[name2],[],[],[],[]]
	for line in inputs.readlines():
		datas = line.split("\t")
		dataset1[1].append(datas[0])
		dataset2[1].append(datas[0])
		dataset1[2].append(float(datas[1]))
		dataset2[2].append(float(datas[2]))
		dataset=[]
		for data in datas[4:]:
			dataset.append(float(data)/(float(datas[3])*2.0))	
		dataset1[4].append(dataset)
		dataset2[4].append(dataset)
	return dataset1,dataset2,len(dataset)
if __name__ == "__main__":
	dataset1, dataset2,num = parseInput("b4G.txt","4Gto4G","4Gto24G")
	B4g4G4 = fnn(dataset1,num)
	B4g4G4.predictN(dataset1)
	#B4g4G24 = fnn(dataset2,num)
	#B4g4G24.predictN(dataset2)
