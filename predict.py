#!bin/usr/python
import sys
sys.path.append("../../")
from parse import *
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
                outLayer = SigmoidLayer(2)
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
		ds = SupervisedDataSet(ninput, 2)
		for i in range(len(dataset[6])):
			ind = numpy.array(dataset[6][i])
			ind_out=numpy.array([dataset[2][i]/2.0,dataset[3][i]/2.0])
			ds.addSample((ind), (ind_out))
		self.trainer = BackpropTrainer(self.net, ds, learningrate = 0.05, momentum=0.4, verbose = True)
		self.trainer.trainUntilConvergence(maxEpochs = 400)
	def predict(self,data):
                ind = numpy.array(data[:])
                return (self.net.activate(ind)*2.0)[0],(self.net.activate(ind)*2.0)[1]
	def predictN(self,dataset):
		dataset[4]=[0.0 for i in range(len(dataset[6]))]
		dataset[5]=[0.0 for i in range(len(dataset[6]))]
		for i in range(len(dataset[4])):
			dataset[4][i],dataset[5][i] = self.predict(dataset[6][i])
		for i in range(len(dataset[2])):
			print dataset[1][i]+"\t"+str(dataset[2][i])+"\t"+str(dataset[3][i])+"\t"+str(dataset[4][i])+"\t"+str(dataset[5][i])
			#pass
		
def parseInput(filename,name):
	inputs = open(filename)		       ## 0	  1		2	          3		    4			5			6
	dataset = [[name],[],[],[],[],[],[]] ## name,benchmark name ,speedup, otherway speedup,predicted speedup,predicted otherwway speedup,hardware counters
	for line in inputs.readlines():
		datas = line.split("\t")
		dataset[1].append(datas[0])
		dataset[2].append(float(datas[1]))
		dataset[3].append(float(datas[2]))
		dat=[]
		for data in datas[4:]:
			dat.append(float(data)/(float(datas[3])*2.0))	
		dataset[6].append(dat)
	return dataset,len(dat)
class pred_parse:
	def __init__(self):
		self.entries=[]
		#fetch_icache = cumRecords(".fetch.icacheStallCycles","\s+(\d+)",["testsys.switch_cpus"+str(i) for i in range(4)])
		self.entries.append(cumRecords("sim_ticks","\s+(\d+)",[""]))
		self.entries.append(cumRecords(".inst","\s+(\d+)",["system.physmem.num_reads::switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".data","\s+(\d+)",["system.physmem.num_reads::switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords("::total","\s+(\d+)",["system.physmem.num_reads"]))

		self.entries.append(cumRecords("","\s+(\d+)",["system.physmem.queueLat::"+str(i) for i in range(4)]))
		self.entries.append(cumRecords("","\s+(\d+)",["system.physmem.DRAMLat::"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".totBankLat","\s+(\d+)",["system.physmem"]))
                self.entries.append(cumRecords(".totQLat","\s+(\d+)",["system.physmem"]))
        	self.entries.append(srecord(".readRowHitRate","\s+(\d+.\d+)",["system.physmem"]))
        	self.entries.append(cumRecords(".totMemAccLat","\s+(\d+)",["system.physmem"]))

		self.entries.append(cumRecords(".inst","\s+(\d+)",["system.l2.ReadReq_misses::switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".data","\s+(\d+)",["system.l2.ReadReq_misses::switch_cpus"+str(i) for i in range(4)]))
        	self.entries.append(cumRecords("::total","\s+(\d+)",["system.l2.ReadReq_misses"]))

		self.entries.append(cumRecords(".commit.committedInsts","\s+(\d+)",["system.switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".commit.loads","\s+(\d+)",["system.switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".commit.fp_insts","\s+(\d+)",["system.switch_cpus"+str(i) for i in range(4)]))

        	self.entries.append(cumRecords("::total","\s+(\d+)",["system.l2.Writeback_hits"]))
		self.entries.append(cumRecords(".data","\s+(\d+)",["system.cpu"+str(i)+".dcache.ReadReq_hits::switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".data","\s+(\d+)",["system.cpu"+str(i)+".dcache.WriteReq_hits::switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".data","\s+(\d+)",["system.cpu"+str(i)+".dcache.ReadReq_misses::switch_cpus"+str(i) for i in range(4)]))
		self.entries.append(cumRecords(".data","\s+(\d+)",["system.cpu"+str(i)+".dcache.WriteReq_misses::switch_cpus"+str(i) for i in range(4)]))

		self.entries.append(cumRecords("rename.ROBFullEvents","\s+(\d+)",["system.switch_cpus"+str(i) for i in range(4)]))
	def matchline(self,lines):
		for line in lines:
			for entry in self.entries:
				entry.matchLine(line)
	def GetEvents(self):
		for entry in self.entries:
			entry.update()
		datas = [[] for i in range(4)]
		time= self.entries[0].getValue(0)+0.00000000001
		for i in range(1,len(self.entries)):
			for j in range(4):
				datas[j].append(self.entries[i].getValue(j)/(2.0*time))
		return datas
				
class predictions:
	def __init__ (self,path):
		self.parse=pred_parse()
		dataset,num = parseInput(path+"b24G.txt","24G")
		self.pd24G=fnn(dataset,num)
		dataset,num = parseInput(path+"b4G.txt","4G")
		self.pd4G=fnn(dataset,num)
	def GetEvents(self,handle):
		datas=self.parse.GetEvents()
		rel_sets = [[] for i in range(4)]
		for i in range(4):
			rel_sets[i]=handle(datas[i])
		return rel_sets
	def predict(self,lines,mode):
		self.parse.matchline(lines)
		if mode == "24G":
			return self.GetEvents(self.pd24G.predict)
		if mode == "4G":
			return self.GetEvents(self.pd4G.predict)
		print "error"
		sys.exit(1)
if __name__ == "__main__":
	pred = predictions("/home/shaoming/script/parse/prediction/")	
	mheader =  header()
	file = open("testlmll_4G.txt")
	lines = []
	nums = [[]for i in range(8)]
	n= 0.0
	for line in file.readlines():	
		lines.append(line)
		if mheader.matchLine(line):
			n += 1.0
			pairs=pred.predict(lines,"4G")
			for i in range(4):
				for j in range(2):
					nums[i+j*4].append(pairs[i][j])
			lines =[]
	results = [[]for i in range(8)]
	for i in range(8):
		results[i]=sum(nums[i])/n
	print results
	#dataset,num = parseInput("b4G.txt","4G")
	#B4g4G4 = fnn(dataset,num)
	#B4g4G4.predictN(dataset)
	#dataset,num = parseInput("b24G.txt","24G")
	#B4g4G4 = fnn(dataset,num)
	#B4g4G4.predictN(dataset)
	#B4g4G24 = fnn(dataset2,num)
	#B4g4G24.predictN(dataset2)
