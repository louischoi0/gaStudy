from matplotlib import pyplot as plt
import pandas as pd
from operator import add
from functools import reduce
from itertools import permutations
import numpy as np
from random import randint

data = [
    [ 0,5,6.4,50,11.4 ],
    [ 5,0,4,54.08,9.22 ],
    [ 6.4,4,0,51.97,13.15 ],
    [ 50,54,51.97,0,61.07 ],
    [ 11.4,9.22,13.14,61.07,0 ]]

df = pd.DataFrame(data,index=["A","B","C","D","E"],columns=["A","B","C","D","E"])

def getDistance(start,end) :
    return df.loc[start,end]

locations = [ "A", "B", "C", "D", "E"]

def getAverageDistanceByReduce(locationList) :

    locationList += [-1]

    def subf(a,b) :

        def subavg(distance,count) :
            return distance / count

        if b[1] == -1 :
            return ( 0, a[1] / a[2] )

        nodeCount = a[2]  + 1
        distanceSum = a[1]
        distanceSum += getDistance(a[0],b)

        avg = subavg(distanceSum,nodeCount)

        return ( b, distanceSum , nodeCount , avg )

    return applyByReduce(locationList,subf)

def applyByReduce(locationList,func) :
    return reduce(func,locationList[1:], ( locationList[0], 0 ) )[1]

def getTotalDistanceByReduce(locationList) :

    def subf(a,b) :
        distance = getDistance(a[0],b)
        return ( b, a[1] + distance )

    return applyByReduce(locationList,subf)

def f(locationList) :
    distance = getTotalDistanceByReduce(locationList)

    if distance <= 0 :
        return 0.000000001

    score = 1 / distance

    duplicated = len(locationList) - len(list(set(locationList))) 
    duplicated = duplicated if duplicated != 0 else 1

    return score / (duplicated**2)

def f2(locationList) :
    return 1 / f(locationList)

distanceSum = getTotalDistanceByReduce(["B","A","C","D","E","D","C","A","B"])
print(distanceSum)

def All(v,f) :

    def sub(a,b) :
        return a * int(f(b))

    return reduce(sub,v,1) == 1

def Any(v,f) :

    def sub(a,b) :
        return a + int(f(b))

    return reduce(sub,v,0) >= 1

def moreThan(v,f,k) :

    def sub(a,b) :
        return a + int(f(b))

    return reduce(sub,v,0) >= k

global autoId
autoId = 0

class Instance :
    def __init__(self,values) :
        self.gene = values
        self.score = 0

        global autoId
        self.id = autoId

        autoId += 1
        print("{} Instance Created".format(autoId))

    def evaluate(self,obj) :
        print(self.gene,end="")
        print(" => {}".format(obj(self.gene)))
        self.score = obj(self.gene)
        return self.score

    def equal(self,other) :
        return self == other

    def show(self) :
        print(self.gene,end="")

class Enviroment :
    def __init__(self) :
        self.instancies = list(Instance(list(x)) for x in permutations(["B","C","D","E"]))
        self.scores = np.array([0 for x in self.instancies])
        self.history = []
    def topN(self,n) :
        return list( ( self.instancies[x] for x in np.argsort(self.scores,)[::-1][:n] ) )

    def updateScore(self,func) :
        self.scores = list(map(lambda x : x.evaluate(func) , self.instancies))

        for i in self.instancies :
            print(i.gene,end=" : ")
            print(i.score)

    def make(self) :
        total = sum(self.scores)

        def toProbability(x) :
            return x / total

        return list(map( toProbability , self.scores))

    def pickByNumber(self,n,func=f) :

        """
            Output List is supposed to be distinct/
        """

        import numpy as np

        r = [0] + self.make()

        sets = np.cumsum(r)

        print(sets)
        vtuples = list(map(lambda x : (sets[x] , sets[x+1]) , range(0,len(r)-1)))

        from random import randint 

        def pick() :
            index = 0
            rvalue = randint(1,100000000) / 100000000
            for t in vtuples :

                if rvalue > t[0] and rvalue <= t[1] :
                    return index

                index += 1

            return -1

        results = []

        def pickRecursively(beforeResult=[]) :
            p = n - len(beforeResult)

            r = list(set(list( pick() for _ in range(p) )))
            r = list(set(r + beforeResult))

            if n - len(r) == 0 :
                return r 

            return pickRecursively(r)

        return pickRecursively()

    def getDead(self,n) :
        return list( ( self.instancies[x] for x in self.pickByNumber(n,f2) ) )

    def cross_over(self,n) :
        goodGenes = self.pickByNumber(n*2)
        goodGenes = list(self.instancies[x].gene for x in goodGenes)

        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))

        genePairs = list(chunker(goodGenes,2))
        # [ (g0,g1) , (g2,g3), (g4,g5) ]
        from random import randint 

        def _cross_over(g1,g2) :
            maxdp = len(g1)
            dp = randint(1,maxdp)

            return Instance(g1[:dp] + g2[dp:])

        print(genePairs[0])

        return list(map(lambda genep : _cross_over(*genep),genePairs))

    def mutate(self,n) :
        t = len(self.instancies)
        r = list( randint(0,t-1) for _ in range(n) )

        def _mutate(inst) :
            idx = randint(0,3)
            p = randint(0,4)

            v = ["A","B","C","D","E"][p]

            inst.gene[idx] = v
            return inst

        for i in r :
            before = self.instancies[i]
            self.instancies[i] = _mutate(self.instancies[i])


    def timeGoesBy(self,n) :
        self.updateScore(f)
        insts = list(n.id for n in self.instancies )
        print(insts)

        newGenes = self.cross_over(n)

        self.instancies.extend(newGenes)

        badGenes = self.pickByNumber(n,f2)
        _b = list( self.instancies[x].id for x in range(len(self.instancies)) if x in badGenes )
        # [ 0, 1 , 4 , 2]

        self.instancies = list(self.instancies[x] for x in range(len(self.instancies)) if not x in badGenes)
        self.check(newGenes,_b)

        self.mutate(3)
        
        scores = list(x.evaluate(f) for x in self.instancies )
        scoreSum = sum( scores )
        scoreSumTop5 = sum(sorted(scores,key= lambda x : -x)[:5])

        self.history.append(scoreSumTop5)

    def check(self,newgenes,badgenes) :
        print(self.instancies)
        insts = np.array(list(n.id for n in self.instancies ))

        print(newgenes)

        n = np.array(list( x.id for x in newgenes))
        b = np.array(badgenes)

        f0 = lambda x : x in insts
        assert All(n,f0) 

        f1 = lambda x : not x in insts
        assert All(b,f1)

        f2 = lambda x : x in n

        assert moreThan(insts,f2,5)

        print(insts)
        print(n)
        print(badgenes)

from sys import exit

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

e = Enviroment()

for _ in range(300) :
    e.timeGoesBy(5)

print(e.history)
plt.plot(e.history)
plt.show()

print("ok")