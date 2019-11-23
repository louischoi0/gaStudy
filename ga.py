import pandas as pd
from operator import add
from functools import reduce
from itertools import permutations

data = [
    [ 0,5,6.4,50,11.4 ],
    [ 5,0,4,54.08,9.22 ],
    [ 6.4,4,0,51.97,13.15 ],
    [ 50,54,51.97,0,61.07 ],
    [ 11.4,9.22,13.14,61.07,0 ]]

df = pd.DataFrame(data,index=["A","B","C","D","E"],columns=["A","B","C","D","E"])

def getDistance(start,end) :
    #print("From {} to {} Distance : {}".format(start,end,df.loc[start,end]))
    return df.loc[start,end]

def getTotalDistance(locationList) :
    s = 0
    s += getDistance("A",locationList[0])
    s += getDistance(locationList[0],locationList[1])
    s += getDistance(locationList[1],locationList[2])
    s += getDistance(locationList[2],locationList[3])
    return s

def getTotalDistanceByReduce(locationList) :

    def subf(a,b) :
        distance = getDistance(a[0],b)
        return ( b, a[1] + distance )

    return reduce(subf,locationList[1:], ( locationList[0] , 0 ) )[1]

def f(locationList) :
    score = 1 / getTotalDistanceByReduce(locationList)
    duplicated = len(locationList) - len(list(set(locationList))) 
    duplicated = duplicated if duplicated != 0 else 1
    #return int(score / (duplicated**2) * 10000) + 1
    return score / (duplicated**2)

def f2(locationList) :
    return 1 / f(locationList)

distanceSum = getTotalDistanceByReduce(["B","A","C","D","E","D","C","A","B"])
print(distanceSum)

class Instance :
    def __init__(self,values) :
        self.gene = values
        self.score = 0

    def evaluate(self,obj) :
        print(self.gene,end="")
        print(" => {}".format(obj(self.gene)))
        self.score = obj(self.gene)
        return self.score

    def show(self) :
        print(self.gene,end="")

class Enviroment :
    def __init__(self) :
        self.instancies = list(Instance(x) for x in permutations(["B","C","D","E"])) * 3
        self.scores = [0 for x in self.instancies]

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
        import numpy as np

        self.updateScore(func)
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

    def timeGoesBy(self,n) :

        newGenes = self.cross_over(n)
        self.instancies.append(newGenes)

        badGenes = self.pickByNumber(n,f2)

        #goodGenes = self.pickByNumber(6,f2)
        # self.instancies 안좋은 놈 쁩은 거를 제외시키기



from sys import exit

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

e = Enviroment()
e.updateScore(f)
rullet = e.make()

print(rullet)
print(sum(rullet))

assert abs( sum(rullet) - 1 ) < 0.0001
import numpy as np

#positions = np.cumsum(rullet)
#oldres = list(e.pick() for x in range(15))
#print(oldres)

#e.checkScore()

badgenes = e.getDead(5)
list( ( i.evaluate(f2) for i in badgenes ) )

exit(0)
#newgenes = e.cross_over()
newgenes[0].evaluate(f)
list( ( i.evaluate(f) for i in newgenes) )


tops = e.topN(5)
print(tops)
#list( ( i.evaluate(f) for i in tops ) )

print("ok")
