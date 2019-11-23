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
    print("From {} to {} Distance : {}".format(start,end,df.loc[start,end]))
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

distanceSum = getTotalDistanceByReduce(["B","A","C","D","E","D","C","A","B"])
print(distanceSum)

class Instance :
    def __init__(self,values) :
        self.gene = values

    def evaluate(self,obj) :
        print(self.gene,end="")
        print(" => {}".format(obj(self.gene)))
        return obj(self.gene)

    def show(self) :
        print(self.gene,end="")

class Enviroment :
    def __init__(self,instanceCount) :
        self.instancies = list(Instance(x) for x in permutations(["B","C","D","E"]))
        self.scores = [0 for x in self.instancies]

    def checkScore(self) :
        self.scores = list(map(lambda x : x.evaluate(getTotalDistance) , self.instancies))

    def make(self) :
        total = sum(self.scores)

        def toProbability(x) :
            return x / total

        return list(map( toProbability , self.scores))

    def newPick(self,n) :
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

        def pickRecursivly(beforeResult=[]) :
            p = n - len(beforeResult)

            r = list(set(list( pick() for _ in range(p) )))
            r = list(set(r + beforeResult))

            if n - len(r) == 0 :
                return r 

            return pickRecursivly(r)

        return pickRecursivly()

    def pick(self) :
        import numpy as np
        r = [0] + self.make()

        sets = np.cumsum(r)

        print(sets)
        vtuples = list(map(lambda x : (sets[x] , sets[x+1]) , range(0,len(r)-1)))

        from random import randint 

        rvalue = randint(1,100000000) / 100000000
        index = 0

        for t in vtuples :

            if rvalue > t[0] and rvalue <= t[1] :
                return index

            index += 1

    #구조,파라미터가 어떻게 들어올지 고민
    def cross_over(self) :
        pass

    def next(self) :
        """
            1. pick -> 제일 좋은 애뽑민
            2. 안좋은애 뽑기() ** 고민
            3. 크로스 오버
        """
        self.cross_over()

        def cross_over() :
            pass


e = Enviroment(10)
res = e.checkScore()
rullet = e.make()

assert abs( sum(rullet) - 1 ) < 0.00000001
import numpy as np

#positions = np.cumsum(rullet)
#oldres = list(e.pick() for x in range(15))
#print(oldres)

res = e.newPick(15)
print(res)

assert len(res) == len(set(res))
print("ok")






