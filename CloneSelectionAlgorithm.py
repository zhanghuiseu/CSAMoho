# -*- coding: utf-8 -*-  
import sys
import os
import math
import random
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CommonDefine import *
from datetime import datetime
from ParkerMethond import ParkerMethondFitness
from ParkerMethondGPU import ParkerMethondFitnessGPU

class CloneSelectionAlgorithm:
    def __init__(self):
        self.methodName =  "CSAPO Solver"
    def CSAParaSet(self):
        self.calcType = CalcType.GPU
        self.varCnt = 2
        self.varMinList = [0.25, 5.0]
        self.varMaxList = [0.65, 35.0]
        self.minIterDelt = 1.0
        self.sizePop = 25
        self.maxGenCnt = 12
        self.bestNRate = 0.9
        self.maxCloneSize = 5
        self.mutationRate = 0.9
        self.alphaMin = 0.1
        self.alphaMax = 5
        self.saveEveryGenResult = True
        self.exportResultPlt = False
        self.initPopDataPath = ""
        self.realSolveVec = [0.441, 22.78]
        self.finalVarMinList = self.varMinList
        self.finalVarMaxList = self.varMaxList
        self.MAX_ITER_CNT = 99999 + random.randint(0, 200)
        self.allPopulationList = list()
        if not os.path.exists('./result'):
          os.mkdir('./result')
        if not os.path.exists('./data'):
          os.mkdir('./data')
        if not os.path.exists('./log'):
          os.mkdir('./log')
        if not os.path.exists('./result/temp'):
          os.mkdir('./result/temp')
        return 0

    def ParkerMethodParaInit(self):
        self.initSeed = time.time()
        random.seed(self.initSeed)
        self.sequenceId = 0
        self.gravityinputpath = './data/gravity_real.txt'
        self.seisminputpath = './data/seismic_real.txt'
        self.criterio = 0.02
        self.WH = 0.0000
        self.SH = 0.0185
        self.truncation = 0.15
        self.seismVerifySize = 30
        if CalcType.SINGLE_CPU == self.calcType:
            self.parkerHander = ParkerMethondFitness(self.gravityinputpath, self.seisminputpath, \
                       self.criterio, self.WH, self.SH, self.truncation, self.seismVerifySize, self.calcType)
        else:
            self.parkerHander = ParkerMethondFitnessGPU(self.gravityinputpath, self.seisminputpath, \
                       self.criterio, self.WH, self.SH, self.truncation, self.seismVerifySize, self.calcType)
        return self.parkerHander.InitPreCalc()

    def InitAllPara(self):
        self.CSAParaSet()
        ret = self.ParkerMethodParaInit()
        if 0 != ret:
            logger.error('ParkerMethodParaInit failed')
        return ret

    def setSizePop(self, sizePop):
        self.sizePop = sizePop
        logger.info('***************************** setSizePop {} Take Effect'.format(self.sizePop))
    def setMaxGenCnt(self, maxGenCnt):
        self.maxGenCnt = maxGenCnt
        logger.info('***************************** setMaxGenCnt {} Take Effect'.format(self.maxGenCnt))
    def setMutationRate(self, mutationRate):
        self.mutationRate = mutationRate
        logger.info('***************************** setMutationRate {} Take Effect'.format(self.mutationRate))
    def EnableExportResultPlt(self):
        self.exportResultPlt = True
        logger.info('***************************** EnableExportResultPlt {} Take Effect'.format(self.exportResultPlt))
    def DisableExportResultPlt(self):
        self.exportResultPlt = False
        logger.info('***************************** DisableExportResultPlt {} Take Effect'.format(self.exportResultPlt))
    def setinitPopDataPath(self, initPopDataPath):
        self.initPopDataPath = initPopDataPath
        logger.info('***************************** setinitPopDataPath {} Take Effect'.format(self.initPopDataPath))

    def setCSAParas(self, calcType = CalcType.GPU, varMinList = [0.38, 40], varMaxList = [0.58, 60], minIterDelt = 1, sizePop = 60, \
                maxGenCnt = 12, bestNRate = 0.9, maxCloneSize = 10, mutationRate = 0.5, alphaMin = 0, alphaMax = 1, \
                saveEveryGenResult = True, initPopDataPath = ""):
        self.calcType = calcType
        self.varMinList = varMinList
        self.varMaxList = varMaxList
        self.minIterDelt = minIterDelt
        self.sizePop = sizePop
        self.maxGenCnt = maxGenCnt
        self.bestNRate = bestNRate
        self.maxCloneSize = maxCloneSize
        self.mutationRate = mutationRate
        self.alphaMin = alphaMin
        self.alphaMax = alphaMax
        self.saveEveryGenResult = saveEveryGenResult
        self.exportResultPlt = True
        self.initPopDataPath = initPopDataPath
        self.realSolveVec = [0.441, 22.78]
        self.finalVarMinList = self.varMinList
        self.finalVarMaxList = self.varMaxList
        self.MAX_ITER_CNT = 99999 + random.randint(0, 200)
        self.allPopulationList = list()

        if not os.path.exists('./result'):
          os.mkdir('./result')
        if not os.path.exists('./data'):
          os.mkdir('./data')
        if not os.path.exists('./log'):
          os.mkdir('./log')
        if not os.path.exists('./result/temp'):
          os.mkdir('./result/temp')
        return 0

    def setParkerParas(self, gravityinputpath = './data/A6Gravity.txt', seisminputpath = './data/ControlPoint.txt', criterio = 0.02, WH = 0.01, SH = 0.0195, truncation = 0.135, seismVerifySize = 20):
        self.initSeed = time.time()
        random.seed(self.initSeed)
        logger.info('setParkerParas initSeed {}'.format(self.initSeed))
        self.sequenceId = 0
        self.gravityinputpath = gravityinputpath
        self.seisminputpath = seisminputpath
        self.criterio = criterio
        self.WH = WH
        self.SH = SH
        self.truncation = truncation
        self.seismVerifySize = seismVerifySize
        if CalcType.SINGLE_CPU == self.calcType:
            self.parkerHander = ParkerMethondFitness(self.gravityinputpath, self.seisminputpath, \
                       self.criterio, self.WH, self.SH, self.truncation, self.seismVerifySize, self.calcType)
        else:
            self.parkerHander = ParkerMethondFitnessGPU(self.gravityinputpath, self.seisminputpath, \
                       self.criterio, self.WH, self.SH, self.truncation, self.seismVerifySize, self.calcType)
        return self.parkerHander.InitPreCalc()

    def calcFitness(self, xVar):
        if self.varCnt != len(xVar):
            logger.error('calcFitness failed MUST BUG1 {}'.format(self.varCnt))
            return -1, -1
        contrast = xVar[0]
        z0 = xVar[1]
        rms, originRms = self.parkerHander.CalcFitness(contrast, z0, self.sequenceId)
        self.sequenceId += 1
        self.originFitness = originRms
        if rms != originRms:
            return rms + self.minIterDelt, originRms
        else:
            return originRms, originRms

    def InitOnePerson(self, index, initialVar = list()):
        contrast = -1
        z0 = -1
        if len(initialVar) == self.varCnt:
            if index == 0:
                contrast = initialVar[0] + random.normalvariate(0, 1) * 0.01
                z0 = initialVar[1] + random.normalvariate(0, 1) * 0.1
                logger.info('InitOnePerson {} {} from initialVar'.format(contrast, z0))
            else:
                contrast = initialVar[0] + (self.varMaxList[0] - self.varMinList[0]) * random.normalvariate(0, 1)
                z0 = initialVar[1] + (self.varMaxList[1] - self.varMinList[1]) * random.normalvariate(0, 1)
                logger.info('InitOnePerson {} {} from initialVar Add Disturbance'.format(contrast, z0))
        else:
            if len(self.realSolveVec) == self.varCnt:
                while(True):
                    contrast = self.varMinList[0] + (self.varMaxList[0] - self.varMinList[0]) * np.random.random()
                    if math.fabs(contrast - self.realSolveVec[0]) > 0.1:
                        break
                    else:
                        logger.warning('InitOnePerson BadCase1 Repeated')
                while(True):
                    z0 = self.varMinList[1] + (self.varMaxList[1] - self.varMinList[1]) * np.random.random()
                    if math.fabs(z0 - self.realSolveVec[1]) > 1.0:
                        break
                    else:
                        logger.warning('InitOnePerson BadCase2 Repeated')
            else:
                contrast = self.varMinList[0] + (self.varMaxList[0] - self.varMinList[0]) * np.random.random()
                z0 = self.varMinList[1] + (self.varMaxList[1] - self.varMinList[1]) * np.random.random()
            logger.info('InitOnePerson {} {} from Random'.format(contrast, z0))

        contrast = max(min(contrast, self.varMaxList[0]), self.varMinList[0])
        z0 = max(min(z0, self.varMaxList[1]), self.varMinList[1])
        return [contrast, z0]

    def initPopulation(self, initialVar = list()):
        if self.initPopDataPath != '' and os.path.exists('./result/' + self.initPopDataPath):
            with open('./result/' + self.initPopDataPath) as fd:
                allData = []
                for data in fd.readlines():
                    tmp = data.split(' ')
                    if len(tmp) != 2:
                        logger.error('initPopulation initPopDataPath {} data {} is INVAILD'.format(self.initPopDataPath, data))
                        continue
                    contrast = float(tmp[0])
                    z0 = float(tmp[1])
                    allData.append([contrast, z0])
                    logger.info('InitOnePerson {} {} from initPopDataPath {}'.format(contrast, z0, self.initPopDataPath))

                self.sizePop = len(allData)
                self.allJudgeRmsArray= np.zeros(self.sizePop)
                self.allOriginRmsArray= np.zeros(self.sizePop)
                for i in range(0, self.sizePop):
                    self.allPopulationList.append(allData[i])
                    self.allJudgeRmsArray[i] = 1e10
                    self.allOriginRmsArray[i] = 1e10
            logger.info('InitOnePerson sizePop {} from initPopDataPath {}'.format(self.sizePop, self.initPopDataPath))
        else:
            self.allJudgeRmsArray= np.zeros(self.sizePop)
            self.allOriginRmsArray= np.zeros(self.sizePop)
            for i in range(0, self.sizePop):
                oneVar = self.InitOnePerson(i, initialVar)
                self.allPopulationList.append(oneVar)
                self.allJudgeRmsArray[i] = 1e10
                self.allOriginRmsArray[i] = 1e10
        logger.info("initPopulation done")

    def judgePopulation(self):
        oneSearchResult = list()
        for i in range(0, self.sizePop):
            judgeRms, originRms = self.calcFitness(self.allPopulationList[i])
            self.allJudgeRmsArray[i] = judgeRms
            self.allOriginRmsArray[i] = originRms
            oneSearchResult.append([self.allPopulationList[i][0], self.allPopulationList[i][1], originRms])
        self.allSearchResult.append(oneSearchResult)

    def OptimizationCSA(self):
        logger.info('CSAParaSet initSeed {} varMinList {} varMaxList {} minIterDelt {} sizePop {} ' \
            'maxGenCnt {} bestNRate {} maxCloneSize {} mutationRate {} alpha[{}, {}] initPopDataPath {}' \
            .format(self.initSeed, str(self.varMinList), str(self.varMaxList), self.minIterDelt, self.sizePop, \
                self.maxGenCnt, self.bestNRate, self.maxCloneSize, self.mutationRate, self.alphaMin, self.alphaMax, self.initPopDataPath))
        startCSATime = time.time()
        self.allSearchResult = list()
        self.indexGen = 0
        initialVar = []
        self.initPopulation(initialVar)
        self.judgePopulation()
        bestRms = np.min(self.allJudgeRmsArray)
        bestIndex = np.argmin(self.allJudgeRmsArray)
        logger.info('CSAParaSet final sizePop {} '.format(self.sizePop))
        self.globalBestVar = list()
        self.globalBestVar[:] = self.allPopulationList[bestIndex][:]
        self.globalBestRms = bestRms
        self.globalFindBetterCnt = 0
        self.globalMutationCnt = 0
        self.recordNowFitList = list()
        self.recordNowFitList.append(bestRms)
        self.recordBestFitList = list()
        self.recordBestFitList.append(self.globalBestRms)
        logger.info("OptimizationCSA Generation {} sequenceId {} BestVar {} BestRms {} globalFindBetterCnt {}".format( \
            self.indexGen, self.sequenceId, str(self.globalBestVar), self.globalBestRms, self.globalFindBetterCnt))
        while (self.indexGen < self.maxGenCnt - 1):
            self.indexGen += 1
            tempPopulationList1 = self.getBestNOperation()
            tempPopulationList2 = self.cloneOperation(tempPopulationList1)
            tempPopulationList3 = self.mutationOperation(tempPopulationList2)
            tempPopulationList4 = self.selectionOperation(tempPopulationList3)
            self.refillOperation(tempPopulationList4)
            self.judgePopulation()
            bestRms = np.min(self.allJudgeRmsArray)
            bestIndex = np.argmin(self.allJudgeRmsArray)
            if bestRms < self.globalBestRms:
                self.globalBestVar[:] = self.allPopulationList[bestIndex][:]
                self.globalBestRms = bestRms
                self.globalFindBetterCnt += 1
            self.recordNowFitList.append(bestRms)
            self.recordBestFitList.append(self.globalBestRms)
            if self.globalBestRms <= self.minIterDelt:
                break
            logger.info("OptimizationCSA Generation {} sequenceId {} BestVar {} BestRms {} globalFindBetterCnt {}".format( \
                self.indexGen, self.sequenceId, str(self.globalBestVar), self.globalBestRms, self.globalFindBetterCnt))

        endCSATime = time.time()
        logger.info('BBBBBBBBBBBB OptimizationCSA Generation {} totalMarkovCnt {} BestVar {} BestRms {} totalCostTime {:.2f} s'.format(self.indexGen, \
            self.sequenceId, str(self.globalBestVar), self.globalBestRms, (endCSATime - startCSATime)))
        logger.info('BBBBBBBBBBBB OptimizationCSA FindBetterRate({}/{}) = {:.2f} %' \
            .format(self.globalFindBetterCnt, self.indexGen, float(self.globalFindBetterCnt) / (1 if self.indexGen == 0 else self.indexGen) * 100.0))

        strNowTime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        if self.saveEveryGenResult == True:
            for i in range(len(self.allSearchResult)):
                oneSearchdata = self.allSearchResult[i]
                with open('result/a' + strNowTime + '_onesearchinfo_' + str(i) + '.dat', 'w+') as fd:
                    for data in oneSearchdata:
                        fd.write(str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + '\n')

        with open('result/a' + strNowTime + '_searchinfo.dat', 'w+') as fd:
            for oneSearchdata in self.allSearchResult:
                for data in oneSearchdata:
                    fd.write(str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + '\n')

        logger.info("Final BestVar {} {} BestRms {} Over".format(self.globalBestVar[0], self.globalBestVar[1], self.globalBestRms))
        contrast = self.globalBestVar[0]
        z0 = self.globalBestVar[1]
        logger.info('>>>>>>>>>>> BEGIN Target contrast {} z0 {} CalcFitness'.format(contrast, z0))
        self.parkerHander.CalcFitness(contrast, z0, 0, False, SemisDataType.TEST)
        self.parkerHander.CalcFitness(contrast, z0, 0, False, SemisDataType.VERIFY)
        self.parkerHander.CalcFitness(contrast, z0, 0, True, SemisDataType.ALL)
        logger.info('>>>>>>>>>>> END Target contrast {} z0 {} CalcFitness'.format(contrast, z0))

    def getBestNOperation(self):
        inputPopulationList = list()
        bestNum = math.ceil(self.bestNRate * self.sizePop)
        if bestNum >= self.sizePop:
            inputPopulationList = copy.deepcopy(self.allPopulationList)
            return inputPopulationList
        sortIndexList = np.argsort(self.allJudgeRmsArray)
        for i in range(0, bestNum):
            targetIndex = sortIndexList[i]
            inputPopulationList.append(copy.deepcopy(self.allPopulationList[targetIndex]))
        return inputPopulationList

    def cloneOperation(self, inputPopulationList):
        finalPopulationList = list()
        for one in inputPopulationList:
            tmpList = [copy.deepcopy(one) for i in range(self.maxCloneSize)]
            finalPopulationList.append(tmpList)
        return finalPopulationList

    def mutationOperation(self, inputPopulationList):
        finalPopulationList = list()
        for i in range(len(inputPopulationList)):
            tmpList = list()
            for j in range(self.maxCloneSize):
                delta = self.alphaMax + (self.indexGen + 1) * (self.alphaMax - self.alphaMin) / self.maxGenCnt
                contrast = inputPopulationList[i][j][0]
                z0 = inputPopulationList[i][j][1]
                prob = np.random.random(1)
                if prob <= self.mutationRate:
                    contrast += random.normalvariate(0, delta) * 0.01 * (self.maxGenCnt - self.indexGen) / self.maxGenCnt
                    z0 += random.normalvariate(0, delta) * 0.1
                    contrast = max(min(contrast, self.varMaxList[0]), self.varMinList[0])
                    z0 = max(min(z0, self.varMaxList[1]), self.varMinList[1])
                judgeRms, originRms = self.calcFitness([contrast, z0])
                tmpList.append([contrast, z0, judgeRms])
            finalPopulationList.append(tmpList)
        return finalPopulationList

    def selectionOperation(self, inputPopulationList):
        finalPopulationList = list()
        for i in range(len(inputPopulationList)):
            targetIndex = self.maxCloneSize
            minRMS = 1e20
            for j in range(self.maxCloneSize):
                if inputPopulationList[i][j][2] < minRMS:
                    minRMS = inputPopulationList[i][j][2]
                    targetIndex = j
            [contrast, z0, rms] = inputPopulationList[i][targetIndex]
            finalPopulationList.append([contrast, z0])
        return finalPopulationList

    def refillOperation(self, inputPopulationList):
        if len(inputPopulationList) < self.sizePop:
            refillSize = self.sizePop - len(inputPopulationList)
            for i in range(refillSize):
                contrast = self.varMinList[0] + (self.varMaxList[0] - self.varMinList[0]) * np.random.random()
                z0 = self.varMinList[1] + (self.varMaxList[1] - self.varMinList[1]) * np.random.random()
                inputPopulationList.append([contrast, z0])
                logger.info('refillOperation progress ({}/{}) {} {} from Random'.format(i+1, refillSize, contrast, z0))
        self.allPopulationList = copy.deepcopy(inputPopulationList)
        return

    def GenInitPopData(self):
        targetSize = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 2]
        maxNum = max(targetSize)
        finalList = []
        for i in range(maxNum):
            contrast = self.varMinList[0] + (self.varMaxList[0] - self.varMinList[0]) * np.random.random()
            z0 = self.varMinList[1] + (self.varMaxList[1] - self.varMinList[1]) * np.random.random()
            finalList.append([contrast, z0])
        chooseIndex = np.array(range(maxNum))
        for num in targetSize:
            chooseOne = np.random.choice(chooseIndex, size=num, replace=False)
            with open("./result/initpop_" + str(num) + ".dat", 'w+') as fd:
                for i in chooseOne:
                    one = finalList[i]
                    fd.write(str(one[0]) + " " + str(one[1]) + '\n')
            logger.info('GenInitPopData {} succ'.format(num))
            chooseIndex = chooseOne
        with open("./result/initpop_edge.dat", 'w+') as fd:
            maxX = self.varMaxList[0]
            minX = self.varMinList[0]
            maxY = self.varMaxList[1]
            minY = self.varMinList[1]
            fd.write(str(maxX) + " " + str(maxY) + '\n')
            fd.write(str(maxX) + " " + str(minY) + '\n')
            fd.write(str(minX) + " " + str(maxY) + '\n')
            fd.write(str(minX) + " " + str(minY) + '\n')

if __name__ == "__main__":
    for arg in sys.argv:
        logger.info('***************************** ArgInfo[{}]: {}'.format(len(sys.argv), arg))
    startTime = time.time()
    logger.info('***************************** Game Begin')
    csa = CloneSelectionAlgorithm()
    if 0 != csa.InitAllPara():
        exit(1)
    if len(sys.argv) == 3:
        cmdType = int(sys.argv[1])
        if cmdType == 0:
            csa.setSizePop(int(sys.argv[2]))
        elif cmdType == 1:
            csa.setMaxGenCnt(int(sys.argv[2]))
        elif cmdType == 2:
            csa.setMutationRate(float(sys.argv[2]))
        else:
            logger.info('***************************** Arg Not Take Effect')

    contrast = 0.441
    z0 = 22.78
    logger.info('>>>>>>>>>>> BEGIN Target contrast {} z0 {} CalcFitness'.format(contrast, z0))
    csa.parkerHander.CalcFitness(contrast, z0, 0, False, SemisDataType.TEST)
    csa.parkerHander.CalcFitness(contrast, z0, 0, False, SemisDataType.VERIFY)
    csa.parkerHander.CalcFitness(contrast, z0, 0, True, SemisDataType.ALL)
    logger.info('>>>>>>>>>>> END Target contrast {} z0 {} CalcFitness'.format(contrast, z0))
    #exit(1)
    csa.OptimizationCSA()
    contrast = 0.441
    z0 = 22.78
    logger.info('>>>>>>>>>>> BEGIN Target contrast {} z0 {} CalcFitness'.format(contrast, z0))
    csa.parkerHander.CalcFitness(contrast, z0, 0, False, SemisDataType.TEST)
    csa.parkerHander.CalcFitness(contrast, z0, 0, False, SemisDataType.VERIFY)
    csa.parkerHander.CalcFitness(contrast, z0, 0, True, SemisDataType.ALL)
    logger.info('>>>>>>>>>>> END Target contrast {} z0 {} CalcFitness'.format(contrast, z0))