# -*- coding: utf-8 -*-  
import sys
import os
import math
import random
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn
import enum
from scipy import signal
from datetime import datetime
from loguru import logger
from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import functools

CalcType = enum.Enum('CalcType', ('SINGLE_CPU', 'MULTI_CPU', 'GPU', 'CPU_GPU'))
SemisDataType = enum.Enum('SemisDataType', ('ALL', 'TEST', 'VERIFY'))
InterDataExportType = enum.Enum('InterDataExportType', ('ALL', 'INCER', 'NONE'))
UpdateVerifyDataSet = enum.Enum('UpdateVerifyDataSet', ('YES', 'NO'))
punishRate = 1000.0
outputInterDataType = InterDataExportType.NONE
updateVerifyFlag = UpdateVerifyDataSet.NO
#logger.remove(handler_id = None)
logger.add("./log/optresult_{time}.log", rotation="10GB", encoding="utf-8", enqueue=True, compression="zip", level="INFO")
logger.debug('AAAAAAAAAAAA TEST debug log')
logger.info('AAAAAAAAAAAA TEST debug log')
logger.warning('AAAAAAAAAAAA TEST debug log')
logger.error('AAAAAAAAAAAA TEST debug log')
logger.info('AAAAAAAAAAAA TEST debug log')
prodResMap = dict()
def prod(a, b):
    if a == 0 or b == 0 or a > b:
        return 1
    if 1 == a and b in prodResMap.keys():
        return prodResMap[b]
    res = 1
    for i in range(a, b + 1):
        res *= i
    if 1 == a:
        prodResMap[b] = res
    return res
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(math.radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance = 2 * math.asin(math.sqrt(a)) * 6371
    return distance

class ParkerMethondBase:
    def __init__(self, gravityinputpath, seisminputpath, criterio, WH, SH, truncation, seismVerifySize, calcType = CalcType.SINGLE_CPU):
      self.gravityinputpath = gravityinputpath
      self.seisminputpath = seisminputpath
      self.allDataPath = 'result/seismall.dat'
      self.testDataPath = 'result/seismtest.dat'
      self.verifyDataPath = 'result/seismverify.dat'
      self.criterio = criterio
      self.WH = WH
      self.SH = SH
      self.truncation = truncation
      self.numrows = 0
      self.numcolumns = 0
      self.longx = 0
      self.longy = 0
      self.lastRmsFitness = 1e10
      self.preCalcFlag = False
      self.seismVerifySize = seismVerifySize
      self.calcType = calcType
      for i in range(1, 12):
         prod(1, i)
      logger.info('ParkerMethondBase succ calcType:{} gravityinputpath:{} criterio:{} seismVerifySize:{} WH:{} SH:{} '
         'truncation:{}'.format(self.calcType.name, self.gravityinputpath,
         self.criterio, self.seismVerifySize, self.WH, self.SH, self.truncation))

    def InitPreCalcBase(self):
      self.xyGravityIndexMap = dict()
      bou = list()
      self.minx, self.maxx = 1e10, -1
      self.miny, self.maxy = 1e10, -1
      with open(self.gravityinputpath) as fd:
         alldatas = fd.readlines()
         index = 0
         for data in alldatas:
            data = data.strip().replace('\t', ' ').strip()
            data = data.replace('    ', ' ').strip()
            data = data.replace('   ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            oneData = [float(i) for i in data.split(' ')]
            if 0 == len(oneData):
               continue
            if len(oneData) != 3:
               logger.error("MUST BUG1 gravity {} size != 3".format(data))
               return -1
            if oneData[0] not in self.xyGravityIndexMap.keys():
               self.xyGravityIndexMap[oneData[0]] = dict()
            if oneData[1] not in self.xyGravityIndexMap[oneData[0]].keys():
               self.xyGravityIndexMap[oneData[0]][oneData[1]] = index
               index += 1
            else:
               logger.error("MUST BUG2 gravity ({}, {}) repeat".format(oneData[0], oneData[1]))
               return -1
            bou.append(oneData[2])
            self.minx = min(self.minx, oneData[0])
            self.maxx = max(self.maxx, oneData[0])
            self.miny = min(self.miny, oneData[1])
            self.maxy = max(self.maxy, oneData[1])
      self.numrows = len(self.xyGravityIndexMap.keys())
      self.numcolumns = 0
      for oneRow in self.xyGravityIndexMap.keys():
         colsize = len(self.xyGravityIndexMap[oneRow])
         if 0 == self.numcolumns:
            self.numcolumns = colsize
         elif self.numcolumns != colsize:
            logger.error("MUST BUG3 x {} size {} numcolumns {} != {}" .format(oneRow, colsize, self.numcolumns))
            return -1
         else:
            pass
      if self.minx >= self.maxx or self.miny >= self.maxy:
         logger.error('MUST BUG4 x in range [{}, {}] y in range [{}, {}] '.format(self.minx, self.maxx, self.miny, self.maxy))
         return -1
      if self.numrows == 0 or self.numcolumns == 0:
         logger.error("MUST BUG5 x numrows {} numcolumns {} is 0" .format(self.numrows, self.numcolumns))
         return -1
      self.longx = (geodistance(self.minx, self.miny, self.maxx, self.miny) + geodistance(self.minx, self.maxy, self.maxx, self.maxy)) / 2.0
      self.longy = (geodistance(self.minx, self.miny, self.minx, self.maxy) + geodistance(self.maxx, self.miny, self.maxx, self.maxy)) / 2.0
      if self.longy == 0 or self.longy == 0:
         logger.error("MUST BUG6 longx {} longy {} is 0" .format(self.longy, self.longy))
         return -1
      logger.info('ParkerMethondFitness succ x in range [{}, {}] y in range [{}, {}] numrows: {} numcolumns: {} '
            'longx: {} longy: {} '.format(self.minx, self.maxx, self.miny, self.maxy, self.numrows, self.numcolumns,
             self.longx, self.longy))
      self.xySeismAllDataMap, self.allDataSize = self.GetSeismData(self.allDataPath)
      if len(self.xySeismAllDataMap) <= 0:
         logger.error("MUST BUG5 xySeismAllDataMap is empty")
         return -1
      if self.seismVerifySize <= 0 or self.seismVerifySize * 5 > self.numrows * self.numcolumns:
         logger.error("MUST BUG6 seismVerifySize {} xySeismAllDataMap {} is Invaild".format(self.seismVerifySize, self.numrows * self.numcolumns))
         return -1
      if updateVerifyFlag == UpdateVerifyDataSet.YES:
         strNowTime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
         if os.path.exists(self.testDataPath):
            os.rename(self.testDataPath, 'result/seismtest' + strNowTime + '.dat')
         if os.path.exists(self.verifyDataPath):
            os.rename(self.verifyDataPath, 'result/seismverify' + strNowTime + '.dat')
         if self.GenerateVerityData(self.xySeismAllDataMap, self.testDataPath, self.verifyDataPath) != 0:
           logger.error("MUST BUG6 GenerateVerityData failed")
           return -1
         logger.info("GenerateVerityData succ")
      self.xySeismTestDataMap, self.testDataSize= self.GetSeismData(self.testDataPath)
      self.xySeismVerifyDataMap, self.verifyDataSize= self.GetSeismData(self.verifyDataPath)
      bou = np.array(bou)
      bou = bou.reshape(self.numcolumns, self.numrows)
      logger.info('InputDataSize: {} '.format(bou.shape))
      self.fftbou = np.fft.fft2(bou)
      meangravity = self.fftbou[0,0] / (self.numrows * self.numcolumns)
      bou = bou - meangravity
      logger.info('meangravity: {} '.format(meangravity))
      wrows = signal.windows.tukey(self.numrows, self.truncation).reshape(self.numrows, 1)
      wcolumns = signal.windows.tukey(self.numcolumns, self.truncation).reshape(self.numcolumns, 1)
      w2 = np.dot(wrows, (wcolumns.T))
      bou = bou * (w2.conjugate().T)
      mapabou = bou.conjugate().T
      self.fftbou = np.fft.fft2(mapabou)
      subRow = (int)(self.numrows / 2)
      subCol = (int)(self.numcolumns / 2)
      spectrum = np.abs(self.fftbou[0:subRow, 0:subCol])
      frequency = np.zeros((subRow + 1, subCol + 1), dtype=float)
      for f in range(1, subRow + 2):
         for g in range(1, subCol + 2):
            frequency[f - 1][g - 1] = np.sqrt(((f - 1) / self.longx) ** 2 + ( (g - 1) / self.longy) ** 2)
      frequency2 = np.fliplr(frequency)
      frequency3 = np.flipud(frequency)
      frequency4 = np.fliplr(np.flipud(frequency))
      entero = np.round(self.numcolumns / 2)
      if self.numcolumns == 2 * entero:
         frequency2 = np.delete(frequency2, 0, axis = 1)
         frequency3 = np.delete(frequency3, 0, axis = 0)
         frequency4 = np.delete(frequency4, 0, axis = 1)
         frequency4 = np.delete(frequency4, 0, axis = 0)
      frequencypart1 = np.append(frequency, frequency2, axis = 1)
      frequencypart2 = np.append(frequency3, frequency4, axis = 1)
      self.frequencytotal = np.append(frequencypart1, frequencypart2, axis=0)
      self.frequencytotal = np.delete(self.frequencytotal, -1, axis = 0)
      self.frequencytotal = np.delete(self.frequencytotal, -1, axis = 1)
      self.frequencytotal = self.frequencytotal * 2 * np.pi
      self.filter = self.frequencytotal * 0
      self.frequencytotal = self.frequencytotal / (2 * np.pi)
      for f in range(self.numrows):
         for g in range(self.numcolumns):
            if self.frequencytotal[f][g] < self.WH:
               self.filter[f][g] = 1.0
            elif self.frequencytotal[f][g] < self.SH:
               self.filter[f][g] = 0.5 * (1 + np.cos((2 * np.pi * self.frequencytotal[f][g] \
                                 - 2 * np.pi * self.WH) / (2 * (self.SH - self.WH))) )
            else:
               self.filter[f][g] = 0
      self.frequencytotal = self.frequencytotal * 2 * np.pi
      return 0

    def GetNearPoint(self, x, y, xyGravityIndexMap):
       res = list()
       if x in xyGravityIndexMap.keys():
        if y in xyGravityIndexMap[x].keys():
           res.append([1, xyGravityIndexMap[x][y]])
           return res
        else:
           nearminy, nearmaxy = -1, 1e10
           for tempy in xyGravityIndexMap[x].keys():
              if tempy < y:
                 nearminy = max(nearminy, tempy)
              if tempy > y:
                 nearmaxy = min(nearmaxy, tempy)
           if nearminy not in xyGravityIndexMap[x].keys() \
              or nearmaxy not in xyGravityIndexMap[x].keys():
              logger.error('MUST BUG1 Not find target [{}, {}]' .format(x, y))
              return list()
           res.append([geodistance(x, y, x, nearminy), xyGravityIndexMap[x][nearminy]])
           res.append([geodistance(x, y, x, nearmaxy), xyGravityIndexMap[x][nearmaxy]])
       else:
           nearminx, nearmaxx = -1, 1e10
           for tempx in xyGravityIndexMap.keys():
              if tempx < x:
                 nearminx = max(nearminx, tempx)
              if tempx > x:
                 nearmaxx = min(nearmaxx, tempx)
           if nearminx not in xyGravityIndexMap.keys()   \
              or nearmaxx not in xyGravityIndexMap.keys():
              logger.error('MUST BUG2 Not find target [{}, {}]' .format(x, y))
              return list()
           nearminy, nearmaxy = -1, 1e10
           for tempy in xyGravityIndexMap[nearminx].keys():
              if tempy == y:
                 nearminy = tempy
                 nearmaxy = tempy
                 break
              if tempy < y:
                 nearminy = max(nearminy, tempy)
              if tempy > y:
                 nearmaxy = min(nearmaxy, tempy)
           if nearminy == y:
              if nearminy not in xyGravityIndexMap[nearminx].keys()     \
                    or nearmaxy not in xyGravityIndexMap[nearminx].keys():
                    logger.error('MUST BUG3 Not find target [{}, {}] {} {} {} {}' .format(x, y, nearminx, nearmaxy))
                    return list()
              res.append([geodistance(x, y, nearminx, y), xyGravityIndexMap[nearminx][y]])
              res.append([geodistance(x, y, nearmaxx, y), xyGravityIndexMap[nearmaxx][y]])
           else:
              if nearminy not in xyGravityIndexMap[nearminx].keys()     \
                 or nearmaxy not in xyGravityIndexMap[nearminx].keys()  \
                 or nearminy not in xyGravityIndexMap[nearmaxx].keys()  \
                 or nearmaxy not in xyGravityIndexMap[nearmaxx].keys() :
                 logger.error('MUST BUG4 Not find target [{}, {}] {} {} {} {}' .format(x, y, nearminx, nearmaxx, nearminy, nearmaxy))
                 return list()
              res.append([geodistance(x, y, nearminx, nearminy), xyGravityIndexMap[nearminx][nearminy]])
              res.append([geodistance(x, y, nearminx, nearmaxy), xyGravityIndexMap[nearminx][nearmaxy]])
              res.append([geodistance(x, y, nearmaxx, nearminy), xyGravityIndexMap[nearmaxx][nearminy]])
              res.append([geodistance(x, y, nearmaxx, nearmaxy), xyGravityIndexMap[nearmaxx][nearmaxy]])
           return res
       return res

    def GenerateVerityData(self, xySeismDataMap, testDatapath, verifyDataPath):
      self.inputX = list()
      for x in xySeismDataMap.keys():
         for y, info in xySeismDataMap[x].items():
            self.inputX.append([x, y])
      self.inputX = np.array(self.inputX)
      self.outputY = KMeans(self.seismVerifySize).fit_predict(self.inputX)
      index = 0
      xy2ClassificRes = dict()
      classific2xyRes = dict()
      for x in xySeismDataMap.keys():
         for y, info in xySeismDataMap[x].items():
            classificType = self.outputY[index]
            index += 1
            xy2ClassificRes[x] = dict()
            xy2ClassificRes[x][y] = classificType

            if classificType not in classific2xyRes.keys():
               classific2xyRes[classificType] = list()
            classific2xyRes[classificType].append([x, y])
      choiceResList = list()
      for classificType, data in classific2xyRes.items():
         choiceResList.append(random.choice(data))
      with open(verifyDataPath, "w+") as fd:
         for choiceRes in choiceResList:
            x = choiceRes[0]
            y = choiceRes[1]
            fd.write(str(x) + ' ' + str(y) + ' ' + str(xySeismDataMap[x][y][0]) + '\n')
      with open(testDatapath, "w+") as fd:
         for x in xySeismDataMap.keys():
            for y, info in xySeismDataMap[x].items():
               if [x, y] not in choiceResList:
                  fd.write(str(x) + ' ' + str(y) + ' ' + str(info[0]) + '\n')
      return 0

    def ClassificShow(self):
      fig = plt.figure()
      plt.title('Classification Results')
      plt.xlabel('XLable')
      plt.ylabel('YLable')
      finalOutputY = list()
      inputX = []
      inputY = []
      for x in self.xySeismAllDataMap.keys():
         for y, info in self.xySeismAllDataMap[x].items():
            if x in self.xySeismVerifyDataMap.keys() and y in self.xySeismVerifyDataMap[x].keys():
               finalOutputY.append(1)
            else:
               finalOutputY.append(0)
            inputX.append(x)
            inputY.append(y)
      plt.scatter(inputX, inputY, c = finalOutputY)
      plt.show()

    def GetSeismData(self, filePath):
      xySeismDataMap = dict()
      xySeismSize = 0
      with open(filePath) as fd:
         alldatas = fd.readlines()
         for data in alldatas:
            data = data.strip().replace('\t', ' ').strip()
            data = data.strip().replace(',', ' ').strip()
            data = data.replace('    ', ' ').strip()
            data = data.replace('   ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            oneData = [float(i) for i in data.split(' ')]
            if 0 == len(oneData):
               continue
            if len(oneData) != 3:
               logger.error("MUST BUG1 seism {} size != 3".format(data))
               return -1
            tempx = oneData[0]
            tempy = oneData[1]
            if tempx < self.minx or tempx > self.maxx or tempy < self.miny or tempy > self.maxy:
               logger.info("depth ({}, {}) out of XRange [{}, {}] YRange [{}, {}]"
                  .format(oneData[0], oneData[1], self.minx, self.maxx, self.miny, self.maxy))
               continue
            nearPointList = ParkerMethondBase.GetNearPoint(self, tempx, tempy, self.xyGravityIndexMap)
            if 0 == len(nearPointList):
               logger.error("MUST BUG2 depth ({}, {}) not find near point" .format(oneData[0], oneData[1]))
               return -1
            if oneData[0] not in xySeismDataMap.keys():
               xySeismDataMap[oneData[0]] = dict()
            if oneData[1] not in xySeismDataMap[oneData[0]].keys():
               xySeismDataMap[oneData[0]][oneData[1]] = list()
            else:
               continue
            xySeismDataMap[oneData[0]][oneData[1]].append(float(oneData[2]))
            xySeismDataMap[oneData[0]][oneData[1]].append(nearPointList)
            xySeismSize += 1
      return xySeismDataMap, xySeismSize

    def CalcRmsWithSeism(self, finalTarget, semisDataType):
      rms = 0.0
      if SemisDataType.TEST == semisDataType:
         xySeismDataMap = self.xySeismTestDataMap
         dataSize = self.testDataSize
      elif SemisDataType.VERIFY == semisDataType:
         xySeismDataMap = self.xySeismVerifyDataMap
         dataSize = self.verifyDataSize
      else:
         xySeismDataMap = self.xySeismAllDataMap
         dataSize = self.allDataSize
      if CalcType.SINGLE_CPU == self.calcType:
          seismDepthArray = np.zeros(dataSize)
          seismDepthArray1 = np.zeros(dataSize)
          seismDepthArray2 = np.zeros(dataSize)
      else:
          seismDepthArray = torch.zeros(dataSize)
          seismDepthArray1 = torch.zeros(dataSize)
          seismDepthArray2 = torch.zeros(dataSize)
      index = 0
      for x in xySeismDataMap.keys():
          for y, info in xySeismDataMap[x].items():
              seismDepth = info[0]
              sum0, sum1 = 0.0, 0.0
              for one in info[1]:
                  sum0 += finalTarget[one[1]] / one[0]
                  sum1 += 1.0 / one[0]
              averDepth = sum0 / sum1
              seismDepthArray[index] = seismDepth - averDepth
              seismDepthArray1[index] = seismDepth
              seismDepthArray2[index] = averDepth
              index = index + 1
      minType = 0
      if CalcType.SINGLE_CPU == self.calcType:
          minType = np.min(finalTarget[:])
          if minType < 0:
            minType = abs(minType) * punishRate
          else:
            minType = 0
          originRms = np.sqrt(np.mean(seismDepthArray ** 2))
          rms = originRms + minType
          maxNum = np.max(seismDepthArray)
          minNum = np.min(seismDepthArray)
          meanNum = np.mean(seismDepthArray)
          stdNum = np.std(seismDepthArray)
          coffMat = np.corrcoef(seismDepthArray1, seismDepthArray2)
          coffNum = coffMat[0][1]
      else:
          minType = torch.min(finalTarget[:])
          if minType < 0:
            minType = abs(minType) * punishRate
          else:
            minType = 0
          originRms = torch.sqrt(torch.mean(seismDepthArray ** 2))
          rms = originRms + minType
          rms = rms.cpu()
          originRms = originRms.cpu()
          maxNum = torch.max(seismDepthArray)
          maxNum = maxNum.cpu()
          minNum = torch.min(seismDepthArray)
          minNum = minNum.cpu()
          meanNum = torch.mean(seismDepthArray)
          meanNum = meanNum.cpu()
          stdNum = torch.std(seismDepthArray)
          stdNum = stdNum.cpu()
          coffMat = np.corrcoef(seismDepthArray1, seismDepthArray2)
          coffNum = coffMat[0][1]
      return float(rms), float(maxNum), float(minNum), float(meanNum), float(stdNum), float(coffNum), float(originRms)

    def SaveTopoData(self, finalTarget, filename):
      alldatas = []
      with open(self.gravityinputpath) as fd:
         index = 0
         for data in fd.readlines():
            data = data.strip().replace('\t', ' ').strip()
            data = data.replace('    ', ' ').strip()
            data = data.replace('   ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            oneData = data.split(' ')
            if len(oneData) != 3:
               logger.error("MUST BUG1 gravity {} size != 3".format(data))
               return -1
            alldatas.append(str(oneData[0] + " " + oneData[1] + " " + str(finalTarget[index][0]) + "\n"))
            index += 1
      with open(filename, "w+") as fd:
         for data in alldatas:
            fd.write(data)
      logger.info("SaveTopoData {} succ".format(filename))

    def CalcCRUSTData(self, crustDataPath, semisDataType):
      xyCrustMap = dict()
      minx, maxx = 1e10, -1
      miny, maxy = 1e10, -1
      with open(crustDataPath) as fd:
         alldatas = fd.readlines()
         for data in alldatas:
            data = data.strip().replace('\t', ' ').strip()
            data = data.replace('    ', ' ').strip()
            data = data.replace('   ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            oneData = [float(i) for i in data.split(' ')]
            if 0 == len(oneData):
               continue
            if len(oneData) != 3:
               logger.error("MUST BUG1 Crust {} size != 3".format(data))
               continue
            if oneData[0] not in xyCrustMap.keys():
               xyCrustMap[oneData[0]] = dict()
            xyCrustMap[oneData[0]][oneData[1]] = oneData[2]
            minx = min(minx, oneData[0])
            maxx = max(maxx, oneData[0])
            miny = min(miny, oneData[1])
            maxy = max(maxy, oneData[1])
      if SemisDataType.TEST == semisDataType:
         xySeismDataMap = self.xySeismTestDataMap
         dataSize = self.testDataSize
      elif SemisDataType.VERIFY == semisDataType:
         xySeismDataMap = self.xySeismVerifyDataMap
         dataSize = self.verifyDataSize
      else:
         xySeismDataMap = self.xySeismAllDataMap
         dataSize = self.allDataSize
      seismDepthArray1 = list()
      seismDepthArray2 = list()
      for x in xySeismDataMap.keys():
        for y, info in xySeismDataMap[x].items():
          seismDepth = info[0]
          if x < minx or x > maxx or y < miny or y > maxy:
            logger.error("MUST BUG2 Crust ({}, {}) out of XRange [{}, {}] YRange [{}, {}]"
               .format(x, y, minx, maxx, miny, maxy))
            continue
          nearPointList = ParkerMethondBase.GetNearPoint(self, x, y, xyCrustMap)
          if 0 == len(nearPointList):
            continue
          sum0, sum1 = 0.0, 0.0
          for nearpoint in nearPointList:
            crestDepth = nearpoint[1]
            sum0 += crestDepth / nearpoint[0]
            sum1 += 1.0 / nearpoint[0]
          averDepth = sum0 / sum1
          seismDepthArray1.append(seismDepth)
          seismDepthArray2.append(averDepth)
      seismDepthArray1 = np.array(seismDepthArray1)
      seismDepthArray2 = np.array(seismDepthArray2)
      seismDepthArray = seismDepthArray1 - seismDepthArray2
      rms = np.sqrt(np.mean(seismDepthArray ** 2))
      maxNum = np.max(seismDepthArray)
      minNum = np.min(seismDepthArray)
      meanNum = np.mean(seismDepthArray)
      stdNum = np.std(seismDepthArray)
      coffMat = np.corrcoef(seismDepthArray1, seismDepthArray2)
      coffNum = coffMat[0][1]
      logger.info('CalcCRUSTData Size{} semisDataType {} RmsWithSeism {:.8f} Max {:.8f} Min {:.8f} Mean {:.8f} Std {:.8f} Coff {:.8f}'
          .format(seismDepthArray.shape, semisDataType.name, rms, maxNum, minNum, meanNum, stdNum, coffNum))
      return float(rms), float(maxNum), float(minNum), float(meanNum), float(stdNum), float(coffNum)

    def GetPositionData(self, filePath):
      xySeismDataMap = dict()
      xySeismSize = 0
      with open(filePath) as fd:
         alldatas = fd.readlines()
         for data in alldatas:
            data = data.strip().replace('\t', ' ').strip()
            data = data.strip().replace(',', ' ').strip()
            data = data.replace('    ', ' ').strip()
            data = data.replace('   ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            data = data.replace('  ', ' ').strip()
            oneData = [float(i) for i in data.split(' ')]
            if 0 == len(oneData):
               continue
            if len(oneData) != 3:
               logger.error("MUST BUG1 seism {} size != 3".format(data))
               return -1
            if oneData[0] not in xySeismDataMap.keys():
               xySeismDataMap[oneData[0]] = dict()
            if oneData[1] not in xySeismDataMap[oneData[0]].keys():
               xySeismDataMap[oneData[0]][oneData[1]] = float(oneData[2])
            else:
               continue
            xySeismSize += 1
      return xySeismDataMap, xySeismSize

    def FilterSemisMat(self, contrast, z0, semisDataType, filterRate = 0.2):
      if 0 == filterRate:
        logger.info("No Need filterRate {}".format(filterRate))
        return 0
      if filterRate < 0 or filterRate >= 1.0:
        logger.error("MUST BUG1 filterRate {} is invaild".format(filterRate))
        return -1
      fileName = 'result/finalTarget_' + str(contrast) + '_' + str(z0) + '_' + semisDataType.name + '.dat'
      xyTargetAllDataMap, allTargetDataSize = self.GetPositionData(fileName)
      if len(xyTargetAllDataMap) <= 0 or allTargetDataSize <=0:
        logger.error("MUST BUG2 xyTargetAllDataMap is empty")
        return -1
      xySeismDataMap, allDataSize = self.GetPositionData(self.seisminputpath)
      if len(xySeismDataMap) <= 0 or allDataSize <=0:
        logger.error("MUST BUG3 xySeismDataMap is empty")
        return -1
      targetSize = (int)(allDataSize * filterRate)
      if targetSize >= allDataSize:
         logger.error("MUST BUG4 targetSize {} > allDataSize {}".format(targetSize, allDataSize))
         return -1
      diffList = list()
      index = 0
      for x in xySeismDataMap.keys():
        for y, seismDepth in xySeismDataMap[x].items():
          nearPointList = ParkerMethondBase.GetNearPoint(self, x, y, xyTargetAllDataMap)
          sum0, sum1 = 0.0, 0.0
          for nearpoint in nearPointList:
            crestDepth = nearpoint[1]
            sum0 += crestDepth / nearpoint[0]
            sum1 += 1.0 / nearpoint[0]
          averDepth = sum0 / sum1
          diffList.append([abs(seismDepth - averDepth), index])
          index += 1
      diffList.sort(reverse = True,key = lambda x:x[0])
      indexSet = set()
      for i in range(targetSize):
         indexSet.add(diffList[i][1])
      with open(self.allDataPath, 'w+') as fd:
         index = 0
         for x in xySeismDataMap.keys():
           for y, seismDepth in xySeismDataMap[x].items():
             if index not in indexSet:
               fd.write(str(x) + " " + str(y) + " " + str(seismDepth) + "\n")
             else:
               logger.info("{} {} {} DepthDiff {} will be ignored".format(x, y, seismDepth, diffList[index][0]))
             index += 1
      return 0

def cmp(x1, x2):
   if x1[1] != x2[1]:
      return 1 if x1[1] < x2[1] else -1
   else:
      return -1 if x1[0] < x2[0] else 1
def GetMeanGravity(inputFileName):
    xyGravityMap = dict()
    with open(inputFileName) as fd:
      alldatas = fd.readlines()
      for data in alldatas:
         data = data.strip().replace('\t', ' ').strip()
         data = data.replace('    ', ' ').strip()
         data = data.replace('   ', ' ').strip()
         data = data.replace('  ', ' ').strip()
         data = data.replace('  ', ' ').strip()
         data = data.replace('  ', ' ').strip()
         oneData = [float(i) for i in data.split(' ')]
         if 0 == len(oneData):
            logger.error("MUST BUG1 gravity {} size != 3".format(data))
            continue
         if len(oneData) != 3:
            logger.error("MUST BUG2 gravity {} size != 3".format(data))
            return -1
         if oneData[0] not in xyGravityMap.keys():
            xyGravityMap[oneData[0]] = dict()
         xyGravityMap[oneData[0]][oneData[1]] = oneData[2]
    xList = list(xyGravityMap.keys())
    xList.sort()
    if len(xList) <= 0:
      logger.error("MUST BUG2 xList is empty")
      return -1
    yList = list(xyGravityMap[xList[-1]].keys())
    yList.sort()
    if len(yList) <= 0:
      logger.error("MUST BUG2 yList is empty")
      return -1
    outputList = list()
    depthSum = 0.0
    for i in range(int(len(xList) / 3)):
      for j in range(int(len(yList) / 3)):
         midX = i * 3
         midY = j * 3
         averDepth = 0
         averDepth += xyGravityMap[xList[midX]][yList[midY]]
         averDepth += xyGravityMap[xList[midX + 1]][yList[midY]]
         averDepth += xyGravityMap[xList[midX + 2]][yList[midY]]
         averDepth += xyGravityMap[xList[midX]][yList[midY + 1]]
         averDepth += xyGravityMap[xList[midX + 1]][yList[midY + 1]]
         averDepth += xyGravityMap[xList[midX + 2]][yList[midY + 1]]
         averDepth += xyGravityMap[xList[midX]][yList[midY + 2]]
         averDepth += xyGravityMap[xList[midX + 1]][yList[midY + 2]]
         averDepth += xyGravityMap[xList[midX + 2]][yList[midY + 2]]
         averDepth = averDepth / 9.0
         outputList.append([xList[midX + 1], yList[midY + 1], averDepth])
         depthSum += averDepth
    averDepthSum = depthSum / len(outputList)
    outputList.sort(key=functools.cmp_to_key(cmp))
    with open(inputFileName + "-mean.txt", 'w+') as fd:
        for line in outputList:
           fd.write(str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n")