# -*- coding: utf-8 -*-  
import sys
import os
import math
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import enum
from CommonDefine import *
from numpy import fft
from scipy import signal
from datetime import datetime
from loguru import logger
from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

class ParkerMethondFitness(ParkerMethondBase):
    def InitPreCalc(self):
      if CalcType.SINGLE_CPU != self.calcType:
         logger.error("MUST BUG calcType {} is Wrong".format(self.calcType.name))
         return -1
      if 0 != ParkerMethondBase.InitPreCalcBase(self):
         logger.error("MUST BUG3 InitPreCalcBase failed")
         return -1
      self.preCalcFlag = True
      return 0
    def CalcFitness(self, contrast, z0, sequenceId = 0, needOutputResult = False, semisDataType = SemisDataType.TEST):
      if not self.preCalcFlag:
         logger.error('CalcFitness Not InitPreCalc MUST BUG')
         return -1
      up =- self.fftbou * (np.exp((z0 * 100000) * (self.frequencytotal * (1 / 100000))))
      down = 2 * np.pi * 6.67 * contrast
      constant = up / down
      constant = constant * self.filter
      topoinverse = np.real(fft.ifft2(constant))
      topo2 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse ** 2)))
      topo2 = topo2 * self.filter
      topo2 = constant - topo2
      topoinverse2 = np.real(fft.ifft2(topo2))
      diference2 = topoinverse2 - topoinverse
      diference2 = diference2 ** 2
      rms2 = np.sqrt(np.sum(np.sum(diference2)) / ( 2 * (self.numrows * self.numcolumns)))
      iter = 2
      rms = rms2
      finaltopoinverse = topoinverse2
      if rms2 >= self.criterio:
        topo3 = (((self.frequencytotal ** (1)) / prod(1, 2)) * fft.fft2(finaltopoinverse ** 2)) + \
            ((((self.frequencytotal ** (2)) / prod(1, 3)) * (fft.fft2(finaltopoinverse ** 3))))
        topo3 = topo3 * self.filter
        topo3 = constant - topo3
        topoinverse3 = np.real(fft.ifft2(topo3))
        diference3 = topoinverse3 - topoinverse2
        diference3 = diference3 ** 2
        rms3 = np.sqrt(np.sum(np.sum(diference3)) / (2 * self.numrows * self.numcolumns))
        if rms3 <= rms2:
          rms = rms3
          finaltopoinverse = topoinverse3 
          iter = 3
        if rms3 > rms2 and rms3 >= self.criterio:
          topo4 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse3 ** 2))) + \
              ((((self.frequencytotal ** (2)) / (prod(1, 3))) * (fft.fft2(topoinverse3 ** 3)))) +     \
              ((((self.frequencytotal ** (3)) / (prod(1, 4))) * (fft.fft2(topoinverse3 ** 4))))
          topo4 = topo4 * self.filter
          topo4 = constant-topo4
          topoinverse4 = np.real(fft.ifft2(topo4))
          diference4 = topoinverse4 - topoinverse3
          diference4 = diference4 ** 2
          rms4 = np.sqrt(np.sum(np.sum(diference4)) / (2 * self.numrows * self.numcolumns))
          if rms4 <= rms3:
            rms = rms4
            finaltopoinverse = topoinverse4
            iter = 4
          if rms4 > rms3 and rms4 >= self.criterio:
            topo5 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse4 ** 2))) + \
                ((((self.frequencytotal ** (2)) / (prod(1, 3))) * (fft.fft2(topoinverse4 ** 3)))) +   \
                ((((self.frequencytotal ** (3)) / (prod(1, 4))) * (fft.fft2(topoinverse4 ** 4)))) +   \
                ((((self.frequencytotal ** (4)) / (prod(1, 5))) * (fft.fft2(topoinverse4 ** 5))))
            topo5 = topo5 * self.filter
            topo5 = constant - topo5
            topoinverse5 = np.real(fft.ifft2(topo5))
            diference5 = topoinverse5-topoinverse4
            diference5 = diference5 ** 2
            rms5 = np.sqrt(np.sum(np.sum(diference5)) / (2 * self.numrows * self.numcolumns))
            if rms5 <= rms4:
              rms = rms5
              finaltopoinverse = topoinverse5
              iter = 5
            if rms5 > rms4 and rms5 >= self.criterio:
              topo6 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse5 ** 2))) + \
                  ((((self.frequencytotal ** (2)) / (prod(1, 3))) * (fft.fft2(topoinverse5 ** 3)))) +   \
                  ((((self.frequencytotal ** (3)) / (prod(1, 4))) * (fft.fft2(topoinverse5 ** 4)))) +   \
                  ((((self.frequencytotal ** (4)) / (prod(1, 5))) * (fft.fft2(topoinverse5 ** 5)))) +   \
                  ((((self.frequencytotal ** (5)) / (prod(1, 6))) * (fft.fft2(topoinverse5 ** 6))))
              topo6 = topo6 * self.filter
              topo6 = constant-topo6
              topoinverse6 = np.real(fft.ifft2(topo6))
              diference6 = topoinverse6-topoinverse5
              diference6 = diference6 ** 2
              rms6 = np.sqrt(np.sum(np.sum(diference6)) / (2 * self.numrows * self.numcolumns))
              if rms6 <= rms5:
                rms = rms6
                finaltopoinverse = topoinverse6
                iter = 6
              if  rms6 > rms5 and rms6 >= self.criterio:
                topo7 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse6 ** 2))) + \
                    ((((self.frequencytotal ** (2)) / (prod(1, 3))) * (fft.fft2(topoinverse6 ** 3)))) +   \
                    ((((self.frequencytotal ** (3)) / (prod(1, 4))) * (fft.fft2(topoinverse6 ** 4)))) +   \
                    ((((self.frequencytotal ** (4)) / (prod(1, 5))) * (fft.fft2(topoinverse6 ** 5)))) +   \
                    ((((self.frequencytotal ** (5)) / (prod(1, 6))) * (fft.fft2(topoinverse6 ** 6)))) +   \
                    ((((self.frequencytotal ** (6)) / (prod(1, 7))) * (fft.fft2(topoinverse6 ** 7))))
                topo7 = topo7 * self.filter
                topo7 = constant-topo7
                topoinverse7 = np.real(fft.ifft2(topo7))
                diference7 = topoinverse7 - topoinverse6
                diference7 = diference7 ** 2
                rms7 = np.sqrt(np.sum(np.sum(diference7)) / (2 * self.numrows * self.numcolumns))
                if rms7 <= rms6:
                  rms = rms7
                  finaltopoinverse = topoinverse7
                  iter = 7
                if rms7 > rms6 and rms7 >= self.criterio:
                  topo8 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse7 ** 2))) + \
                      ((((self.frequencytotal ** (2)) / (prod(1, 3))) * (fft.fft2(topoinverse7 ** 3)))) + \
                      ((((self.frequencytotal ** (3)) / (prod(1, 4))) * (fft.fft2(topoinverse7 ** 4)))) + \
                      ((((self.frequencytotal ** (4)) / (prod(1, 5))) * (fft.fft2(topoinverse7 ** 5)))) + \
                      ((((self.frequencytotal ** (5)) / (prod(1, 6))) * (fft.fft2(topoinverse7 ** 6)))) + \
                      ((((self.frequencytotal ** (6)) / (prod(1, 7))) * (fft.fft2(topoinverse7 ** 7)))) + \
                      ((((self.frequencytotal ** (7)) / (prod(1, 8))) * (fft.fft2(topoinverse7 ** 8))))
                  topo8 = topo8 * self.filter
                  topo8 = constant - topo8
                  topoinverse8 = np.real(fft.ifft2(topo8))
                  diference8 = topoinverse8 - topoinverse7
                  diference8 = diference8 ** 2
                  rms8 = np.sqrt(np.sum(np.sum(diference8)) / (2 * self.numrows * self.numcolumns))
                  if rms8 <= rms7:
                    rms = rms8
                    finaltopoinverse = topoinverse8
                    iter = 8
                  if rms8 > rms7 and rms8 >= self.criterio:
                    topo9 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse8 ** 2))) + \
                        ((((self.frequencytotal ** (2)) / (prod(1, 3))) * (fft.fft2(topoinverse8 ** 3)))) +   \
                        ((((self.frequencytotal ** (3)) / (prod(1, 4))) * (fft.fft2(topoinverse8 ** 4)))) +   \
                        ((((self.frequencytotal ** (4)) / (prod(1, 5))) * (fft.fft2(topoinverse8 ** 5)))) +   \
                        ((((self.frequencytotal ** (5)) / (prod(1, 6))) * (fft.fft2(topoinverse8 ** 6)))) +   \
                        ((((self.frequencytotal ** (6)) / (prod(1, 7))) * (fft.fft2(topoinverse8 ** 7)))) +   \
                        ((((self.frequencytotal ** (7)) / (prod(1, 8))) * (fft.fft2(topoinverse8 ** 8)))) +   \
                        ((((self.frequencytotal ** (8)) / (prod(1, 9))) * (fft.fft2(topoinverse8 ** 9))))
                    topo9 = topo9 * self.filter
                    topo9 = constant-topo9
                    topoinverse9 = np.real(fft.ifft2(topo9))
                    diference9 = topoinverse9-topoinverse8
                    diference9 = diference9 ** 2
                    rms9 = np.sqrt(np.sum(np.sum(diference9)) / (2 * self.numrows * self.numcolumns))
                    if rms9 <= rms8:
                      rms = rms9
                      finaltopoinverse=topoinverse9
                      iter = 9
                      if rms9 > rms8 and rms9 >= self.criterio:
                        topo10 = (((self.frequencytotal ** (1)) / (prod(1, 2))) * (fft.fft2(topoinverse9 ** 2))) + \
                        ((((self.frequencytotal ** (2)) / (prod(1, 3))) * (fft.fft2(topoinverse9 ** 3)))) +        \
                        ((((self.frequencytotal ** (3)) / (prod(1, 4))) * (fft.fft2(topoinverse9 ** 4)))) +        \
                        ((((self.frequencytotal ** (4)) / (prod(1, 5))) * (fft.fft2(topoinverse9 ** 5)))) +        \
                        ((((self.frequencytotal ** (5)) / (prod(1, 6))) * (fft.fft2(topoinverse9 ** 6)))) +        \
                        ((((self.frequencytotal ** (6)) / (prod(1, 7))) * (fft.fft2(topoinverse9 ** 7)))) +        \
                        ((((self.frequencytotal ** (7)) / (prod(1, 8))) * (fft.fft2(topoinverse9 ** 8)))) +        \
                        ((((self.frequencytotal ** (8)) / (prod(1, 9))) * (fft.fft2(topoinverse9 ** 9)))) +        \
                        ((((self.frequencytotal ** (9)) / (prod(1, 10))) * (fft.fft2(topoinverse9 ** 10))))
                        topo10 = topo10 * self.filter
                        topo10 = constant - topo10
                        topoinverse10 = np.real(fft.ifft2(topo10))
                        diference10 = topoinverse10 - topoinverse9
                        diference10 = diference10 ** 2
                        rms10 = np.sqrt(np.sum(np.sum(diference10)) / (2 * self.numrows * self.numcolumns))
                        if rms10 <= rms9:
                          rms = rms10
                          finaltopoinverse = topoinverse10
                          iter=10
      acutalTarget = finaltopoinverse + z0
      trans = acutalTarget.conjugate().T
      finalTarget = trans.reshape(self.numrows * self.numcolumns, -1)
      thisRms, maxNum, minNum, meanNum, stdNum, coffNum, originRms = ParkerMethondBase.CalcRmsWithSeism(self, finalTarget, semisDataType)
      hasImprove = False
      if thisRms < self.lastRmsFitness:
        self.lastRmsFitness = thisRms
        hasImprove = True
      logger.info('CalcFitness contrast {:.8f} z0 {:.8f} semisDataType {} SequenceId {} iterCnt {} IterRms {:.8f} RmsWithSeism {:.8f} originRms {:.8f} Max {:.8f} Min {:.8f} Mean {:.8f} Std {:.8f} Coff {:.8f}' \
        .format(contrast, z0, semisDataType.name, sequenceId, iter, rms, thisRms, originRms, maxNum, minNum, meanNum, stdNum, coffNum))
      if InterDataExportType.ALL == outputInterDataType:
        np.savetxt('result/temp/finalTarget.dat' + str(sequenceId), finalTarget)
      elif InterDataExportType.INCER == outputInterDataType and hasImprove == True:
        np.savetxt('result/temp/finalTarget.dat' + str(sequenceId), finalTarget)
      else:
        pass
      if needOutputResult:
        fileName = 'result/finalTarget_' + str(contrast) + '_' + str(z0) + '_' + semisDataType.name + '.dat'
        ParkerMethondBase.SaveTopoData(self, finalTarget, fileName)
        logger.info('CalcFitness contrast {:.8f} z0 {:.8f} semisDataType {} SequenceId {} iterCnt {} IterRms {:.8f} RmsWithSeism {:.8f} originRms {:.8f} Max {:.8f} Min {:.8f} Mean {:.8f} Std {:.8f} Coff {:.8f}' \
          .format(contrast, z0, semisDataType.name, sequenceId, iter, rms, thisRms, originRms, maxNum, minNum, meanNum, stdNum, coffNum))
      sys.stdout.flush()
      return thisRms, originRms