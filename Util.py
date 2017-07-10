__author__ = 'PC-LiNing'

import codecs
import numpy


# read hmm model
def  ReadHmm(hmm_file):
    f = codecs.open(hmm_file,encoding='utf8')
    m = int(f.readline().replace("M=",""))
    n = int(f.readline().replace("N=",""))
    f.readline()
    a = numpy.zeros(shape=(n+1,n+1),dtype=numpy.float32)
    for i in range(1,n+1):
        line = f.readline().strip('\n').strip().split(" ")
        a[i] = numpy.asarray([0.0]+[float(item) for item in line],dtype=numpy.float32)

    f.readline()
    b = numpy.zeros(shape=(n+1, m+1), dtype=numpy.float32)
    for i in range(1,n+1):
        line = f.readline().strip('\n').strip().split(" ")
        b[i] = numpy.asarray([0.0]+[float(item) for item in line],dtype=numpy.float32)
    f.readline()
    line = f.readline().strip('\n').strip().split(" ")
    pi = numpy.asarray([0]+[float(item) for item in line], dtype=numpy.float32)
    return m,n,a,b,pi


# read Observation squence
def ReadObS(obs_file):
    f = codecs.open(obs_file,encoding='utf8')
    t = int(f.readline().replace("T=",""))
    line = f.readline().strip('\n').strip().split(" ")
    seq_o = numpy.asarray([0.0]+[float(item) for item in line], dtype=numpy.float32)
    return t,seq_o

