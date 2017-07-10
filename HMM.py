__author__ = 'PC-LiNing'

import Util
import numpy

class Hmm:
    """
    def __init__(self,m,n,a,b,pi):
        self.M = m
        self.N = n
        self.A = a
        self.B = b
        self.Pi = pi
    """
    def read_HMMmodel(self, hmm_file):
        m,n,a,b,pi = Util.ReadHmm(hmm_file)
        self.M = m
        self.N = n
        self.A = a
        self.B = b
        self.Pi = pi
        #alpha beta psi gamma xi about seq_o
        self.alpha = None
        self.beta = None
        self.psi = None
        self.gamma = None
        self.xi = None
        # stats
        self.seq_count = 0

    def Forward(self, T, seq_o):
        alpha = numpy.zeros(shape=(T+1,self.N+1),dtype=numpy.float32)
        # init t=1
        for i in range(1,self.N+1):
            alpha[1][i] = self.Pi[i]*self.B[i][seq_o[1]]
        # init t=2,3,...,T
        for t in range(1,T):
            for j in range(1,self.N+1):
                sum = 0.0
                for i in range(1,self.N+1):
                    sum += alpha[t][i]*self.A[i][j]

                alpha[t+1][j] = self.B[j][seq_o[t+1]] * sum

        # collect prob in t = T
        prob = 0;
        for i in range(1,self.N+1):
            prob += alpha[T][i]

        self.alpha = alpha
        return prob

    def Viterbi(self, T, seq_o):
        delta = numpy.zeros(shape=(T+1,self.N+1),dtype=numpy.float32)
        psi = numpy.zeros(shape=(T+1,self.N+1),dtype=numpy.int32)
        state_q = numpy.zeros(shape=(T+1,),dtype=numpy.int32)

        # init t=1 delta,psi
        for i in range(1,self.N+1):
            delta[1][i] = self.Pi[i]*self.B[i][seq_o[1]]
            psi[1][i] = 0

        # t=2,3,...T
        for t in range(2,T+1):
            for j in range(1,self.N+1):
                maxval = 0.0
                maxvalind = 1
                for i in range(1,self.N+1):
                    val = delta[t-1][i] * self.A[i][j]
                    if val > maxval:
                        maxval = val
                        maxvalind = i

                delta[t][j] = maxval * self.B[j][seq_o[t]]
                psi[t][j] = maxvalind

        # find path
        prob = 0.0
        state_q[T] = 1
        for i in range(1,self.N+1):
            if delta[T][i] > prob:
                prob = delta[T][i]
                state_q[T] = i
        for t in range(1,T)[::-1]:
            state_q[t] = psi[t+1][state_q[t+1]]

        self.beta = delta
        self.psi = psi
        return state_q

    def Backword(self,T,seq_o):
        beta = numpy.zeros(shape=(T+1,self.N+1),dtype=numpy.float32)
        # init t = T
        for i in range(1,self.N+1):
            beta[T][i]=1.0

        # fill beta
        for t in range(1,T)[::-1]:
            for i in range(1,self.N+1):
                sum = 0.0
                for j in range(1,self.N+1):
                    sum += self.A[i][j]*self.B[j][seq_o[t+1]]*beta[t+1][j]
                beta[t][i] = sum

        # result
        prob = 0.0
        for i in range(1,self.N+1):
            prob += beta[1][i]

        self.beta = beta
        return prob

    # util func
    def ComputeGamma(self,T):
        for t in range(1,T+1):
            den = 0.0
            for i in range(1,self.N+1):
                self.gamma[t][i] = self.alpha[t][i]*self.beta[t][i]
                den += self.gamma[t][i]

            for i in range(1,self.N+1):
                self.gamma[t][i] = self.gamma[t][i] / den

    def ComputeXi(self,T,seq_o):
        for t in range(1,T):
            sum = 0.0
            for i in range(1,self.N+1):
                for j in range(1,self.N+1):
                    self.xi[t][i][j] = self.alpha[t][i]*self.A[i][j]*self.B[j][seq_o[t+1]]*self.beta[t+1][j]
                    sum += self.xi[t][i][j]

            for i in range(1,self.N+1):
                for j in range(1,self.N+1):
                    self.xi[t][i][j] = self.xi[t][i][j] / sum

    def update(self,T,seq_o):
        # pi
        for i in range(1,self.N+1):
            self.Pi[i] = 0.001 + 0.999 * self.gamma[1][i]
        # a b
        for i in range(1,self.N+1):
            denA = 0.0
            for t in range(1,T):
                denA += self.gamma[t][i]
            for j in range(1,self.N+1):
                numA = 0.0
                for t in range(1,T):
                    numA += self.xi[t][i][j]
                self.A[i][j] = 0.001 + 0.999*numA / denA
            denB = denA + self.gamma[t][i]
            for k in range(1,self.M+1):
                numB = 0.0
                for t in range(1,T+1):
                    if seq_o[t] == k:
                        numB += self.gamma[t][i]
                self.B[i][k] = 0.001 + 0.999*numB/denB

    def Train_one_seq(self,T,seq_o):
        prob_f = self.Forward(T,seq_o)
        print("init forward prob: "+str(prob_f))
        self.Backword(T,seq_o)
        self.ComputeGamma(T)
        self.ComputeXi(T,seq_o)
        prob_prev = prob_f
        DELAT = 0.0001
        delta = 0.1
        iter = 0
        while delta > DELAT:
            self.update(T,seq_o)
            prob_f = self.Forward(T,seq_o)
            print(str(iter+1)+",current forward prob: "+str(prob_f))
            self.Backword(T,seq_o)
            self.ComputeGamma(T)
            self.ComputeXi(T,seq_o)
            delta = prob_f - prob_prev
            prob_prev = prob_f
            iter += 1
        print("final forward prob: "+str(prob_f))

    def EstHmm(self,seq_list):
        for one in seq_list:
            T = one[0]
            self.gamma = numpy.zeros(shape=(T+1,self.N+1),dtype=numpy.float32)
            self.xi = numpy.zeros(shape=(T+1,self.N+1,self.N+1),dtype=numpy.float32)
            self.Train_one_seq(one[0],one[1])

        print("Train end.")
        print("HMM Model:")
        print("N="+str(self.N)+","+"M="+str(self.M))
        print("A:")
        print(self.A[1:,1:])
        print("B:")
        print(self.B[1:,1:])
        print("Pi:")
        print(self.Pi[1:])













