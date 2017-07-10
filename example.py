__author__ = 'PC-LiNing'
from HMM import Hmm
import Util


hmm = Hmm()
hmm.read_HMMmodel('test.hmm')
T, seq_o = Util.ReadObS('test.seq')
prob = hmm.Forward(T,seq_o)
state_q = hmm.Viterbi(T,seq_o)
print(prob)
print(state_q)
seq_list = [(T,seq_o)]
hmm.EstHmm(seq_list)
print()
