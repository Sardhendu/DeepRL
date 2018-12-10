
import numpy as np

def exp_decay(alpha, decay_rate, iteration_num, min_value=0):
   alpha_d = alpha * np.exp(-decay_rate*iteration_num)
   return max(min_value, alpha_d)



def debug():
    TAU = 0.1
    decay_rate = 0.003
    TAU_MIN = 0.01
    a_ = []
    for i in range(0, 2000):
        a = exp_decay(TAU, decay_rate, i, TAU_MIN)
        a_.append(a)
        
    import matplotlib.pyplot as plt
    plt.plot(a_)
    plt.show()
    
# debug()