import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



def exp_decay(alpha, decay_rate, iteration_num, min_value=0):
   alpha_d = alpha * np.exp(-decay_rate*iteration_num)
   return max(min_value, alpha_d)

def dump_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_json(file_path):
    with open(file_path, 'r') as f:
        input_meta = json.load(f)
    return input_meta




def Plot():
    
    def plot(plot_list, plot_names, nrows, ncols):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*6), facecolor='white')
        ax = ax.ravel()
        
        for idx, (values, name) in enumerate(zip(plot_list, plot_names)):
            ax[idx].plot(values, 'o', color='blue', markersize=2)
            ax[idx].set_title(name)
            plt.xlabel('timesteps', fontsize=18)
            
        return plt
    
    def stats_plot(stats_dict_or_path, max_cols=3, exclude_plots=[]):
        if type(stats_dict_or_path) == str:
            stats_dict = read_json(stats_dict_or_path)
        else:
            stats_dict = stats_dict_or_path
        plot_names = stats_dict.keys()
        ncols = max_cols
        nrows = int(np.ceil((len(plot_names)-len(exclude_plots))/ncols))
        plot_list = []
        plot_names = []
        
        for k, v in stats_dict.items():
            if k in exclude_plots:
                continue
            plot_names.append(k)
            plot_list.append(v)
            
        return plot(plot_list, plot_names, nrows, ncols)
    
    return stats_plot




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
