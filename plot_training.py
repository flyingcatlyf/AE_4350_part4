import numpy as np
import matplotlib.pyplot as plt
import matplotlib



#training_data_DDPG-SCA
score_history_training_ddpg_SCA_1 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sca/1_training/scorehistory_training')
score_history_training_ddpg_SCA_2 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sca/2_training/scorehistory_training')
score_history_training_ddpg_SCA_3 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sca/3_training/scorehistory_training')
score_history_training_ddpg_SCA_4 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sca/4_training/scorehistory_training')
score_history_training_ddpg_SCA_5 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sca/5_training/scorehistory_training')

#training_data_DDPG-SDA
score_history_training_ddpg_SDA_1 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sda/1_training/scorehistory_training')
score_history_training_ddpg_SDA_2 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sda/6_training/scorehistory_training')
score_history_training_ddpg_SDA_3 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sda/3_training/scorehistory_training')
score_history_training_ddpg_SDA_4 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sda/4_training/scorehistory_training')
score_history_training_ddpg_SDA_5 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg_sda/2_training/scorehistory_training')
#score_history_training_ddpg_SDA_6 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sda_betainreward/6_training/scorehistory_training')
#score_history_training_ddpg_SDA_7 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sda_betainreward/7_training/scorehistory_training')

#training_data_DDPG
score_history_training_ddpg_1 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg/1_training/scorehistory_training')
score_history_training_ddpg_2 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg/2_training/scorehistory_training')
score_history_training_ddpg_3 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg/3_training/scorehistory_training')
score_history_training_ddpg_4 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg/4_training/scorehistory_training')
score_history_training_ddpg_5 = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Training&Operation/Data_ddpg/5_training/scorehistory_training')

#calculate average return
#DDPG-SCA
Ave_return_history_ddpg_SCA_1=[]
Ave_return_history_ddpg_SCA_2=[]
Ave_return_history_ddpg_SCA_3=[]
Ave_return_history_ddpg_SCA_4=[]
Ave_return_history_ddpg_SCA_5=[]
for i in range(1,3001):
    rollout_1 = score_history_training_ddpg_SCA_1[:i]
    rollout_2 = score_history_training_ddpg_SCA_2[:i]
    rollout_3 = score_history_training_ddpg_SCA_3[:i]
    rollout_4 = score_history_training_ddpg_SCA_4[:i]
    rollout_5 = score_history_training_ddpg_SCA_5[:i]

    return_average_ddpg_SCA_1 = np.mean(rollout_1[-100:])
    return_average_ddpg_SCA_2 = np.mean(rollout_2[-100:])
    return_average_ddpg_SCA_3 = np.mean(rollout_3[-100:])
    return_average_ddpg_SCA_4 = np.mean(rollout_4[-100:])
    return_average_ddpg_SCA_5 = np.mean(rollout_5[-100:])

    Ave_return_history_ddpg_SCA_1.append(return_average_ddpg_SCA_1)
    Ave_return_history_ddpg_SCA_2.append(return_average_ddpg_SCA_2)
    Ave_return_history_ddpg_SCA_3.append(return_average_ddpg_SCA_3)
    Ave_return_history_ddpg_SCA_4.append(return_average_ddpg_SCA_4)
    Ave_return_history_ddpg_SCA_5.append(return_average_ddpg_SCA_5)

Ave_return_history_ddpg_SCA_sum = np.vstack((Ave_return_history_ddpg_SCA_1,Ave_return_history_ddpg_SCA_2,Ave_return_history_ddpg_SCA_3,Ave_return_history_ddpg_SCA_4,Ave_return_history_ddpg_SCA_5))
#Ave_return_history_ddpg_SCA_mean = np.mean(Ave_return_history_ddpg_SCA_sum, axis=0)
Ave_return_history_ddpg_SCA_mean = Ave_return_history_ddpg_SCA_sum.mean(axis=0)
Ave_return_history_ddpg_SCA_max = Ave_return_history_ddpg_SCA_sum.max(axis=0)
Ave_return_history_ddpg_SCA_min = Ave_return_history_ddpg_SCA_sum.min(axis=0)
Ave_return_history_ddpg_SCA_std = Ave_return_history_ddpg_SCA_sum.std(axis=0)
ss= Ave_return_history_ddpg_SCA_sum[:,0]

Ave_gradient_ddpg_SCA_500 = (Ave_return_history_ddpg_SCA_sum[:,499] - Ave_return_history_ddpg_SCA_sum[:,0])/500
Ave_gradient_ddpg_SCA_500_mean = Ave_gradient_ddpg_SCA_500.mean(axis=0)
Ave_gradient_ddpg_SCA_500_std = Ave_gradient_ddpg_SCA_500.std(axis=0)

Ave_gradient_ddpg_SCA_3000 = (Ave_return_history_ddpg_SCA_sum[:,2999] - Ave_return_history_ddpg_SCA_sum[:,2499])/500
Ave_gradient_ddpg_SCA_3000_mean = Ave_gradient_ddpg_SCA_3000.mean(axis=0)
Ave_gradient_ddpg_SCA_3000_std = Ave_gradient_ddpg_SCA_3000.std(axis=0)

# DDPG-SDA
Ave_return_history_ddpg_SDA_1 = []
Ave_return_history_ddpg_SDA_2 = []
Ave_return_history_ddpg_SDA_3 = []
Ave_return_history_ddpg_SDA_4 = []
Ave_return_history_ddpg_SDA_5 = []
for i in range(1, 3001):
    rollout_1 = score_history_training_ddpg_SDA_1[:i]
    rollout_2 = score_history_training_ddpg_SDA_2[:i]
    rollout_3 = score_history_training_ddpg_SDA_3[:i]
    rollout_4 = score_history_training_ddpg_SDA_4[:i]
    rollout_5 = score_history_training_ddpg_SDA_5[:i]

    return_average_ddpg_SDA_1 = np.mean(rollout_1[-100:])
    return_average_ddpg_SDA_2 = np.mean(rollout_2[-100:])
    return_average_ddpg_SDA_3 = np.mean(rollout_3[-100:])
    return_average_ddpg_SDA_4 = np.mean(rollout_4[-100:])
    return_average_ddpg_SDA_5 = np.mean(rollout_5[-100:])

    Ave_return_history_ddpg_SDA_1.append(return_average_ddpg_SDA_1)
    Ave_return_history_ddpg_SDA_2.append(return_average_ddpg_SDA_2)
    Ave_return_history_ddpg_SDA_3.append(return_average_ddpg_SDA_3)
    Ave_return_history_ddpg_SDA_4.append(return_average_ddpg_SDA_4)
    Ave_return_history_ddpg_SDA_5.append(return_average_ddpg_SDA_5)

Ave_return_history_ddpg_SDA_sum = np.vstack((Ave_return_history_ddpg_SDA_1,Ave_return_history_ddpg_SDA_2,Ave_return_history_ddpg_SDA_3,Ave_return_history_ddpg_SDA_4,Ave_return_history_ddpg_SDA_5))
Ave_return_history_ddpg_SDA_mean = Ave_return_history_ddpg_SDA_sum.mean(axis=0)
Ave_return_history_ddpg_SDA_max = Ave_return_history_ddpg_SDA_sum.max(axis=0)
Ave_return_history_ddpg_SDA_min = Ave_return_history_ddpg_SDA_sum.min(axis=0)
Ave_return_history_ddpg_SDA_std = Ave_return_history_ddpg_SDA_sum.std(axis=0)

Ave_gradient_ddpg_SDA_500 = (Ave_return_history_ddpg_SDA_sum[:,499] - Ave_return_history_ddpg_SDA_sum[:,0])/500
Ave_gradient_ddpg_SDA_500_mean = Ave_gradient_ddpg_SDA_500.mean(axis=0)
Ave_gradient_ddpg_SDA_500_std = Ave_gradient_ddpg_SDA_500.std(axis=0)

Ave_gradient_ddpg_SDA_3000 = (Ave_return_history_ddpg_SDA_sum[:,2999] - Ave_return_history_ddpg_SDA_sum[:,2499])/500
Ave_gradient_ddpg_SDA_3000_mean = Ave_gradient_ddpg_SDA_3000.mean(axis=0)
Ave_gradient_ddpg_SDA_3000_std = Ave_gradient_ddpg_SDA_3000.std(axis=0)
# DDPG
Ave_return_history_ddpg_1 = []
Ave_return_history_ddpg_2 = []
Ave_return_history_ddpg_3 = []
Ave_return_history_ddpg_4 = []
Ave_return_history_ddpg_5 = []
for i in range(1, 3001):
    rollout_1 = score_history_training_ddpg_1[:i]
    rollout_2 = score_history_training_ddpg_2[:i]
    rollout_3 = score_history_training_ddpg_3[:i]
    rollout_4 = score_history_training_ddpg_4[:i]
    rollout_5 = score_history_training_ddpg_5[:i]

    return_average_ddpg_1 = np.mean(rollout_1[-100:])
    return_average_ddpg_2 = np.mean(rollout_2[-100:])
    return_average_ddpg_3 = np.mean(rollout_3[-100:])
    return_average_ddpg_4 = np.mean(rollout_4[-100:])
    return_average_ddpg_5 = np.mean(rollout_5[-100:])

    Ave_return_history_ddpg_1.append(return_average_ddpg_1)
    Ave_return_history_ddpg_2.append(return_average_ddpg_2)
    Ave_return_history_ddpg_3.append(return_average_ddpg_3)
    Ave_return_history_ddpg_4.append(return_average_ddpg_4)
    Ave_return_history_ddpg_5.append(return_average_ddpg_5)

Ave_return_history_ddpg_sum = np.vstack((Ave_return_history_ddpg_1,Ave_return_history_ddpg_2,Ave_return_history_ddpg_3,Ave_return_history_ddpg_4,Ave_return_history_ddpg_5))
Ave_return_history_ddpg_mean = Ave_return_history_ddpg_sum.mean(axis=0)
Ave_return_history_ddpg_max = Ave_return_history_ddpg_sum.max(axis=0)
Ave_return_history_ddpg_min = Ave_return_history_ddpg_sum.min(axis=0)
Ave_return_history_ddpg_std = Ave_return_history_ddpg_sum.std(axis=0)

Ave_gradient_ddpg_500 = (Ave_return_history_ddpg_sum[:,499] - Ave_return_history_ddpg_sum[:,0])/500
Ave_gradient_ddpg_500_mean = Ave_gradient_ddpg_500.mean(axis=0)
Ave_gradient_ddpg_500_std = Ave_gradient_ddpg_500.std(axis=0)

Ave_gradient_ddpg_3000 = (Ave_return_history_ddpg_sum[:,2999] - Ave_return_history_ddpg_sum[:,2499])/500
Ave_gradient_ddpg_3000_mean = Ave_gradient_ddpg_3000.mean(axis=0)
Ave_gradient_ddpg_3000_std = Ave_gradient_ddpg_3000.std(axis=0)

Episode = np.array(range(0,3000,1))



#plot_training_baseline
fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Ave_return_history_ddpg_SCA_mean,linewidth=4.0,label='DDPG-SCA',color='C3')
plt.fill_between(Episode,Ave_return_history_ddpg_SCA_max,Ave_return_history_ddpg_SCA_min,color='C3',alpha=0.2)
plt.plot(Ave_return_history_ddpg_SDA_mean,linewidth=4.0,label='DDPG-SDA',color='C1')
plt.fill_between(Episode,Ave_return_history_ddpg_SDA_max,Ave_return_history_ddpg_SDA_min,color='C1',alpha=0.2)
plt.plot(Ave_return_history_ddpg_mean,linewidth=4.0,label='DDPG',color='C0')
plt.fill_between(Episode,Ave_return_history_ddpg_max,Ave_return_history_ddpg_min,alpha=0.2,color='C0')
plt.xlabel('Episode [-]',fontdict={'size':35})
plt.ylabel('Average return [-]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True)
plt.legend(loc='lower right',fontsize=35)
plt.ylim(-10000,0.1)
plt.yticks(np.arange(-10000, 0, 1000))
plt.yticks([-10000,-9000, -8000, -7000, -6000, -5000, -4000, -3000, -2000, -1000, 0], ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0'])
#plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
#plt.gca().ticklabel_format(useMathText=True)
plt.title(r'$ \times 10^{3}$', loc='left', fontsize = 30)



#Direct_mean_without_rollout
#DDPG-SCA
score_history_ddpg_SCA_sum = np.vstack((score_history_training_ddpg_SCA_1,score_history_training_ddpg_SCA_2,score_history_training_ddpg_SCA_3,score_history_training_ddpg_SCA_4,score_history_training_ddpg_SCA_5))
score_history_ddpg_SCA_mean = score_history_ddpg_SCA_sum.mean(axis=0)
score_history_ddpg_SCA_max = score_history_ddpg_SCA_sum.max(axis=0)
score_history_ddpg_SCA_min = score_history_ddpg_SCA_sum.min(axis=0)

#DDPG-SDA
score_history_ddpg_SDA_sum = np.vstack((score_history_training_ddpg_SDA_1,score_history_training_ddpg_SDA_2,score_history_training_ddpg_SDA_3,score_history_training_ddpg_SDA_4,score_history_training_ddpg_SDA_5))
score_history_ddpg_SDA_mean = score_history_ddpg_SDA_sum.mean(axis=0)
score_history_ddpg_SDA_max = score_history_ddpg_SDA_sum.max(axis=0)
score_history_ddpg_SDA_min = score_history_ddpg_SDA_sum.min(axis=0)

#DDPG
score_history_ddpg_sum = np.vstack((score_history_training_ddpg_1,score_history_training_ddpg_2,score_history_training_ddpg_3,score_history_training_ddpg_4,score_history_training_ddpg_5))
score_history_ddpg_mean = score_history_ddpg_sum.mean(axis=0)
score_history_ddpg_max = score_history_ddpg_sum.max(axis=0)
score_history_ddpg_min = score_history_ddpg_sum.min(axis=0)


#fig2= plt.figure(figsize=(18.0,9.0))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(score_history_ddpg_SCA_mean,linewidth=4.0,label='DDPG-SCA',color='C3')
#plt.fill_between(Episode,score_history_ddpg_SCA_max,score_history_ddpg_SCA_min,color='C3',alpha=0.2)
#plt.plot(score_history_ddpg_SDA_mean,linewidth=4.0,label='DDPG-SDA',color='C1')
#plt.fill_between(Episode,score_history_ddpg_SDA_max,score_history_ddpg_SDA_min,color='C1',alpha=0.2)
#plt.plot(score_history_ddpg_mean,linewidth=4.0,label='DDPG',color='C0')
#plt.fill_between(Episode,score_history_ddpg_max,score_history_ddpg_min,alpha=0.2,color='C0')
#plt.xlabel('Episode [-]',fontdict={'size':35})
#plt.ylabel('Average return [-]',fontdict={'size':35})
#plt.xticks(fontsize=35)
#plt.yticks(fontsize=35)
#plt.grid(True)
#plt.legend(loc='lower right',fontsize=35)
#plt.ylim(-10000,0.1)
#plt.yticks(np.arange(-10000, 0, 1000))
#plt.yticks([-10000,-9000, -8000, -7000, -6000, -5000, -4000, -3000, -2000, -1000, 0], ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0'])
#plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
#plt.gca().ticklabel_format(useMathText=True)
#plt.title(r'$ \times 10^{3}$', loc='left', fontsize = 30)

fig2= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(score_history_ddpg_SCA_mean,linewidth=4.0,label='DDPG-SCA',color='C3')
plt.fill_between(Episode,score_history_ddpg_SCA_max,score_history_ddpg_SCA_min,color='C3',alpha=0.2)
plt.plot(score_history_ddpg_SDA_mean,linewidth=4.0,label='DDPG-SDA',color='C1')
plt.fill_between(Episode,score_history_ddpg_SDA_max,score_history_ddpg_SDA_min,color='C1',alpha=0.2)
plt.plot(score_history_ddpg_mean,linewidth=4.0,label='DDPG',color='C0')
plt.fill_between(Episode,score_history_ddpg_max,score_history_ddpg_min,alpha=0.2,color='C0')
plt.xlabel('Episode [-]',fontdict={'size':35})
plt.ylabel('Average return [-]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True)
plt.legend(loc='lower right',fontsize=35)
plt.xlim(0,500)
plt.ylim(-10000,0.1)
plt.yticks(np.arange(-10000, 0, 1000))
plt.yticks([-10000,-9000, -8000, -7000, -6000, -5000, -4000, -3000, -2000, -1000, 0], ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0'])
#plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
#plt.gca().ticklabel_format(useMathText=True)
plt.title(r'$ \times 10^{3}$', loc='left', fontsize = 30)
plt.show()

#calculate metrices
#DDPG-SCA
gradient_ddpg_sca_2000 = (Ave_return_history_ddpg_SCA_mean[299]-Ave_return_history_ddpg_SCA_mean[95])/204
gradient_ddpg_sca_4000 = (Ave_return_history_ddpg_SCA_mean[95]-Ave_return_history_ddpg_SCA_mean[0])/96
gradient_ddpg_sca_3000_6000 = (Ave_return_history_ddpg_SCA_mean[149]-Ave_return_history_ddpg_SCA_mean[20])/129
#0-500 episodes
data_0500_ddpg_sca = Ave_return_history_ddpg_SCA_mean[0:500]
mean_0500_ddpg_sca=np.mean(data_0500_ddpg_sca)
std_data_0500_ddpg_sca=np.std(data_0500_ddpg_sca)
gradient_0500_ddpg_sca = (data_0500_ddpg_sca[499] - data_0500_ddpg_sca[0])/500
max_0500_ddpg_sca=np.max(data_0500_ddpg_sca)
min_0500_ddpg_sca=np.min(data_0500_ddpg_sca)
final_0500_ddpg_sca = data_0500_ddpg_sca[499]

std_500_ddpg_sca=Ave_return_history_ddpg_SCA_std[0:500]
std_500_mean_ddpg_sca = np.mean(std_500_ddpg_sca)
std_500_var_ddpg_sca = np.std(std_500_ddpg_sca)
std_500_grad_ddpg_sca = (std_500_ddpg_sca[499] - std_500_ddpg_sca[0])/500


#2500-3000 episodes
data_500_ddpg_sca = Ave_return_history_ddpg_SCA_mean[-500:]
mean_500_ddpg_sca=np.mean(data_500_ddpg_sca)
std_data_500_ddpg_sca=np.std(data_500_ddpg_sca)
gradient_500_ddpg_sca=(data_500_ddpg_sca[499]-data_500_ddpg_sca[0])/500
max_500_ddpg_sca=np.max(data_500_ddpg_sca)
min_500_ddpg_sca=np.min(data_500_ddpg_sca)
final_500_ddpg_sca = data_500_ddpg_sca[499]

std_last_500_ddpg_sca=Ave_return_history_ddpg_SCA_std[-500:]
std_last_500_mean_ddpg_sca = np.mean(std_last_500_ddpg_sca)
std_last_500_var_ddpg_sca = np.std(std_last_500_ddpg_sca)
std_last_500_grad_ddpg_sca = (std_last_500_ddpg_sca[499] - std_last_500_ddpg_sca[0])/500


#DDPG-SDA
gradient_ddpg_sda_2000 = (Ave_return_history_ddpg_SDA_mean[1331]-Ave_return_history_ddpg_SDA_mean[112])/1219
gradient_ddpg_sda_4000 = (Ave_return_history_ddpg_SDA_mean[112]-Ave_return_history_ddpg_SDA_mean[0])/113
gradient_ddpg_sda_3000_6000 = (Ave_return_history_ddpg_SCA_mean[302]-Ave_return_history_ddpg_SCA_mean[38])/264
#0-500 episodes
data_0500_ddpg_sda = Ave_return_history_ddpg_SDA_mean[0:500]
mean_0500_ddpg_sda=np.mean(data_0500_ddpg_sda)
std_data_0500_ddpg_sda=np.std(data_0500_ddpg_sda)
gradient_0500_ddpg_sda = (data_0500_ddpg_sda[499] - data_0500_ddpg_sda[0])/500
max_0500_ddpg_sda=np.max(data_0500_ddpg_sda)
min_0500_ddpg_sda=np.min(data_0500_ddpg_sda)
final_0500_ddpg_sda = data_0500_ddpg_sda[499]

std_500_ddpg_sda=Ave_return_history_ddpg_SDA_std[0:500]
std_500_mean_ddpg_sda = np.mean(std_500_ddpg_sda)
std_500_var_ddpg_sda = np.std(std_500_ddpg_sda)
std_500_grad_ddpg_sda = (std_500_ddpg_sda[499] - std_500_ddpg_sda[0])/500


#2500-3000 episodes
data_500_ddpg_sda = Ave_return_history_ddpg_SDA_mean[-500:]
mean_500_ddpg_sda=np.mean(data_500_ddpg_sda)
std_data_500_ddpg_sda=np.std(data_500_ddpg_sda)
gradient_500_ddpg_sda=(data_500_ddpg_sda[499]-data_500_ddpg_sda[0])/500
max_500_ddpg_sda=np.max(data_500_ddpg_sda)
min_500_ddpg_sda=np.min(data_500_ddpg_sda)
final_500_ddpg_sda = data_500_ddpg_sda[499]

std_last_500_ddpg_sda=Ave_return_history_ddpg_SDA_std[-500:]
std_last_500_mean_ddpg_sda = np.mean(std_last_500_ddpg_sda)
std_last_500_var_ddpg_sda = np.std(std_last_500_ddpg_sda)
std_last_500_grad_ddpg_sda = (std_last_500_ddpg_sda[499] - std_last_500_ddpg_sda[0])/500

#DDPG
gradient_ddpg_2000 = (Ave_return_history_ddpg_mean[642]-Ave_return_history_ddpg_mean[190])/452
gradient_ddpg_4000 = (Ave_return_history_ddpg_mean[190]-Ave_return_history_ddpg_mean[0])/191
gradient_ddpg_3000_6000 = (Ave_return_history_ddpg_mean[311]-Ave_return_history_ddpg_mean[86])/225
#0-500 episodes
data_0500_ddpg = Ave_return_history_ddpg_mean[0:500]
mean_0500_ddpg=np.mean(data_0500_ddpg)
std_data_0500_ddpg=np.std(data_0500_ddpg)
gradient_0500_ddpg = (data_0500_ddpg[499] - data_0500_ddpg[0])/500
max_0500_ddpg=np.max(data_0500_ddpg)
min_0500_ddpg=np.min(data_0500_ddpg)
final_0500_ddpg = data_0500_ddpg[499]

std_500_ddpg=Ave_return_history_ddpg_std[0:500]
std_500_mean_ddpg = np.mean(std_500_ddpg)
std_500_var_ddpg = np.std(std_500_ddpg)
std_500_grad_ddpg = (std_500_ddpg[499] - std_500_ddpg[0])/500


#2500--3000 episodes
data_500_ddpg = Ave_return_history_ddpg_mean[-500:]
mean_500_ddpg=np.mean(data_500_ddpg)
std_data_500_ddpg=np.std(data_500_ddpg)
gradient_500_ddpg=(data_500_ddpg[499]-data_500_ddpg[0])/500
max_500_ddpg=np.max(data_500_ddpg)
min_500_ddpg=np.min(data_500_ddpg)
final_500_ddpg = data_500_ddpg[499]

std_last_500_ddpg=Ave_return_history_ddpg_std[-500:]
std_last_500_mean_ddpg = np.mean(std_last_500_ddpg)
std_last_500_var_ddpg = np.std(std_last_500_ddpg)
std_last_500_grad_ddpg = (std_last_500_ddpg[499] - std_last_500_ddpg[0])/500




#calculate average return
#DDPG-SCA
Ave_return_history_symddpg_tanhrelu=[]
Max_return_history_symddpg_tanhrelu=[]
Min_return_history_symddpg_tanhrelu=[]
Std_return_history_symddpg_tanhrelu=[]

for i in range(1,3001):
    rollout = Ave_return_history_ddpg_SCA_mean[:i]

    return_average_symddpg_tanhrelu = np.mean(rollout[-100:])
    return_std_symddpg_tanhrelu = np.std(rollout[-100:])
    return_std_max_symddpg_tanhrelu = return_average_symddpg_tanhrelu + return_std_symddpg_tanhrelu
    return_std_min_symddpg_tanhrelu = return_average_symddpg_tanhrelu - return_std_symddpg_tanhrelu

    Ave_return_history_symddpg_tanhrelu.append(return_average_symddpg_tanhrelu)
    Max_return_history_symddpg_tanhrelu.append(return_std_max_symddpg_tanhrelu)
    Min_return_history_symddpg_tanhrelu.append(return_std_min_symddpg_tanhrelu)
    Std_return_history_symddpg_tanhrelu.append(return_std_symddpg_tanhrelu)


gradient_symddpg_tanhrelu_1000 = (Ave_return_history_symddpg_tanhrelu[106]-Ave_return_history_symddpg_tanhrelu[20])/87
data_500_symddpg_tanhrelu=Ave_return_history_symddpg_tanhrelu[-500:]
mean_symddpg_tanhrelu_500 = np.mean(data_500_symddpg_tanhrelu)
std_symddpg_tanhrelu_500 = np.std(data_500_symddpg_tanhrelu)
gradient_symmddpg_tanhrelu_500 = (data_500_symddpg_tanhrelu[499] - data_500_symddpg_tanhrelu[0])/500
max_500_symddpg_tanhrelu=np.max(data_500_symddpg_tanhrelu)
min_500_symddpg_tanhrelu=np.min(data_500_symddpg_tanhrelu)
final_500_symddpg_tanhrelu = data_500_symddpg_tanhrelu[499]

std_500_symddpg_tanhrelu=Std_return_history_symddpg_tanhrelu[-500:]
std_500_mean_symddpg_tanhrelu = np.mean(std_500_symddpg_tanhrelu)
std_500_std_symddpg_tanhrelu = np.std(std_500_symddpg_tanhrelu)
std_500_gradient_symddpg_tanhrelu = (std_500_symddpg_tanhrelu[499] - std_500_symddpg_tanhrelu[0])/500


#symddpg_relurelu
Ave_return_history_symddpg_relurelu=[]
Max_return_history_symddpg_relurelu=[]
Min_return_history_symddpg_relurelu=[]
Std_return_history_symddpg_relurelu=[]

for i in range(1,3001):
    rollout = score_history_training_symddpg_relurelu[:i]

    return_average_symddpg_relurelu = np.mean(rollout[-100:])
    return_std_symddpg_relurelu = np.std(rollout[-100:])
    return_std_max_symddpg_relurelu = return_average_symddpg_relurelu + return_std_symddpg_relurelu
    return_std_min_symddpg_relurelu = return_average_symddpg_relurelu - return_std_symddpg_relurelu

    Ave_return_history_symddpg_relurelu.append(return_average_symddpg_relurelu)
    Max_return_history_symddpg_relurelu.append(return_std_max_symddpg_relurelu)
    Min_return_history_symddpg_relurelu.append(return_std_min_symddpg_relurelu)
    Std_return_history_symddpg_relurelu.append(return_std_symddpg_relurelu)


gradient_symddpg_relurelu_2000 = (Ave_return_history_symddpg_relurelu[64]-Ave_return_history_symddpg_relurelu[0])/65
gradient_symddpg_relurelu_1000 = (Ave_return_history_symddpg_relurelu[286]-Ave_return_history_symddpg_relurelu[64])/204
data_500_symddpg_relurelu=Ave_return_history_symddpg_relurelu[-500:]
mean_symddpg_relurelu_500 = np.mean(data_500_symddpg_relurelu)
std_symddpg_relurelu_500 = np.std(data_500_symddpg_relurelu)
gradient_symmddpg_relurelu_500 = (data_500_symddpg_relurelu[499] - data_500_symddpg_relurelu[0])/500
max_500_symddpg_relurelu=np.max(data_500_symddpg_relurelu)
min_500_symddpg_relurelu=np.min(data_500_symddpg_relurelu)
final_500_symddpg_relurelu = data_500_symddpg_relurelu[499]

std_500_symddpg_relurelu=Std_return_history_symddpg_relurelu[-500:]
std_500_mean_symddpg_relurelu = np.mean(std_500_symddpg_relurelu)
std_500_std_symddpg_relurelu = np.std(std_500_symddpg_relurelu)
std_500_gradient_symddpg_relurelu = (std_500_symddpg_relurelu[499] - std_500_symddpg_relurelu[0])/500

#symddpg_tanhtanh
Ave_return_history_symddpg_tanhtanh=[]
Max_return_history_symddpg_tanhtanh=[]
Min_return_history_symddpg_tanhtanh=[]
Std_return_history_symddpg_tanhtanh=[]

for i in range(1,3001):
    rollout = score_history_training_symddpg_tanhtanh[:i]

    return_average_symddpg_tanhtanh = np.mean(rollout[-100:])
    return_std_symddpg_tanhtanh = np.std(rollout[-100:])
    return_std_max_symddpg_tanhtanh = return_average_symddpg_tanhtanh + return_std_symddpg_tanhtanh
    return_std_min_symddpg_tanhtanh = return_average_symddpg_tanhtanh - return_std_symddpg_tanhtanh

    Ave_return_history_symddpg_tanhtanh.append(return_average_symddpg_tanhtanh)
    Max_return_history_symddpg_tanhtanh.append(return_std_max_symddpg_tanhtanh)
    Min_return_history_symddpg_tanhtanh.append(return_std_min_symddpg_tanhtanh)
    Std_return_history_symddpg_tanhtanh.append(return_std_symddpg_tanhtanh)


gradient_symddpg_tanhtanh_2000 = (Ave_return_history_symddpg_tanhtanh[109]-Ave_return_history_symddpg_tanhtanh[0])/65
gradient_symddpg_tanhtanh_1000 = (Ave_return_history_symddpg_tanhtanh[463]-Ave_return_history_symddpg_tanhtanh[109])/355
data_500_symddpg_tanhtanh=Ave_return_history_symddpg_tanhtanh[-500:]
mean_symddpg_tanhtanh_500 = np.mean(data_500_symddpg_tanhtanh)
std_symddpg_tanhtanh_500 = np.std(data_500_symddpg_tanhtanh)
gradient_symmddpg_tanhtanh_500 = (data_500_symddpg_tanhtanh[499] - data_500_symddpg_tanhtanh[0])/500
max_500_symddpg_tanhtanh=np.max(data_500_symddpg_tanhtanh)
min_500_symddpg_tanhtanh=np.min(data_500_symddpg_tanhtanh)
final_500_symddpg_tanhtanh = data_500_symddpg_tanhtanh[499]

std_500_symddpg_tanhtanh=Std_return_history_symddpg_tanhtanh[-500:]
std_500_mean_symddpg_tanhtanh = np.mean(std_500_symddpg_tanhtanh)
std_500_std_symddpg_tanhtanh = np.std(std_500_symddpg_tanhtanh)
std_500_gradient_symddpg_tanhtanh = (std_500_symddpg_tanhtanh[499] - std_500_symddpg_tanhtanh[0])/500

#ddpg
Ave_return_history_ddpg=[]
Max_return_history_ddpg=[]
Min_return_history_ddpg=[]
Std_return_history_ddpg=[]

for k in range(1,3001):
    rollout = score_history_training_ddpg[:k]

    return_average_ddpg = np.mean(rollout[-100:])
    return_std_ddpg = np.std(rollout[-100:])
    return_std_max_ddpg = return_average_ddpg + return_std_ddpg
    return_std_min_ddpg = return_average_ddpg - return_std_ddpg

    Ave_return_history_ddpg.append(return_average_ddpg)
    Max_return_history_ddpg.append(return_std_max_ddpg)
    Min_return_history_ddpg.append(return_std_min_ddpg)
    Std_return_history_ddpg.append(return_std_ddpg)


gradient_ddpg_2000 = (Ave_return_history_ddpg[366]-Ave_return_history_ddpg[0])/367
gradient_ddpg_1000 = (Ave_return_history_ddpg[2999]-Ave_return_history_ddpg[366])/2634
data_500_ddpg=Ave_return_history_ddpg[-500:]
mean_ddpg_500 = np.mean(data_500_ddpg)
std_ddpg_500 = np.std(data_500_ddpg)
gradient_500_ddpg = (data_500_ddpg[499] - data_500_ddpg[0])/500
max_500_ddpg=np.max(data_500_ddpg)
min_500_ddpg=np.min(data_500_ddpg)
final_500_ddpg = data_500_ddpg[499]

std_500_ddpg=Std_return_history_ddpg[-500:]
std_500_mean_ddpg = np.mean(std_500_ddpg)
std_500_std_ddpg = np.std(std_500_ddpg)
std_500_gradient_ddpg = (std_500_ddpg[499] - std_500_ddpg[0])/500
#Ave_return_history=Ave_return_history[1:3001]
#Max_return_history=Max_return_history[1:3001]
#Min_return_history=Min_return_history[1:3001]

#symddpg_mixdata
Ave_return_history_symddpg_mixdata=[]
Max_return_history_symddpg_mixdata=[]
Min_return_history_symddpg_mixdata=[]
Std_return_history_symddpg_mixdata=[]

for i in range(1,3001):
    rollout = score_history_training_symddpg_mixdata[:i]

    return_average_symddpg_mixdata = np.mean(rollout[-100:])
    return_std_symddpg_mixdata = np.std(rollout[-100:])
    return_std_max_symddpg_mixdata = return_average_symddpg_mixdata + return_std_symddpg_mixdata
    return_std_min_symddpg_mixdata = return_average_symddpg_mixdata - return_std_symddpg_mixdata

    Ave_return_history_symddpg_mixdata.append(return_average_symddpg_mixdata)
    Max_return_history_symddpg_mixdata.append(return_std_max_symddpg_mixdata)
    Min_return_history_symddpg_mixdata.append(return_std_min_symddpg_mixdata)
    Std_return_history_symddpg_mixdata.append(return_std_symddpg_mixdata)


gradient_symddpg_mixdata_2000 = (Ave_return_history_symddpg_mixdata[154]-Ave_return_history_symddpg_mixdata[0])/154
gradient_symddpg_mixdata_1000 = (Ave_return_history_symddpg_mixdata[1340]-Ave_return_history_symddpg_mixdata[154])/1187
data_500_symddpg_mixdata=Ave_return_history_symddpg_mixdata[-500:]
mean_symddpg_mixdata_500 = np.mean(data_500_symddpg_mixdata)
std_symddpg_mixdata_500 = np.std(data_500_symddpg_mixdata)
gradient_symmddpg_mixdata_500 = (data_500_symddpg_mixdata[499] - data_500_symddpg_mixdata[0])/500
max_500_symddpg_mixdata=np.max(data_500_symddpg_mixdata)
min_500_symddpg_mixdata=np.min(data_500_symddpg_mixdata)
final_500_symddpg_mixdata = data_500_symddpg_mixdata[499]

std_500_symddpg_mixdata=Std_return_history_symddpg_mixdata[-500:]
std_500_mean_symddpg_mixdata = np.mean(std_500_symddpg_mixdata)
std_500_std_symddpg_mixdata = np.std(std_500_symddpg_mixdata)
std_500_gradient_symddpg_mixdata = (std_500_symddpg_mixdata[499] - std_500_symddpg_mixdata[0])/500


Episode = np.array(range(0,3000,1))


#plot_training_baseline
fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Ave_return_history_symddpg_tanhrelu,linewidth=4.0,label='DDPG-SCA',color='C3')
plt.fill_between(Episode,Max_return_history_symddpg_tanhrelu,Min_return_history_symddpg_tanhrelu,color='C3',alpha=0.2)
plt.plot(Ave_return_history_symddpg_mixdata,linewidth=4.0,label='DDPG-SDA',color='C1')
plt.fill_between(Episode,Max_return_history_symddpg_mixdata,Min_return_history_symddpg_mixdata,color='C1',alpha=0.2)
plt.plot(Ave_return_history_ddpg,linewidth=4.0,label='DDPG',color='C0')
plt.fill_between(Episode,Max_return_history_ddpg,Min_return_history_ddpg,alpha=0.2,color='C0')
plt.xlabel('Episode [-]',fontdict={'size':35})
plt.ylabel('Average return [-]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True)
plt.legend(loc='lower right',fontsize=35)
plt.ylim(-7000, 0)



#plot_training_5algorithms
#fig3 = plt.figure(figsize=(18.0,9.0))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(Ave_return_history_symddpg_tanhrelu,linewidth=4.0,label='DDPG-SCA (tanh-ReLU)')
#plt.fill_between(Episode,Max_return_history_symddpg_tanhrelu,Min_return_history_symddpg_tanhrelu,alpha=0.2)
#plt.plot(Ave_return_history_symddpg_relurelu,linewidth=4.0,label='DDPG-SCA (ReLU-ReLU)')
#plt.fill_between(Episode,Max_return_history_symddpg_relurelu,Min_return_history_symddpg_relurelu,alpha=0.2)
#plt.plot(Ave_return_history_symddpg_tanhtanh,linewidth=4.0,label='DDPG-SCA (tanh-tanh)')
#plt.fill_between(Episode,Max_return_history_symddpg_tanhtanh,Min_return_history_symddpg_tanhtanh,alpha=0.2)
#plt.plot(Ave_return_history_symddpg_mixdata,linewidth=4.0,label='DDPG-SDA (tanh-ReLU)')
#plt.fill_between(Episode,Max_return_history_symddpg_mixdata,Min_return_history_symddpg_mixdata,alpha=0.2)
#plt.plot(Ave_return_history_ddpg,linewidth=4.0,label='DDPG')
#plt.fill_between(Episode,Max_return_history_ddpg,Min_return_history_ddpg,alpha=0.2)
#plt.xlabel('Episode [-]',fontdict={'size':35})
#plt.ylabel('Average return [-]',fontdict={'size':35})
#plt.xticks(fontsize=35)
#plt.yticks(fontsize=35)
#plt.grid(True)
#plt.legend(loc='lower right',fontsize=35)
#plt.ylim(-7000, 0)
#plt.show()
