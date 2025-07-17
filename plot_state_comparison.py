import numpy as np
import matplotlib.pyplot as plt
import matplotlib

e_phi_ddpg_sca_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_sca_reshape_sum_mean')
e_phi_ddpg_sca_reshape_sum_max  = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_sca_reshape_sum_max')
e_phi_ddpg_sca_reshape_sum_min  = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_sca_reshape_sum_min')

e_phi_ddpg_sda_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_sda_reshape_sum_mean')
e_phi_ddpg_sda_reshape_sum_max  = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_sda_reshape_sum_max')
e_phi_ddpg_sda_reshape_sum_min  = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_sda_reshape_sum_min')

e_phi_ddpg_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_reshape_sum_mean')
e_phi_ddpg_reshape_sum_max  = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_reshape_sum_max')
e_phi_ddpg_reshape_sum_min  = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/e_phi_ddpg_reshape_sum_min')

phi_ddpg_sca_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_sca_reshape_sum_mean')
phi_ddpg_sca_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_sca_reshape_sum_max')
phi_ddpg_sca_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_sca_reshape_sum_min')

phi_ddpg_sda_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_sda_reshape_sum_mean')
phi_ddpg_sda_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_sda_reshape_sum_max')
phi_ddpg_sda_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_sda_reshape_sum_min')

phi_ddpg_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_reshape_sum_mean')
phi_ddpg_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_reshape_sum_max')
phi_ddpg_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/phi_ddpg_reshape_sum_min')

p_ddpg_sca_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_sca_reshape_sum_mean')
p_ddpg_sca_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_sca_reshape_sum_max')
p_ddpg_sca_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_sca_reshape_sum_min')

p_ddpg_sda_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_sda_reshape_sum_mean')
p_ddpg_sda_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_sda_reshape_sum_max')
p_ddpg_sda_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_sda_reshape_sum_min')

p_ddpg_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_reshape_sum_mean')
p_ddpg_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_reshape_sum_max')
p_ddpg_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/p_ddpg_reshape_sum_min')

beta_ddpg_sca_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_sca_reshape_sum_mean')
beta_ddpg_sca_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_sca_reshape_sum_max')
beta_ddpg_sca_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_sca_reshape_sum_min')

beta_ddpg_sda_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_sda_reshape_sum_mean')
beta_ddpg_sda_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_sda_reshape_sum_max')
beta_ddpg_sda_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_sda_reshape_sum_min')

beta_ddpg_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_reshape_sum_mean')
beta_ddpg_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_reshape_sum_max')
beta_ddpg_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/beta_ddpg_reshape_sum_min')

r_ddpg_sca_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_sca_reshape_sum_mean')
r_ddpg_sca_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_sca_reshape_sum_max')
r_ddpg_sca_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_sca_reshape_sum_min')

r_ddpg_sda_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_sda_reshape_sum_mean')
r_ddpg_sda_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_sda_reshape_sum_max')
r_ddpg_sda_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_sda_reshape_sum_min')

r_ddpg_reshape_sum_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_reshape_sum_mean')
r_ddpg_reshape_sum_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_reshape_sum_max')
r_ddpg_reshape_sum_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/r_ddpg_reshape_sum_min')



deltaa_ddpg_sca_reshape_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_sca_reshape_sum_mean')
deltaa_ddpg_sca_reshape_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_sca_reshape_sum_max')
deltaa_ddpg_sca_reshape_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_sca_reshape_sum_min')

deltaa_ddpg_sda_reshape_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_sda_reshape_sum_mean')
deltaa_ddpg_sda_reshape_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_sda_reshape_sum_max')
deltaa_ddpg_sda_reshape_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_sda_reshape_sum_min')

deltaa_ddpg_reshape_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_reshape_sum_mean')
deltaa_ddpg_reshape_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_reshape_sum_max')
deltaa_ddpg_reshape_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltaa_ddpg_reshape_sum_min')

deltar_ddpg_sca_reshape_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_sca_reshape_sum_mean')
deltar_ddpg_sca_reshape_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_sca_reshape_sum_max')
deltar_ddpg_sca_reshape_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_sca_reshape_sum_min')

deltar_ddpg_sda_reshape_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_sda_reshape_sum_mean')
deltar_ddpg_sda_reshape_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_sda_reshape_sum_max')
deltar_ddpg_sda_reshape_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_sda_reshape_sum_min')

deltar_ddpg_reshape_mean = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_reshape_sum_mean')
deltar_ddpg_reshape_max = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_reshape_sum_max')
deltar_ddpg_reshape_min = np.loadtxt('/home/yifei/PycharmProjects/Chapter5/Data_Reshape/deltar_ddpg_reshape_sum_min')

#unite change
#DDPG-SCA
e_phi_ddpg_sca_reshape_sum_mean = e_phi_ddpg_sca_reshape_sum_mean * 180/np.pi
e_phi_ddpg_sca_reshape_sum_max = e_phi_ddpg_sca_reshape_sum_max * 180/np.pi
e_phi_ddpg_sca_reshape_sum_min = e_phi_ddpg_sca_reshape_sum_min * 180/np.pi

e_phi_ddpg_sda_reshape_sum_mean = e_phi_ddpg_sda_reshape_sum_mean * 180/np.pi
e_phi_ddpg_sda_reshape_sum_max = e_phi_ddpg_sda_reshape_sum_max * 180/np.pi
e_phi_ddpg_sda_reshape_sum_min = e_phi_ddpg_sda_reshape_sum_min * 180/np.pi

e_phi_ddpg_reshape_sum_mean = e_phi_ddpg_reshape_sum_mean * 180/np.pi
e_phi_ddpg_reshape_sum_max = e_phi_ddpg_reshape_sum_max * 180/np.pi
e_phi_ddpg_reshape_sum_min = e_phi_ddpg_reshape_sum_min * 180/np.pi

phi_ddpg_sca_reshape_sum_mean = phi_ddpg_sca_reshape_sum_mean * 180/np.pi
phi_ddpg_sca_reshape_sum_max = phi_ddpg_sca_reshape_sum_max * 180/np.pi
phi_ddpg_sca_reshape_sum_min = phi_ddpg_sca_reshape_sum_min * 180/np.pi

phi_ddpg_sda_reshape_sum_mean = phi_ddpg_sda_reshape_sum_mean * 180/np.pi
phi_ddpg_sda_reshape_sum_max = phi_ddpg_sda_reshape_sum_max * 180/np.pi
phi_ddpg_sda_reshape_sum_min = phi_ddpg_sda_reshape_sum_min * 180/np.pi

phi_ddpg_reshape_sum_mean = phi_ddpg_reshape_sum_mean * 180/np.pi
phi_ddpg_reshape_sum_max = phi_ddpg_reshape_sum_max * 180/np.pi
phi_ddpg_reshape_sum_min = phi_ddpg_reshape_sum_min * 180/np.pi

p_ddpg_sca_reshape_sum_mean = p_ddpg_sca_reshape_sum_mean * 180/np.pi
p_ddpg_sca_reshape_sum_max = p_ddpg_sca_reshape_sum_max * 180/np.pi
p_ddpg_sca_reshape_sum_min = p_ddpg_sca_reshape_sum_min * 180/np.pi

p_ddpg_sda_reshape_sum_mean = p_ddpg_sda_reshape_sum_mean * 180/np.pi
p_ddpg_sda_reshape_sum_max = p_ddpg_sda_reshape_sum_max * 180/np.pi
p_ddpg_sda_reshape_sum_min = p_ddpg_sda_reshape_sum_min * 180/np.pi

p_ddpg_reshape_sum_mean = p_ddpg_reshape_sum_mean * 180/np.pi
p_ddpg_reshape_sum_max = p_ddpg_reshape_sum_max * 180/np.pi
p_ddpg_reshape_sum_min = p_ddpg_reshape_sum_min * 180/np.pi

beta_ddpg_sca_reshape_sum_mean = beta_ddpg_sca_reshape_sum_mean * 180/np.pi
beta_ddpg_sca_reshape_sum_max = beta_ddpg_sca_reshape_sum_max * 180/np.pi
beta_ddpg_sca_reshape_sum_min = beta_ddpg_sca_reshape_sum_min * 180/np.pi

beta_ddpg_sda_reshape_sum_mean = beta_ddpg_sda_reshape_sum_mean * 180/np.pi
beta_ddpg_sda_reshape_sum_max = beta_ddpg_sda_reshape_sum_max * 180/np.pi
beta_ddpg_sda_reshape_sum_min = beta_ddpg_sda_reshape_sum_min * 180/np.pi

beta_ddpg_reshape_sum_mean = beta_ddpg_reshape_sum_mean * 180/np.pi
beta_ddpg_reshape_sum_max = beta_ddpg_reshape_sum_max * 180/np.pi
beta_ddpg_reshape_sum_min = beta_ddpg_reshape_sum_min * 180/np.pi

r_ddpg_sca_reshape_sum_mean = r_ddpg_sca_reshape_sum_mean * 180/np.pi
r_ddpg_sca_reshape_sum_max = r_ddpg_sca_reshape_sum_max * 180/np.pi
r_ddpg_sca_reshape_sum_min = r_ddpg_sca_reshape_sum_min * 180/np.pi

r_ddpg_sda_reshape_sum_mean = r_ddpg_sda_reshape_sum_mean * 180/np.pi
r_ddpg_sda_reshape_sum_max = r_ddpg_sda_reshape_sum_max * 180/np.pi
r_ddpg_sda_reshape_sum_min = r_ddpg_sda_reshape_sum_min * 180/np.pi

r_ddpg_reshape_sum_mean = r_ddpg_reshape_sum_mean * 180/np.pi
r_ddpg_reshape_sum_max = r_ddpg_reshape_sum_max * 180/np.pi
r_ddpg_reshape_sum_min = r_ddpg_reshape_sum_min * 180/np.pi

deltaa_ddpg_sca_reshape_mean = deltaa_ddpg_sca_reshape_mean * 180/np.pi
deltaa_ddpg_sca_reshape_max = deltaa_ddpg_sca_reshape_max * 180/np.pi
deltaa_ddpg_sca_reshape_min = deltaa_ddpg_sca_reshape_min * 180/np.pi

deltaa_ddpg_sda_reshape_mean = deltaa_ddpg_sda_reshape_mean * 180/np.pi
deltaa_ddpg_sda_reshape_max = deltaa_ddpg_sda_reshape_max * 180/np.pi
deltaa_ddpg_sda_reshape_min = deltaa_ddpg_sda_reshape_min * 180/np.pi

deltaa_ddpg_reshape_mean = deltaa_ddpg_reshape_mean * 180/np.pi
deltaa_ddpg_reshape_max = deltaa_ddpg_reshape_max * 180/np.pi
deltaa_ddpg_reshape_min = deltaa_ddpg_reshape_min * 180/np.pi

deltar_ddpg_sca_reshape_mean = deltar_ddpg_sca_reshape_mean * 180/np.pi
deltar_ddpg_sca_reshape_max = deltar_ddpg_sca_reshape_max * 180/np.pi
deltar_ddpg_sca_reshape_min = deltar_ddpg_sca_reshape_min * 180/np.pi

deltar_ddpg_sda_reshape_mean = deltar_ddpg_sda_reshape_mean * 180/np.pi
deltar_ddpg_sda_reshape_max = deltar_ddpg_sda_reshape_max * 180/np.pi
deltar_ddpg_sda_reshape_min = deltar_ddpg_sda_reshape_min * 180/np.pi

deltar_ddpg_reshape_mean = deltar_ddpg_reshape_mean * 180/np.pi
deltar_ddpg_reshape_max = deltar_ddpg_reshape_max * 180/np.pi
deltar_ddpg_reshape_min = deltar_ddpg_reshape_min * 180/np.pi

time = np.array(range(0,300,1))
Error_ref0 = np.zeros(300)
Phi_ref = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/1_operation/Phi_ref')
phi_ref = Phi_ref[1,:]
phi_ref = phi_ref * 180/np.pi
#Plot_e_Phi
fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(e_phi_ddpg_sca_reshape_sum_mean,linewidth=4.0,label='DDPG-SCA',color = 'C3')
plt.fill_between(time,e_phi_ddpg_sca_reshape_sum_max,e_phi_ddpg_sca_reshape_sum_min,alpha=0.2,color = 'C3')
plt.plot(e_phi_ddpg_sda_reshape_sum_mean,linewidth=4.0,label='DDPG-SDA',color = 'C1')
plt.fill_between(time,e_phi_ddpg_sda_reshape_sum_max,e_phi_ddpg_sda_reshape_sum_min,alpha=0.2,color = 'C1')
plt.plot(e_phi_ddpg_reshape_sum_mean,linewidth=4.0,label='DDPG',color = 'C0')
plt.fill_between(time,e_phi_ddpg_reshape_sum_max,e_phi_ddpg_reshape_sum_min,alpha=0.2,color = 'C0')
plt.plot(Error_ref0,linewidth=2.0,color = 'k',linestyle='--')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$e_{\phi}$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.ylim((-20,20))
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

#9_subplots_for_states
fig2 = plt.figure(figsize=(18.0,9.0))
plt.subplot(6,3,1)
plt.plot(phi_ddpg_sca_reshape_sum_mean ,linewidth=1.0)
plt.fill_between(time,phi_ddpg_sca_reshape_sum_max, phi_ddpg_sca_reshape_sum_min,alpha=0.2)
plt.plot(phi_ref,linewidth=1.0,label='$\phi_{ref}$',linestyle='--',color = 'C3')
plt.ylabel(r'$\phi$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.title('DDPG-SCA')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,2)
plt.plot(phi_ddpg_sda_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,phi_ddpg_sda_reshape_sum_max,phi_ddpg_sda_reshape_sum_min,alpha=0.2)
plt.plot(phi_ref,linewidth=1.0,label='$\phi_{ref}$',linestyle='--',color = 'C3')
plt.ylabel(r'$\phi$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.title('DDPG-SDA')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,3)
plt.plot(phi_ddpg_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,phi_ddpg_reshape_sum_max,phi_ddpg_reshape_sum_min,alpha=0.2)
plt.plot(phi_ref,linewidth=1.0,label='$\phi_{ref}$',linestyle='--',color = 'C3')
plt.ylabel(r'$\phi$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.title('DDPG')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])


plt.subplot(6,3,4)
plt.plot(p_ddpg_sca_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,p_ddpg_sca_reshape_sum_max,p_ddpg_sca_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$p$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,5)
plt.plot(p_ddpg_sda_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,p_ddpg_sda_reshape_sum_max,p_ddpg_sda_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$p$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,6)
plt.plot(p_ddpg_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,p_ddpg_reshape_sum_max,p_ddpg_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$p$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,7)
plt.plot(deltaa_ddpg_sca_reshape_mean,linewidth=1.0)
plt.fill_between(time,deltaa_ddpg_sca_reshape_max,deltaa_ddpg_sca_reshape_min,alpha=0.2)
plt.ylabel(r'$\delta_{a}$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,8)
plt.plot(deltaa_ddpg_sda_reshape_mean,linewidth=1.0)
plt.fill_between(time,deltaa_ddpg_sda_reshape_max,deltaa_ddpg_sda_reshape_min,alpha=0.2)
plt.ylabel(r'$\delta_{a}$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,9)
plt.plot(deltaa_ddpg_reshape_mean,linewidth=1.0)
plt.fill_between(time,deltaa_ddpg_reshape_max,deltaa_ddpg_reshape_min,alpha=0.2)
plt.ylabel(r'$\delta_{a}$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,10)
plt.plot(beta_ddpg_sca_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,beta_ddpg_sca_reshape_sum_max,beta_ddpg_sca_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$\beta$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,11)
plt.plot(beta_ddpg_sda_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,beta_ddpg_sda_reshape_sum_max,beta_ddpg_sda_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$\beta$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,12)
plt.plot(beta_ddpg_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,beta_ddpg_reshape_sum_max,beta_ddpg_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$\beta$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,13)
plt.plot(r_ddpg_sca_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,r_ddpg_sca_reshape_sum_max,r_ddpg_sca_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$r$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,14)
plt.plot(r_ddpg_sda_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,r_ddpg_sda_reshape_sum_max,r_ddpg_sda_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$r$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

plt.subplot(6,3,15)
plt.plot(r_ddpg_reshape_sum_mean,linewidth=1.0)
plt.fill_between(time,r_ddpg_reshape_sum_max,r_ddpg_reshape_sum_min,alpha=0.2)
plt.ylabel(r'$r$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])


plt.subplot(6,3,16)
plt.plot(deltar_ddpg_sca_reshape_mean,linewidth=1.0)
plt.fill_between(time,deltar_ddpg_sca_reshape_max,deltar_ddpg_sca_reshape_min,alpha=0.2)
plt.ylabel(r'$\delta_{r}$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])


plt.subplot(6,3,17)
plt.plot(deltar_ddpg_sda_reshape_mean,linewidth=1.0)
plt.fill_between(time,deltar_ddpg_sda_reshape_max,deltar_ddpg_sda_reshape_min,alpha=0.2)
plt.ylabel(r'$\delta_{r}$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])


plt.subplot(6,3,18)
plt.plot(deltar_ddpg_reshape_mean,linewidth=1.0)
plt.fill_between(time,deltar_ddpg_reshape_max,deltar_ddpg_reshape_min,alpha=0.2)
plt.ylabel(r'$\delta_{r}$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])


plt.show()
