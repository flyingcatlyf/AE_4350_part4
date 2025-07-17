import numpy as np
import matplotlib.pyplot as plt
import matplotlib



#operation_data_DDPG-SCA
score_history_operation_ddpg_SCA_1 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/1_operation/memory1_State_operation')
#e_phi_ddpg_sca_1  = score_history_operation_ddpg_SCA_1[:,4]
#e_phi_ddpg_sca_1_arrary = e_phi_ddpg_sca_1[0: 900000]
#e_phi_ddpg_sca_1_reshape = e_phi_ddpg_sca_1_arrary.reshape(3000,300)
#e_phi_ddpg_sca_1_reshape_abs = np.abs(e_phi_ddpg_sca_1_reshape)
#e_phi_ddpg_sca_1_reshape_abs_mean = np.mean(e_phi_ddpg_sca_1_reshape_abs,axis=0)
#e_phi_ddpg_sca_1_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sca_1_reshape_abs_mean)
#Index = np.array(range(0,300,1))
#ITAE = np.multiply(e_phi_ddpg_sca_1_reshape_abs_mean,Index)

score_history_operation_ddpg_SCA_2 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/2_operation/memory1_State_operation')
score_history_operation_ddpg_SCA_3 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/3_operation/memory1_State_operation')
score_history_operation_ddpg_SCA_4 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/4_operation/memory1_State_operation')
score_history_operation_ddpg_SCA_5 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/5_operation/memory1_State_operation')

#operation_data_DDPG-SDA
score_history_operation_ddpg_SDA_1 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sda_betainreward/1_operation/memory1_State_operation')
score_history_operation_ddpg_SDA_2 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sda_betainreward/6_operation/memory1_State_operation')
score_history_operation_ddpg_SDA_3 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sda_betainreward/3_operation/memory1_State_operation')
score_history_operation_ddpg_SDA_4 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sda_betainreward/4_operation/memory1_State_operation')
score_history_operation_ddpg_SDA_5 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sda_betainreward/2_operation/memory1_State_operation')

#operation_data_DDPG
score_history_operation_ddpg_1 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/1_operation/memory1_State_operation')
score_history_operation_ddpg_2 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/2_operation/memory1_State_operation')
score_history_operation_ddpg_3 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/3_operation/memory1_State_operation')
score_history_operation_ddpg_4 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/4_operation/memory1_State_operation')
score_history_operation_ddpg_5 = np.loadtxt('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/5_operation/memory1_State_operation')


#error_ddpg_sca
e_phi_ddpg_sca_1  = score_history_operation_ddpg_SCA_1[:,4]
e_phi_ddpg_sca_2  = score_history_operation_ddpg_SCA_2[:,4]
e_phi_ddpg_sca_3  = score_history_operation_ddpg_SCA_3[:,4]
e_phi_ddpg_sca_4  = score_history_operation_ddpg_SCA_4[:,4]
e_phi_ddpg_sca_5  = score_history_operation_ddpg_SCA_5[:,4]

e_beta_ddpg_sca_1 = score_history_operation_ddpg_SCA_1[:,6]
e_beta_ddpg_sca_2 = score_history_operation_ddpg_SCA_2[:,6]
e_beta_ddpg_sca_3 = score_history_operation_ddpg_SCA_3[:,6]
e_beta_ddpg_sca_4 = score_history_operation_ddpg_SCA_4[:,6]
e_beta_ddpg_sca_5 = score_history_operation_ddpg_SCA_5[:,6]

#error_ddpg_sda
e_phi_ddpg_sda_1  = score_history_operation_ddpg_SDA_1[:,4]
e_phi_ddpg_sda_2  = score_history_operation_ddpg_SDA_2[:,4]
e_phi_ddpg_sda_3  = score_history_operation_ddpg_SDA_3[:,4]
e_phi_ddpg_sda_4  = score_history_operation_ddpg_SDA_4[:,4]
e_phi_ddpg_sda_5  = score_history_operation_ddpg_SDA_5[:,4]

e_beta_ddpg_sda_1 = score_history_operation_ddpg_SDA_1[:,6]
e_beta_ddpg_sda_2 = score_history_operation_ddpg_SDA_2[:,6]
e_beta_ddpg_sda_3 = score_history_operation_ddpg_SDA_3[:,6]
e_beta_ddpg_sda_4 = score_history_operation_ddpg_SDA_4[:,6]
e_beta_ddpg_sda_5 = score_history_operation_ddpg_SDA_5[:,6]

#error_ddpg
e_phi_ddpg_1  = score_history_operation_ddpg_SDA_1[:,4]
e_phi_ddpg_2  = score_history_operation_ddpg_SDA_2[:,4]
e_phi_ddpg_3  = score_history_operation_ddpg_SDA_3[:,4]
e_phi_ddpg_4  = score_history_operation_ddpg_SDA_4[:,4]
e_phi_ddpg_5  = score_history_operation_ddpg_SDA_5[:,4]

e_beta_ddpg_1 = score_history_operation_ddpg_1[:,6]
e_beta_ddpg_2 = score_history_operation_ddpg_2[:,6]
e_beta_ddpg_3 = score_history_operation_ddpg_3[:,6]
e_beta_ddpg_4 = score_history_operation_ddpg_4[:,6]
e_beta_ddpg_5 = score_history_operation_ddpg_5[:,6]

#error_matrix
#ddpg-sca
e_phi_ddpg_sca_1_arrary = e_phi_ddpg_sca_1[0: 900000]
e_phi_ddpg_sca_1_reshape = e_phi_ddpg_sca_1_arrary.reshape(3000,300)

e_phi_ddpg_sca_2_arrary = e_phi_ddpg_sca_2[0: 900000]
e_phi_ddpg_sca_2_reshape = e_phi_ddpg_sca_2_arrary.reshape(3000,300)

e_phi_ddpg_sca_3_arrary = e_phi_ddpg_sca_3[0: 900000]
e_phi_ddpg_sca_3_reshape = e_phi_ddpg_sca_3_arrary.reshape(3000,300)

e_phi_ddpg_sca_4_arrary = e_phi_ddpg_sca_4[0: 900000]
e_phi_ddpg_sca_4_reshape = e_phi_ddpg_sca_4_arrary.reshape(3000,300)

e_phi_ddpg_sca_5_arrary = e_phi_ddpg_sca_5[0: 900000]
e_phi_ddpg_sca_5_reshape = e_phi_ddpg_sca_5_arrary.reshape(3000,300)


#ddpg-sda
e_phi_ddpg_sda_1_arrary = e_phi_ddpg_sda_1[0: 900000]
e_phi_ddpg_sda_1_reshape = e_phi_ddpg_sda_1_arrary.reshape(3000,300)

e_phi_ddpg_sda_2_arrary = e_phi_ddpg_sda_2[0: 900000]
e_phi_ddpg_sda_2_reshape = e_phi_ddpg_sda_2_arrary.reshape(3000,300)

e_phi_ddpg_sda_3_arrary = e_phi_ddpg_sda_3[0: 900000]
e_phi_ddpg_sda_3_reshape = e_phi_ddpg_sda_3_arrary.reshape(3000,300)

e_phi_ddpg_sda_4_arrary = e_phi_ddpg_sda_4[0: 900000]
e_phi_ddpg_sda_4_reshape = e_phi_ddpg_sda_4_arrary.reshape(3000,300)

e_phi_ddpg_sda_5_arrary = e_phi_ddpg_sda_5[0: 900000]
e_phi_ddpg_sda_5_reshape = e_phi_ddpg_sda_5_arrary.reshape(3000,300)

#ddpg
e_phi_ddpg_1_arrary = e_phi_ddpg_1[0: 900000]
e_phi_ddpg_1_reshape = e_phi_ddpg_1_arrary.reshape(3000,300)

e_phi_ddpg_2_arrary = e_phi_ddpg_2[0: 900000]
e_phi_ddpg_2_reshape = e_phi_ddpg_2_arrary.reshape(3000,300)

e_phi_ddpg_3_arrary = e_phi_ddpg_3[0: 900000]
e_phi_ddpg_3_reshape = e_phi_ddpg_3_arrary.reshape(3000,300)

e_phi_ddpg_4_arrary = e_phi_ddpg_4[0: 900000]
e_phi_ddpg_4_reshape = e_phi_ddpg_4_arrary.reshape(3000,300)

e_phi_ddpg_5_arrary = e_phi_ddpg_5[0: 900000]
e_phi_ddpg_5_reshape = e_phi_ddpg_5_arrary.reshape(3000,300)

#Matrix
#ddpg-sca
e_phi_ddpg_sca_1_reshape_abs = np.abs(e_phi_ddpg_sca_1_arrary.reshape(3000,300))
e_phi_ddpg_sca_2_reshape_abs = np.abs(e_phi_ddpg_sca_2_arrary.reshape(3000,300))
e_phi_ddpg_sca_3_reshape_abs = np.abs(e_phi_ddpg_sca_3_arrary.reshape(3000,300))
e_phi_ddpg_sca_4_reshape_abs = np.abs(e_phi_ddpg_sca_4_arrary.reshape(3000,300))
e_phi_ddpg_sca_5_reshape_abs = np.abs(e_phi_ddpg_sca_5_arrary.reshape(3000,300))

#ddpg-sda
e_phi_ddpg_sda_1_reshape_abs = np.abs(e_phi_ddpg_sda_1_arrary.reshape(3000,300))
e_phi_ddpg_sda_2_reshape_abs = np.abs(e_phi_ddpg_sda_2_arrary.reshape(3000,300))
e_phi_ddpg_sda_3_reshape_abs = np.abs(e_phi_ddpg_sda_3_arrary.reshape(3000,300))
e_phi_ddpg_sda_4_reshape_abs = np.abs(e_phi_ddpg_sda_4_arrary.reshape(3000,300))
e_phi_ddpg_sda_5_reshape_abs = np.abs(e_phi_ddpg_sda_5_arrary.reshape(3000,300))

#ddpg
e_phi_ddpg_1_reshape_abs = np.abs(e_phi_ddpg_1_arrary.reshape(3000,300))
e_phi_ddpg_2_reshape_abs = np.abs(e_phi_ddpg_2_arrary.reshape(3000,300))
e_phi_ddpg_3_reshape_abs = np.abs(e_phi_ddpg_3_arrary.reshape(3000,300))
e_phi_ddpg_4_reshape_abs = np.abs(e_phi_ddpg_4_arrary.reshape(3000,300))
e_phi_ddpg_5_reshape_abs = np.abs(e_phi_ddpg_5_arrary.reshape(3000,300))

#ddpg-sca
e_phi_ddpg_sca_1_reshape_abs_mean = np.mean(e_phi_ddpg_sca_1_reshape_abs,axis=0)
e_phi_ddpg_sca_2_reshape_abs_mean = np.mean(e_phi_ddpg_sca_2_reshape_abs,axis=0)
e_phi_ddpg_sca_3_reshape_abs_mean = np.mean(e_phi_ddpg_sca_3_reshape_abs,axis=0)
e_phi_ddpg_sca_4_reshape_abs_mean = np.mean(e_phi_ddpg_sca_4_reshape_abs,axis=0)
e_phi_ddpg_sca_5_reshape_abs_mean = np.mean(e_phi_ddpg_sca_5_reshape_abs,axis=0)

#ddpg-sda
e_phi_ddpg_sda_1_reshape_abs_mean = np.mean(e_phi_ddpg_sda_1_reshape_abs,axis=0)
e_phi_ddpg_sda_2_reshape_abs_mean = np.mean(e_phi_ddpg_sda_2_reshape_abs,axis=0)
e_phi_ddpg_sda_3_reshape_abs_mean = np.mean(e_phi_ddpg_sda_3_reshape_abs,axis=0)
e_phi_ddpg_sda_4_reshape_abs_mean = np.mean(e_phi_ddpg_sda_4_reshape_abs,axis=0)
e_phi_ddpg_sda_5_reshape_abs_mean = np.mean(e_phi_ddpg_sda_5_reshape_abs,axis=0)

#ddpg
e_phi_ddpg_1_reshape_abs_mean = np.mean(e_phi_ddpg_1_reshape_abs,axis=0)
e_phi_ddpg_2_reshape_abs_mean = np.mean(e_phi_ddpg_2_reshape_abs,axis=0)
e_phi_ddpg_3_reshape_abs_mean = np.mean(e_phi_ddpg_3_reshape_abs,axis=0)
e_phi_ddpg_4_reshape_abs_mean = np.mean(e_phi_ddpg_4_reshape_abs,axis=0)
e_phi_ddpg_5_reshape_abs_mean = np.mean(e_phi_ddpg_5_reshape_abs,axis=0)

#Calculate_IAE
#ddpg-sca
e_phi_ddpg_sca_1_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sca_1_reshape_abs_mean)
e_phi_ddpg_sca_2_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sca_2_reshape_abs_mean)
e_phi_ddpg_sca_3_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sca_3_reshape_abs_mean)
e_phi_ddpg_sca_4_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sca_4_reshape_abs_mean)
e_phi_ddpg_sca_5_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sca_5_reshape_abs_mean)

#ddpg-sda
e_phi_ddpg_sda_1_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sda_1_reshape_abs_mean)
e_phi_ddpg_sda_2_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sda_2_reshape_abs_mean)
e_phi_ddpg_sda_3_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sda_3_reshape_abs_mean)
e_phi_ddpg_sda_4_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sda_4_reshape_abs_mean)
e_phi_ddpg_sda_5_reshape_abs_mean_sum = np.sum(e_phi_ddpg_sda_5_reshape_abs_mean)

#ddpg
e_phi_ddpg_1_reshape_abs_mean_sum = np.sum(e_phi_ddpg_1_reshape_abs_mean)
e_phi_ddpg_2_reshape_abs_mean_sum = np.sum(e_phi_ddpg_2_reshape_abs_mean)
e_phi_ddpg_3_reshape_abs_mean_sum = np.sum(e_phi_ddpg_3_reshape_abs_mean)
e_phi_ddpg_4_reshape_abs_mean_sum = np.sum(e_phi_ddpg_4_reshape_abs_mean)
e_phi_ddpg_5_reshape_abs_mean_sum = np.sum(e_phi_ddpg_5_reshape_abs_mean)



#Calculate_ITAE
Index = np.array(range(0,30,0.1))
#ddpg-sca
e_phi_ddpg_sca_1_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sca_1_reshape_abs_mean, Index)
e_phi_ddpg_sca_2_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sca_2_reshape_abs_mean, Index)
e_phi_ddpg_sca_3_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sca_3_reshape_abs_mean, Index)
e_phi_ddpg_sca_4_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sca_4_reshape_abs_mean, Index)
e_phi_ddpg_sca_5_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sca_5_reshape_abs_mean, Index)
#ddpg-sda
e_phi_ddpg_sda_1_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sda_1_reshape_abs_mean, Index)
e_phi_ddpg_sda_2_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sda_2_reshape_abs_mean, Index)
e_phi_ddpg_sda_3_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sda_3_reshape_abs_mean, Index)
e_phi_ddpg_sda_4_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sda_4_reshape_abs_mean, Index)
e_phi_ddpg_sda_5_reshape_abs_mean_t = np.multiply(e_phi_ddpg_sda_5_reshape_abs_mean, Index)
#ddpg
e_phi_ddpg_1_reshape_abs_mean_t = np.multiply(e_phi_ddpg_1_reshape_abs_mean, Index)
e_phi_ddpg_2_reshape_abs_mean_t = np.multiply(e_phi_ddpg_2_reshape_abs_mean, Index)
e_phi_ddpg_3_reshape_abs_mean_t = np.multiply(e_phi_ddpg_3_reshape_abs_mean, Index)
e_phi_ddpg_4_reshape_abs_mean_t = np.multiply(e_phi_ddpg_4_reshape_abs_mean, Index)
e_phi_ddpg_5_reshape_abs_mean_t = np.multiply(e_phi_ddpg_5_reshape_abs_mean, Index)



