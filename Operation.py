import gym
from ddpg_torch import Agent
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from RK import RK4
from Ref import Ref_Sin



# 1_Training Phase
env_aircraft = RK4(step=0.1) #environment

ref_sin_operation = Ref_Sin(T=10)
agent_operation = Agent(alpha=0.001, beta=0.01, input_dims=8, lay1_dims=64, lay2_dims=64, n_actions=2,
              gamma=0.99, tau=0.01, env=env_aircraft, batch_size=256, max_size=9000000) #Agent

#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/Actor_ddpg'))
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/Actor_ddpg_tanhReLU'))
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_relurelu/4_training/Actor_ddpg'))
agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/1_training/Actor_ddpg'))

#DDPG-SCA-Actor
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/Actor_ddpg_tanhtanh'))
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/2_training/Actor_ddpg'))


#DDPG-SCA-Actor
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_betainreward/2_training/Actor_ddpg'))

#DDPG-SCA-Actor
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_sca_tanhtanh/4_training/Actor_ddpg'))

#DDPG-Actor
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/Beta_control/data_ddpg_betainreward/1_training/Actor_ddpg_betainreward'))


#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/code1_symDDPG/tmp/ddpg/data_symddpg_mixdata/training/Actor_ddpg'))
#agent_operation.actor.load_state_dict(T.load('/home/yifei/PycharmProjects/DDPG_lateralflightcontrol_training_original/code/code1_symDDPG/tmp/ddpg/data_symddpg_relurelu2/training/Actor_ddpg'))

np.random.seed(0)


t0 = np.array([0])


#obs = np.array([0.1,0.1,0.1,0.1])

Index0 = 0
score0 = -1000
score_history=[]
average_score_history=[]
Phi_ref = []

amplitude = 20*np.pi/180

for i in range(3000): #each episode means each round of game.
    done = False
    score = 0
    step  = 0.1
    phi_ref = np.zeros(300)

    #x0 = np.array([np.random.normal(30*np.pi/180,10*np.pi/180),np.random.normal(10*np.pi/180,5*np.pi/180),np.random.normal(30*np.pi/180,10*np.pi/180),np.random.normal(10*np.pi/180,5*np.pi/180)]) # Sym-DDPG actor #[roll rate, roll angle, side rate, side angle]obs is the initial value of state variable
    x0 = np.array([np.random.uniform(-30*np.pi/180,30*np.pi/180),np.random.uniform(-10*np.pi/180,10*np.pi/180),np.random.uniform(-30*np.pi/180,30*np.pi/180), np.random.uniform(-10*np.pi/180,10*np.pi/180)]) #DDPG_2buffers actor

    #obs = obs[0]
    k=0
    while k < 300:

        phi0_ref = ref_sin_operation.fun(k, amplitude)
        x0_ref = np.array([phi0_ref, 0, 0, 0])
        e0 = np.subtract(x0, x0_ref)

        u0 = agent_operation.choose_action(np.concatenate((x0, e0))) #计算actor输出

        if u0[0] > 60*np.pi/180:
            u0[0] = 60 * np.pi / 180
        if u0[0] < - 60*np.pi/180:
            u0[0] = -60 * np.pi / 180
        if u0[1] > 30 * np.pi / 180:
            u0[1] = 30 * np.pi / 180
        if u0[1] < - 30 * np.pi / 180:
            u0[1] = -30 * np.pi / 180

        new_state, done, info, empty = env_aircraft.solver(t0, x0, u0) #计算当前一步状态

        phi1_ref = ref_sin_operation.fun(k+1, amplitude)
        x1_ref = np.array([phi1_ref, 0, 0, 0])
        e1 = np.subtract(new_state, x1_ref)

        t1 = t0 + 10 * step #时间更新

        # 计算奖励
        reward = -10 * abs(np.clip(5 * e1[0], -1, 1)) -10 * abs(np.clip(5 * e1[2], -1, 1)) - 1 * abs(new_state[1]) - 1 * abs(new_state[3]) - 0.01 * abs(u0[0]) - 0.01 * abs(u0[1])  # score=-44


        #DDPG算法
        #agent.learn() #训练一次
        score = score + reward#记录分数

        # 数据记录
        agent_operation.remember(np.concatenate((x0, e0)), np.concatenate((new_state, e1)), u0, reward, int(done))  # 存储一步状态转移数据
        phi_ref[k] = phi0_ref

        #状态更新
        k = k + 1
        t0 = t1
        x0 = new_state

    Phi_ref.append(phi_ref)
    score_history.append(score)
    average_score = np.mean(score_history[-100:])
    average_score_history.append(average_score)
    # agent.memory.index = agent.memory.index + 100
    if score > score0:
        #agent_operation.save_models()
        score0 = score
        Index0 = i + 1
        print('save index', i + 1)
    print('episode', i + 1, 'score % .2f' % score,
          '100 game average %.2f' % np.mean(score_history[-100:]), 'maxscore', score0, 'no.', Index0)




np.savetxt('memory1_Action_operation',agent_operation.memory1.Action,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory1_NewState_operation',agent_operation.memory1.Newstate,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory1_Reward_operation',agent_operation.memory1.Reward,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory1_State_operation',agent_operation.memory1.State,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('scorehistory_operation',score_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('Phi_ref',Phi_ref,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)









