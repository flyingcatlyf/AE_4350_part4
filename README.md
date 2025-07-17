# AE_4350_part4
The 4th part of code for course AE-4350: plot files
# content
A) Operation&Plot
    0) Operation.py  -Main file for operating the trained flight control system (run to start the operation process)
    1) RK.py  -iteratively solves lateral dynamical equations over time with Runge Kutta method
    2) Ref.py -Producecs reference signals
    3) Computation_IAE_ITAE  -Computes tracking performance meausres in operation phase 
    4) plot_operation_average_reward.py  -Compares average reward in operation phase
    5) plot_state_comparison.py  -Compares states and actions in operation phase
    6) plot_training.py  -Compares training performance
# how to use the code
A) RUN TRAINING
     0) Install an interpreter environment that includes packages in interpreter_packages.txt
     1) Run Training.py to start training of the flight controller.  
     2) Save data automatically in file 'tmp/ddpg'

B) RUN OPERATION
     0) Install an interpreter environment that includes packages in interpreter_packages.txt
     1) Run Operation.py, to start operation phase. The file 'Actor_ddpg.zip' includes actor weights file and should be uploaded to run operation.
     2) Save data automatically
