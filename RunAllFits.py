import multiprocessing
import os    
import sys
#Markov chain Monte Carlo (MCMC) Bayesian analysis fits
# Creating the tuple of all the processes
dirc= '/beegfs/desy/user/lalasfar/trilinear4tops'
all_processes = (
    dirc+'/Scripts/FitCqt1.py', dirc+'/Scripts/FitCqt8.py',  
              dirc+'/Scripts/FitCqtqb1.py', dirc+'/Scripts/FitCqtqb8.py', dirc+'/Scripts/FourParamFit.py')                                                                                                                            
#all_processes =(dirc+'/Scripts/FitCqt1-HLLHC.py', dirc+'/Scripts/FitCqt8-HLLHC.py', dirc+'/Scripts/FitCqtqb1-HLLHC.py', dirc+'/Scripts/FitCqtqb8-HLLHC.py', dirc+'/Scripts/FourParamFit.py')                                                                                                                            
def execute(process):                                                             
    os.system(f'python3.9 {process}')                                       
                                                                                
                                                                                
process_pool = multiprocessing.Pool(processes = 5)                                                        
process_pool.map(execute, all_processes)
