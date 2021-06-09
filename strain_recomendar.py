#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:14:40 2021

@author: maryam sabzevari
"""

import sys
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import Normalizer
import random
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from numpy import sqrt, diag, zeros, dot, abs
from random import choice
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import yaml
#############################################################################
import matlab.engine
eng = matlab.engine.start_matlab()
class Recommender:
    def __init__(self,seed1,seed2,product,param_dict):
        # load the parameters and initialization
        self.reas_indx = param_dict[product+"_reas_indx"]
        self.target_rea_indx = param_dict[product+"_target_rea_indx"]
        self.warmup_its = param_dict["warmup_its"]
        self.agent_num = param_dict["agent_num"]
        self.glucose_rea_indx = param_dict["glucose_rea_indx"]
        self.enz_ub = param_dict["enz_ub"]
        self.enz_lb = param_dict["enz_lb"]
        self.state_size = param_dict["metab_size"]
        self.grp_size = param_dict["grp_size"]
        self.step_size = param_dict["step_size"]
        self.total_its = param_dict["total_its"]
        self.agents_list = []
        self.new_st = []
        self.perturb_size = len(self.reas_indx)
        self.prs_hist =  np.zeros((self.agent_num, self.perturb_size, self.warmup_its))
        self.response_hist = np.zeros((self.agent_num, 1, self.warmup_its)) 
        self.obs_hist = np.zeros((self.agent_num, self.state_size, self.warmup_its))
        self.sim = Simulator()
        self.perturb_list = list()
        self.response_list = list()
        self.new_pvals_list = list()
        all_agents = list(range(0, self.agent_num))
        self.lead = all_agents[0::self.grp_size]
        self.fol = all_agents[1::self.grp_size]
        self.seed1 = seed1
        self.seed2 = seed2

    def norm_updates (self,x):
        # limit the enzyme levels modifications using the step_size
        x_norm=[self.step_size if y>self.step_size else -self.step_size if y<-self.step_size else y for y in x]
        return x_norm

    def vals_to_prs (self,val):
        # convert the simulator format enzyme levels to the specified range [-1,0.82]
        print("type(val)",type(val))
        if (any(isinstance(i, list) for i in val)):
            xn = [[(-1+y)/(y+1)  for y in x] for x in val]
        else:
            xn = [(-1+y)/(y+1) for y in val ]
        return xn

    #def valstoprs (self,val):
    #    xn = [[(-1+y)/(y+1)  for y in x] for x in val]
    #    return xn
    
    def prs_to_vals(self, prs):
        # convert back the enzyme level to the simulator format
        print("type(prs)",type(prs))
        if (any(isinstance(i, list) for i in prs)):
           vals = [[-(y+1)/(y-1)  for y in x] for x in prs]
        else:
           vals  = [-(x+1)/(x-1)  for x in prs]
        return vals
    
    #def prs_to_vals_l(self, prs):
    #    xn = [[-(y+1)/(y-1)  for y in x] for x in prs]
    #    return xn    
    
    def chk_valid_bounds(self,prs):
        #check the converted enzyme levels are not violating the bounds
        prs = [[random.uniform(self.enz_ub-0.05, self.enz_ub) if y > self.enz_ub else random.uniform(self.enz_lb, self.enz_lb+0.05) if y < self.enz_lb else y for y in x] for x in prs]
        return prs
 
    def warmup(self):
        # load the LHS data (generated data in  LHS folder) for warm-up
        dim13_LHS = np.load("./warmup_data/dim13_LHS.npy")
        prs_temp = dim13_LHS[int(self.seed1),]
        # loop over warm-up iterations
        for ind in range(self.warmup_its):
            print("warm-up iteration: ",ind)
            print("******")
            for ip in range (self.agent_num):
                print("agent ID",ip)
                print("*****")
                prs_in = prs_temp[ip+(ind*self.agent_num),]
                testVector_values  = self.prs_to_vals(prs_in) 
                #print("enzyme levels within warm-up period",testVector_values)
                self.obs_hist[ip,:,ind], self.response_hist[ip,:,ind], sim_in = self.sim.run(self.reas_indx,list(testVector_values))
                print("enzyme levels within warm-up period",self.vals_to_prs(sim_in))
                print("yield",self.response_hist[ip,:,ind])
                self.prs_hist[ip,:,ind]= self.vals_to_prs(sim_in)
           

    def agents_init(self):
        # agents initialization
        np.random.seed(int(self.seed2))
        random.seed(int(self.seed1))
        init_prs = np.random.uniform(self.enz_lb,self.enz_ub,(10,self.perturb_size) )
        prs = init_prs[0:self.agent_num,:]
        ind_max=np.where(self.response_hist > np.quantile(self.response_hist, 0.8))
        prs[0,]= np.median(self.prs_hist[ind_max[0],:,ind_max[2]],axis=0)
        prs = self.chk_valid_bounds (prs)
        return prs
    
    def exps(self,prs,step):
        # call the simulator and action prediction for the next round
        testVector_values  = self.prs_to_vals(prs) #change the current enzyme levels to the simulator format
        response = np.zeros(self.agent_num)
        out_conc = np.zeros((self.agent_num, self.state_size))
        in_conc = np.zeros((self.agent_num, self.perturb_size))
        # for all agents generate the results, can be parallelized
        for ip in range(self.agent_num):
            out_conc[ip], response[ip],in_conc[ip] = self.sim.run(self.reas_indx,list(testVector_values[ip]))
        prs =   np.asanyarray(self.vals_to_prs(in_conc))
        print("converted enzyme levels_corrected",prs)
        print("yield",response)
        # update lists of original enzyme levels, yields,  
        self.perturb_list.append(in_conc) # update the list of enzyme levels in simulator format
        self.response_list.append(response) # update the list of reponses (yields) 
        self.prs_hist = np.dstack((self.prs_hist,prs)) # update the matrix correpond to transofromed enzyme levels
        self.obs_hist = np.dstack((self.obs_hist,out_conc)) # update the matrix correpond to output concentrations (observations)
        self.response_hist = np.dstack((self.response_hist,response.reshape(-1,1))) # update the vector correpond to response (yield)
        # predict the next action and prs
        new_pvals,prs = self.actor_prediction(prs,step)
        # update the action list
        self.new_pvals_list.append( np.transpose(new_pvals))    
        return prs            
        
    def kernel_bandw_comp (self,mat):
       # RBF kernel bandwidth computation
       b = mat.reshape(mat.shape[0], 1, mat.shape[1])
       st_k=np.sqrt(np.einsum('ijk, ijk->ij', mat-b, mat-b))
       gamma =(1/((np.quantile(st_k, 0.5))+np.std(st_k)))
       return gamma

    def proj(self,v,u):
     # compute the projection for gsBasis function (Gram–Schmidt process)
     u_norm = np.sqrt(sum(u**2))
     # project u on the v    
     proj_val = (np.dot(v, u)/u_norm**2)*u
     return proj_val

    def  gsBasis(self,A):
        # Gram–Schmidt process
        B = np.array(A, dtype=np.float_) 
        # loop over all vectors, starting with zero, label them with i
        for i in range(B.shape[1]) : #i=1 j=0
            # inside that loop, loop over all previous vectors, j, to subtract.
            sub = B[:, i]
            for j in range(i) :
                if (abs(sub - self.proj(B[:, i],B[:, j])<0.000000001)).any():
                    B[:, i] = np.random.uniform(np.min(B[:, i]),np.max(B[:, i]),np.shape(sub) )
                    sub = B[:, i]
                sub = sub - self.proj(B[:, i],B[:, j])
       
            B[:, i]=sub
        return B

    def actor_prediction(self,prs_current,step):
      # predict the next actions for the agents
      yield_hist = np.copy(self.response_hist)
      obs_hist = np.copy(self.obs_hist)
      # initialisation
      train_len = np.shape(self.obs_hist)[2]-1  
      states= np.empty((0,self.state_size))
      actions =np.empty((0,self.perturb_size))
      new_pvals = np.zeros((self.agent_num,self.perturb_size))
      #reward list 
      rew_list = np.diff(yield_hist,axis = 2)
      rew_list[rew_list<0]=0 #negative rewards are substituted with zero
      #action list
      action_list = np.diff(self.prs_hist,axis = 2)
      #Every 5 iterations, discard the current enzyme level correpond to worst agent      
      worst_ag=-1
      best_ag=-1
      if (step>0 and step%5==0):
        #determine the agent with worst recent yields, and substitute it randomly with another agent
        worst_ag=np.argmin(np.median(yield_hist[:,0,-5:],axis=1)) 
        all_ags=self.lead+self.fol
        all_ags.remove(worst_ag)
        best_ag = random.choice( all_ags ) 
        print("worst_ag",worst_ag)
        print("best_ag",best_ag)
        #print("prs_current",prs_current)
        prs_current[worst_ag]=prs_current[best_ag]
      for ip in self.lead:
          #rand_index determines the indexes that will be used for training, at the current version we use all
          rand_index = list(range(0, train_len))
          states =  obs_hist[ip,:,rand_index]
          actions =  action_list[ip,:,rand_index]
          rew_list_vals = np.squeeze(rew_list[ip,:,rand_index]) 
          for io in range(ip+1,ip+self.grp_size):#each leader with its followers will form a kernel of states and actions 
              states =  np.vstack((states, obs_hist[io,:,rand_index]))
              actions =  np.vstack((actions,  action_list[io,:,rand_index] ))
              rew_list_vals = np.append(rew_list_vals, np.squeeze(rew_list[io,:,rand_index]))
 
          states_org=np.copy(states)
          pipe = make_pipeline(RobustScaler(), Normalizer())
          tr_st = pipe.fit(states)  # apply scaling on training data
          states=tr_st.transform(states)            
          # standardize and normalize
          tr_ac = pipe.fit(actions)
          actions=tr_ac.transform(actions)
          gamma_state = self.kernel_bandw_comp (states)
          # compute states kernel      
          KX =  rbf_kernel(states, gamma = gamma_state)
          gamma_action = self.kernel_bandw_comp (actions)
          # compute actions kernel
          KY = rbf_kernel(actions, gamma =gamma_action)
          ##################################################################################################
          LM_obj= LearningModel() # calling MMR
          qs=np.log2(rew_list_vals+1) # margin
          beta_opt = LM_obj.mmr_solver(KX,KY,qs)
          #################################################################################################
          # compute the original actions for each agent
          for ip_ind in range(ip,ip+self.grp_size): #ip_ind=0
            states_test =  np.vstack([states_org,obs_hist[ip_ind,:,train_len]])
            tr_st_test = pipe.fit(states_test)
            states_test =tr_st_test.transform(states_test)
            KX_test =  rbf_kernel(states_test, gamma = gamma_state)
            coef_norm = beta_opt * KX_test[-1,0:-1]
            coef_norm = coef_norm/(np.sum(coef_norm)+0.00001)
            ######################################################################
            #computed actions
            new_pvals[ip_ind] = np.squeeze( np.dot(np.array(actions).T  , np.transpose(coef_norm).reshape(-1,1) ))       
            ######################################################################################################
      final_prs_norm = np.zeros((self.agent_num,self.perturb_size))
      # Gram–Schmidt process for action perturbation
      for ii in self.lead:# ii=0
        prs_trs=np.transpose(new_pvals[ii:ii+self.grp_size,:]) 
        prs_GSch = np.transpose(self.gsBasis(prs_trs))
        ind_prs=0
        for ind in range(ii,ii+self.grp_size):
            prs_GSch[ind_prs] = LA.norm(new_pvals[ind]) * prs_GSch[ind_prs] / LA.norm(prs_GSch[ind_prs])
            ind_prs = ind_prs +1
        final_prs = 0.2*new_pvals[ii:ii+self.grp_size,:] +0.8*prs_GSch
        ######################################################################################################
        # transform to the original length of the actions
        ind_final=0
        for ii_pair in range(ii,ii+2):#ii_pair=2
            final_prs_norm[ii_pair] = self.norm_updates(LA.norm(new_pvals[ii_pair])*final_prs[ind_final] /LA.norm(final_prs[ind_final])) #+ #np.random.uniform(-0.05,0.05,perturb_size)
            ind_final=ind_final+1
      prs_current = prs_current + final_prs_norm
  
      return final_prs_norm, prs_current

    def run(self):
        # call the warm-up and experiment running functions
        self.warmup()
        prs = self.agents_init()
        print("################################")
        print("################################")
        print("initial prs",prs)
        print("################################") 
        print("################################")
        for step in range(self.total_its):
            prs = self.exps(prs,step)
            prs = self.chk_valid_bounds (prs)
            print("step",step)
            print("prs",prs)
            np.save("./str_recom_acet/response_list"+str(self.seed1),self.response_list)
            np.save("./str_recom_acet/perturb_list"+str(self.seed1),self.perturb_list)
            np.save("./str_recom_acet/response_hist"+str(self.seed1),self.response_hist) 
            np.save("./str_recom_acet/prs_hist"+str(self.seed1),self.prs_hist)
            #print bestfound strain and its yield
            #at every step, save all the matrices updates
  
    
class Simulator:
     # call the simulator
     def __init__(self):
         eng.cd(r'./model/')

     def run(self,reas_indx,params):
         params=[random.uniform(0.1, 0.2)  if y<0.1 else y for y in params]  
         fluxes,output_concentrations,Error = eng.Main_Module(matlab.double(reas_indx),matlab.double(params),"aerobic_glucose",3000,nargout=3)
         params_tmp = np.copy(params)
         if Error > 5:
             fluxes,output_concentrations,params_tmp = self.find_valid_nn(reas_indx,params_tmp,Error)
         fluxes=list(np.squeeze(np.asanyarray(fluxes)[:,-1]))
         out_yield = np.abs(fluxes[18]/float(fluxes[24]))*100
         output_concentrations =  list(np.squeeze(np.asanyarray(output_concentrations)[:,-1]))
         return output_concentrations,out_yield,np.squeeze(params_tmp)
  
    
     def find_valid_nn(self,reas_indx,params,Error):
         # if Error>5 find a valid point in the close neighbourhood         
         while Error>5:
             params_tmp = params
             params_tmp = [x+random.uniform(-1, 1) if x > 1 else x+random.uniform(-0.1, 0.1) for x in params_tmp]
             params_tmp = [random.uniform(9.5, 10) if x > 10 else random.uniform(0.1, 0.2) if x<0.1 else x  for x in params_tmp]
             fluxes,output_concentrations,Error = eng.Main_Module(matlab.double(reas_indx),matlab.double(list(params_tmp)),"aerobic_glucose",3000,nargout=3)
         return fluxes,output_concentrations,params_tmp
     
#     def chk_non_zero (in_vals):
#         return [random.uniform(0.1, 0.2)  if y<0.1 else y for y in in_vals]         
class Agent:
     def __init__(self, ID, curr_st):
        self.ID = ID
        self.curr_st = curr_st 
     def print(self):
         print("agent_number:",self.ID)
     def init_agents(self):
        for i in range(self.agent_num):
            agent = Agent(i,random.random())
            self.agents_list.append(agent)
        return self.agents_list
    

class LearningModel:
    # learning model (MMR)
    def __init__(self):
        self.maxiter=50
        self.err_tolerance=0.001
        self.xeps=10**(-4)
        self.normy1=1
        self.normy2=1
        self.normx1=1
        self.normx2=1
        self.C=1
        self.D=0
    def mmr_solver(self,Kx,Ky,qs0):

        dx=diag(Kx)
        dy=diag(Ky)
        dx=dx+(abs(dx)+self.xeps)*(dx<=0)
        dy=dy+(abs(dy)+self.xeps)*(dy<=0)
        dKx=sqrt(dx)
        dKy=sqrt(dy)
    
        dKxy1=dKx**self.normx1*dKy**self.normy1   # norm based scale of the margin
        dKxy2=dKx**self.normx2*dKy**self.normy2   # norm based scale of the loss
    
        dKxy2+=1.0*(dKxy2==0)    # to avoid zero
    
        lB=float(self.D)/(dKxy2)               # scale the ranges
        uB=float(self.C)/(dKxy2)
    
        Bdiff=uB-lB
        z_eps=self.err_tolerance*sqrt(sum(Bdiff*Bdiff))
    
        if qs0 is None:
          qs=-dKxy1
        else:
          qs=-qs0*dKxy1
    
        Kxy=Kx*Ky
        m=Kxy.shape[0]
      ## scaling by diagonal elements  
    #    dKxy=diag(Kx)
    #    dKxy=dKxy+(dKxy==0)
    #    Kxy=Kxy/outer(dKxy,dKxy)
    #    qs=qs/dKxy
    
        for irow in range(m):#irow=0, irepeat=0
          if Kxy[irow,irow]==0:
            Kxy[irow,irow]=1
    
      ##  xalpha=zeros(m)
        xalpha=0.5*(uB+lB)
        xalpha0=xalpha.copy()
        for irepeat in range(self.maxiter):
          #print(irepeat)
          for irow in range(m):
            t=(-qs[irow]-dot(Kxy[irow],xalpha0))/Kxy[irow,irow]
            ## t=-qs[irow]-dot(Kxy[irow],xalpha0)
            xnew=xalpha0[irow]+t
            lbi=lB[irow]
            ubi=uB[irow]
            if lbi<xnew:
              if ubi>xnew:
                xalpha0[irow]=xnew
              else:
                xalpha0[irow]=ubi
            else:
              xalpha0[irow]=lbi
          xdiff=xalpha0-xalpha
          zerr=sqrt(sum(xdiff*xdiff))     # L2 norm error
      ##     zerr=max(abs(xdiff))     # L_infty norm error
          xalpha=xalpha0.copy()
          if zerr<z_eps:
      ##       print irepeat
            break
      # xalpha the dual solution
        return(xalpha)
    

        
    
        
            
def main():
    seed1= sys.argv[1]
    print("seed1:",seed1)
    seed2= sys.argv[2]
    print("seed2:",seed2)
    random.seed(int(seed1))
    np.random.seed(int(seed2))
    f = open("./param_file.yaml")
    param_dict = yaml.load(f)
    f.close()
    product = "acetate" # It can be tested also for "succinate" and "ethanol" or any other product that its parameters are added to the param_file.yaml
    obj = Recommender(seed1,seed2,product,param_dict)
    obj.run()
if __name__ == "__main__":
     main()

    
