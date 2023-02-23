import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pingouin as pg


envs = ["Pendulum-v1","MountainCarContinuous-v0","Reacher-v2"]
ts = [30_000,75_000,40_000]
seeds = [24,62,94,19,51]

plot = True        


rew_pendulum = np.zeros((625,37))
rew_mountaincar_tmp = np.zeros((625,155))
rew_mountaincar = np.zeros((625,17))
rew_reacher = np.zeros((625,200))

ts_pendulum = np.linspace(800,29600,37)
ts_mountaincar_tmp = np.zeros((625,155))
ts_mountaincar = np.linspace(3996,67932,17)
ts_reacher = np.linspace(200,40000,200)

seed_mat = np.zeros((625,4))
k = 0
for s_act in seeds:
    for s_env in seeds:
        for s_sgd in seeds:
            for s_nn in seeds:
                seed_mat[k] = [s_act,s_env,s_sgd,s_nn]
                k += 1



for env in envs:
    k = 0
    for s_act in seeds:
        for s_env in seeds:
            for s_sgd in seeds:
                for s_nn in seeds:
                    csv_path = os.path.join(env, str(s_act) + "_" 
                                            + str(s_env) + "_" + str(s_sgd)
                                            + "_" + str(s_nn),
                                              "progress.csv")
                    tmp = np.genfromtxt(csv_path, dtype=float, delimiter=",", names=True)
                    rew_tmp = tmp["rolloutep_rew_mean"]
                    ts_tmp = tmp["timetotal_timesteps"]
                    if env == envs[0]:
                        rew_pendulum[k] = rew_tmp
                    elif env == envs[1]:
                        n = len(rew_tmp)
                        for i in range(n):
                            rew_mountaincar_tmp[k,i] = rew_tmp[i]
                            ts_mountaincar_tmp[k,i] = ts_tmp[i]
                    elif env == envs[2]:
                        rew_reacher[k] = rew_tmp
                    k += 1



for i in range(625):
    for j in range(17):
        timestep = ts_mountaincar[j]
        k = 0        
        while ts_mountaincar_tmp[i,k] < timestep:
            k +=1
        if ts_mountaincar_tmp[i,k] == timestep:
            rew_mountaincar[i,j] = rew_mountaincar_tmp[i,j]
        else:
            rew_mountaincar[i,j] = ((rew_mountaincar_tmp[i,k-1] * (ts_mountaincar_tmp[i,k] - timestep) + 
                                     rew_mountaincar_tmp[i,k] * (timestep - ts_mountaincar_tmp[i,k-1]))/
                                    (ts_mountaincar_tmp[i,k] - ts_mountaincar_tmp[i,k-1]))
            


sum_pendulum = np.sum(rew_pendulum, axis=1)
sum_mountaincar = np.sum(rew_mountaincar, axis=1)
sum_reacher = np.sum(rew_reacher, axis=1)

fin_pendulum = rew_pendulum[:,-1]
fin_mountaincar = rew_pendulum[:,-1]
fin_reacher = rew_reacher[:,-1]

learn_pendulum = np.zeros((80,37))
learn_mountaincar = np.zeros((80,17))
learn_reacher = np.zeros((80,200))




if plot:
    for i in range(625):
        for s in range(5):
            for t in range(4):
                if seed_mat[i,t] == seeds[s]:
                    learn_pendulum[5*t + s] += rew_pendulum[i]
                    learn_mountaincar[5*t + s] += rew_mountaincar[i]
                    learn_reacher[5*t + s] += rew_reacher[i]
                    for j in range(37):
                        if learn_pendulum[5*t + s + 20,j] == 0:
                            learn_pendulum[5*t + s + 20,j] = rew_pendulum[i,j]
                            learn_pendulum[5*t + s + 40,j] = rew_pendulum[i,j]
                        else:
                            learn_pendulum[5*t + s + 20,j] = np.min([learn_pendulum[5*t + s + 20,j],rew_pendulum[i,j]])
                            learn_pendulum[5*t + s + 40,j] = np.max([learn_pendulum[5*t + s + 40,j],rew_pendulum[i,j]])
                    for j in range(17):
                        if learn_mountaincar[5*t + s + 20,j] == 0:
                            learn_mountaincar[5*t + s + 20,j] = rew_mountaincar[i,j]
                            learn_mountaincar[5*t + s + 40,j] = rew_mountaincar[i,j]
                        else:
                            learn_mountaincar[5*t + s + 20,j] = np.min([learn_mountaincar[5*t + s + 20,j],rew_mountaincar[i,j]])
                            learn_mountaincar[5*t + s + 40,j] = np.max([learn_mountaincar[5*t + s + 40,j],rew_mountaincar[i,j]])
                    for j in range(200):
                        if learn_reacher[5*t + s + 20,j] == 0:
                            learn_reacher[5*t + s + 20,j] = rew_reacher[i,j]
                            learn_reacher[5*t + s + 40,j] = rew_reacher[i,j]
                        else:
                            learn_reacher[5*t + s + 20,j] = np.min([learn_reacher[5*t + s + 20,j],rew_reacher[i,j]])
                            learn_reacher[5*t + s + 40,j] = np.max([learn_reacher[5*t + s + 40,j],rew_reacher[i,j]])
    for i in range(20):
        learn_pendulum[i] = learn_pendulum[i]/125
        learn_mountaincar[i] = learn_mountaincar[i]/125
        learn_reacher[i] = learn_reacher[i]/125
        
        learn_pendulum[i+60] = abs(learn_pendulum[i+40] - learn_pendulum[i+20])
        learn_mountaincar[i+60] = abs(learn_mountaincar[i+40] - learn_mountaincar[i+20])
        learn_reacher[i+60] = abs(learn_reacher[i+40] - learn_reacher[i+20])
    
    
    mean_pendulum = np.sum(rew_pendulum, axis = 0)/625
    mean_mountaincar = np.sum(rew_mountaincar, axis = 0)/625
    mean_reacher = np.sum(rew_reacher, axis = 0)/625
    min_pendulum = np.min(rew_pendulum, axis = 0)
    min_mountaincar = np.min(rew_mountaincar, axis = 0)
    min_reacher = np.min(rew_reacher, axis = 0)
    max_pendulum = np.max(rew_pendulum, axis = 0)
    max_mountaincar = np.max(rew_mountaincar, axis = 0)
    max_reacher = np.max(rew_reacher, axis = 0)


    plt.figure(figsize=(8,6))
    plt.grid()
    plt.xlabel("Timesteps")
    plt.ylabel("Accumulated Rewards")
    plt.ylim([-1600,0])
    plt.fill_between(ts_pendulum, min_pendulum, max_pendulum, alpha=0.2, color = "b")
    plt.plot(ts_pendulum, min_pendulum, color = "b", linestyle = "--", linewidth = 1)
    plt.plot(ts_pendulum, max_pendulum, color = "b", linestyle = "--", linewidth = 1)
    plt.plot(ts_pendulum, mean_pendulum, color = "b", linestyle = "-", linewidth = 2)
    plt.title(envs[0])
    plt.savefig(envs[0] + "-overview.png")
    
    
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.xlabel("Timesteps")
    plt.ylabel("Accumulated Rewards")
    plt.ylim([-50,100])
    plt.fill_between(ts_mountaincar, min_mountaincar, max_mountaincar, alpha=0.2, color = "b")
    plt.plot(ts_mountaincar, min_mountaincar, color = "b", linestyle = "--", linewidth = 1)
    plt.plot(ts_mountaincar, max_mountaincar, color = "b", linestyle = "--", linewidth = 1)
    plt.plot(ts_mountaincar, mean_mountaincar, color = "b", linestyle = "-", linewidth = 2)
    plt.title(envs[1])
    plt.savefig(envs[1] + "-overview.png")
    
    
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.xlabel("Timesteps")
    plt.ylabel("Accumulated Rewards")
    plt.ylim([-50,0])
    plt.fill_between(ts_reacher, min_reacher, max_reacher, alpha=0.2, color = "b")
    plt.plot(ts_reacher, min_reacher, color = "b", linestyle = "--", linewidth = 1)
    plt.plot(ts_reacher, max_reacher, color = "b", linestyle = "--", linewidth = 1)
    plt.plot(ts_reacher, mean_reacher, color = "b", linestyle = "-", linewidth = 2)
    plt.title(envs[2])
    plt.savefig(envs[2] + "-overview.png")
    
    
    seed_type = ["actionspace","environment","numpy","pytorch"]
    
    
    for i in range(4):
        plt.figure(figsize=(8,6))
        plt.grid()
        plt.xlabel("Timesteps")
        plt.ylabel("Accumulated Rewards")
        plt.ylim([-1600,0])
        for j in range(5):
            plt.plot(ts_pendulum, learn_pendulum[5*i + j], linestyle = "-", linewidth = 2, label = "Seed = " + str(seeds[j]))
        plt.title(envs[0] + "-" + seed_type[i])
        plt.legend()
        plt.savefig(envs[0] + "-" + seed_type[i] + "-mean.png")
    for i in range(4):
        plt.figure(figsize=(8,6))
        plt.grid()
        plt.xlabel("Timesteps")
        plt.ylabel("Accumulated Rewards")
        plt.ylim([-50,100])
        for j in range(5):
            plt.plot(ts_mountaincar, learn_mountaincar[5*i + j], linestyle = "-", linewidth = 2, label = "Seed = " + str(seeds[j]))
        plt.title(envs[1] + "-" + seed_type[i])
        plt.legend()
        plt.savefig(envs[1] + "-" + seed_type[i] + "-mean.png")
    for i in range(4):
        plt.figure(figsize=(8,6))
        plt.grid()
        plt.xlabel("Timesteps")
        plt.ylabel("Accumulated Rewards")
        plt.ylim([-50,0])
        for j in range(5):
            plt.plot(ts_reacher, learn_reacher[5*i + j], linestyle = "-", linewidth = 2, label = "Seed = " + str(seeds[j]))
        plt.title(envs[2] + "-" + seed_type[i])
        plt.legend()
        plt.savefig(envs[2] + "-" + seed_type[i] + "-mean.png")
    
    
    for i in range(4):
        plt.figure(figsize=(8,6))
        plt.grid()
        plt.xlabel("Timesteps")
        plt.ylabel("Difference of Accumulated Rewards")
        plt.ylim([0,500])
        for j in range(5):
            plt.plot(ts_pendulum, learn_pendulum[5*i + j + 60], linestyle = "-", linewidth = 2, label = "Seed = " + str(seeds[j]))
        plt.title(envs[0] + "-" + seed_type[i])
        plt.legend()
        plt.savefig(envs[0] + "-" + seed_type[i] + "-var.png")
    for i in range(4):
        plt.figure(figsize=(8,6))
        plt.grid()
        plt.xlabel("Timesteps")
        plt.ylabel("Difference of Accumulated Rewards")
        plt.ylim([0,150])
        for j in range(5):
            plt.plot(ts_mountaincar, learn_mountaincar[5*i + j + 60], linestyle = "-", linewidth = 2, label = "Seed = " + str(seeds[j]))
        plt.title(envs[1] + "-" + seed_type[i])
        plt.legend()
        plt.savefig(envs[1] + "-" + seed_type[i] + "-var.png")
    for i in range(4):
        plt.figure(figsize=(8,6))
        plt.grid()
        plt.xlabel("Timesteps")
        plt.ylabel("Difference of Accumulated Rewards")
        plt.ylim([0,10])
        for j in range(5):
            plt.plot(ts_reacher, learn_reacher[5*i + j + 60], linestyle = "-", linewidth = 2, label = "Seed = " + str(seeds[j]))
        plt.title(envs[2] + "-" + seed_type[i])
        plt.legend()
        plt.savefig(envs[2] + "-" + seed_type[i] + "-var.png")



norm_sum_pendulum  = (sum_pendulum - np.mean(sum_pendulum))/(np.std(sum_pendulum))
norm_sum_mountaincar  = (sum_mountaincar - np.mean(sum_mountaincar))/(np.std(sum_mountaincar))
norm_sum_reacher  = (sum_reacher - np.mean(sum_reacher))/(np.std(sum_reacher))

norm_fin_pendulum  = (fin_pendulum - np.mean(fin_pendulum))/(np.std(fin_pendulum))
norm_fin_mountaincar  = (fin_mountaincar - np.mean(fin_mountaincar))/(np.std(fin_mountaincar))
norm_fin_reacher  = (fin_reacher - np.mean(fin_reacher))/(np.std(fin_reacher))

norm_sum = np.concatenate((norm_sum_pendulum, norm_sum_mountaincar, norm_sum_reacher))
norm_fin = np.concatenate((norm_fin_pendulum, norm_fin_mountaincar, norm_fin_reacher))

data_seeds = np.concatenate((seed_mat, seed_mat, seed_mat), axis = 0)
data_env_types = np.concatenate(([0 for i in range(625)], [1 for i in range(625)], [2 for i in range(625)]))



data_sum = {"s_act": data_seeds[:,0].astype(str),
            "s_env": data_seeds[:,1].astype(str),
            "s_sgd": data_seeds[:,2].astype(str),
            "s_nn": data_seeds[:,3].astype(str),
            "env_type": data_env_types.astype(str),
            "reward": norm_sum}
data_fin = {"s_act": data_seeds[:,0].astype(str),
            "s_env": data_seeds[:,1].astype(str),
            "s_sgd": data_seeds[:,2].astype(str),
            "s_nn": data_seeds[:,3].astype(str),
            "env_type": data_env_types.astype(str),
            "reward": norm_fin}


data_sum_pendulum = {"s_act": seed_mat[:,0],
                     "s_env": seed_mat[:,1],
                     "s_sgd": seed_mat[:,2],
                     "s_nn": seed_mat[:,3],
                     "reward": norm_sum_pendulum}
data_fin_pendulum = {"s_act": seed_mat[:,0],
                     "s_env": seed_mat[:,1],
                     "s_sgd": seed_mat[:,2],
                     "s_nn": seed_mat[:,3],
                     "reward": norm_fin_pendulum}
data_sum_mountaincar = {"s_act": seed_mat[:,0],
                     "s_env": seed_mat[:,1],
                     "s_sgd": seed_mat[:,2],
                     "s_nn": seed_mat[:,3],
                     "reward": norm_sum_mountaincar}
data_fin_mountaincar = {"s_act": seed_mat[:,0],
                     "s_env": seed_mat[:,1],
                     "s_sgd": seed_mat[:,2],
                     "s_nn": seed_mat[:,3],
                     "reward": norm_fin_mountaincar}
data_sum_reacher = {"s_act": seed_mat[:,0],
                     "s_env": seed_mat[:,1],
                     "s_sgd": seed_mat[:,2],
                     "s_nn": seed_mat[:,3],
                     "reward": norm_sum_reacher}
data_fin_reacher = {"s_act": seed_mat[:,0],
                     "s_env": seed_mat[:,1],
                     "s_sgd": seed_mat[:,2],
                     "s_nn": seed_mat[:,3],
                     "reward": norm_fin_reacher}


df_sum = pd.DataFrame(data = data_sum)
df_fin = pd.DataFrame(data = data_fin)

df_sum_pendulum = pd.DataFrame(data = data_sum_pendulum)
df_fin_pendulum = pd.DataFrame(data = data_fin_pendulum)
df_sum_mountaincar = pd.DataFrame(data = data_sum_mountaincar)
df_fin_mountaincar = pd.DataFrame(data = data_fin_mountaincar)
df_sum_reacher = pd.DataFrame(data = data_sum_reacher)
df_fin_reacher = pd.DataFrame(data = data_fin_reacher)



aov_sum_env_type = pg.anova(dv="reward", between="env_type", data = df_sum, detailed=True, effsize="n2")
aov_sum = pg.anova(dv="reward", between=["s_act","s_env","s_sgd","s_nn"], data = df_sum, detailed=True, effsize="n2")

"""
aov_fin_act = pg.anova(dv="reward", between="s_act", data = df_fin, detailed=True, effsize="n2")
aov_fin_env = pg.anova(dv="reward", between="s_env", data = df_fin, detailed=True, effsize="n2")
aov_fin_sgd = pg.anova(dv="reward", between="s_sgd", data = df_fin, detailed=True, effsize="n2")
aov_fin_nn = pg.anova(dv="reward", between="s_nn", data = df_fin, detailed=True, effsize="n2")
"""
aov_fin_env_type = pg.anova(dv="reward", between="env_type", data = df_fin, detailed=True, effsize="n2")
aov_fin = pg.anova(dv="reward", between=["s_act","s_env","s_sgd","s_nn"], data = df_fin, detailed=True, effsize="n2")

pg.anova(dv="reward",between="s_act",data=pd.DataFrame(data=data_sum_pendulum),detailed=True, effsize="n2")

aov_sum_pendulum1 = pg.anova(dv="reward", between=["s_act","s_env"], data = df_sum_pendulum, detailed=True, effsize="n2")
aov_sum_pendulum2 = pg.anova(dv="reward", between=["s_sgd","s_nn"], data = df_sum_pendulum, detailed=True, effsize="n2")
aov_fin_pendulum1 = pg.anova(dv="reward", between=["s_act","s_env"], data = df_fin_pendulum, detailed=True, effsize="n2")
aov_fin_pendulum2 = pg.anova(dv="reward", between=["s_sgd","s_nn"], data = df_fin_pendulum, detailed=True, effsize="n2")

aov_sum_mountaincar1 = pg.anova(dv="reward", between=["s_act","s_env"], data = df_sum_mountaincar, detailed=True, effsize="n2")
aov_sum_mountaincar2 = pg.anova(dv="reward", between=["s_sgd","s_nn"], data = df_sum_mountaincar, detailed=True, effsize="n2")
aov_fin_mountaincar1 = pg.anova(dv="reward", between=["s_act","s_env"], data = df_fin_mountaincar, detailed=True, effsize="n2")
aov_fin_mountaincar2 = pg.anova(dv="reward", between=["s_sgd","s_nn"], data = df_fin_mountaincar, detailed=True, effsize="n2")

aov_sum_reacher1 = pg.anova(dv="reward", between=["s_act","s_env"], data = df_sum_reacher, detailed=True, effsize="n2")
aov_sum_reacher2 = pg.anova(dv="reward", between=["s_sgd","s_nn"], data = df_sum_reacher, detailed=True, effsize="n2")
aov_fin_reacher1 = pg.anova(dv="reward", between=["s_act","s_env"], data = df_fin_reacher, detailed=True, effsize="n2")
aov_fin_reacher2 = pg.anova(dv="reward", between=["s_sgd","s_nn"], data = df_fin_reacher, detailed=True, effsize="n2")




aov_sum.to_csv("AOV_sum.csv", index=False)
aov_fin.to_csv("AOV_fin.csv", index=False)

aov_sum_pendulum1.to_csv("AOV_sum_pendulum1.csv",index=False)
aov_sum_pendulum2.to_csv("AOV_sum_pendulum2.csv",index=False)
aov_fin_pendulum1.to_csv("AOV_fin_pendulum1.csv",index=False)
aov_fin_pendulum2.to_csv("AOV_fin_pendulum2.csv",index=False)

aov_sum_mountaincar1.to_csv("AOV_sum_mountaincar1.csv",index=False)
aov_sum_mountaincar2.to_csv("AOV_sum_mountaincar2.csv",index=False)
aov_fin_mountaincar1.to_csv("AOV_fin_mountaincar1.csv",index=False)
aov_fin_mountaincar2.to_csv("AOV_fin_mountaincar2.csv",index=False)

aov_sum_reacher1.to_csv("AOV_sum_reacher1.csv",index=False)
aov_sum_reacher2.to_csv("AOV_sum_reacher2.csv",index=False)
aov_fin_reacher1.to_csv("AOV_fin_reacher1.csv",index=False)
aov_fin_reacher2.to_csv("AOV_fin_reacher2.csv",index=False)
