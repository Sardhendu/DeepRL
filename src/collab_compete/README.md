




### Cautions:
1. This is a multiagent task, where the agent may not learn anything till 2000 episode. Be patient!
2. The agent could show a combined score of >0.5 for more that 100 consecutive episode but there can be scenarios 
that the score decreases after that, which means the agent learning is unstable.
3. 

### What helped:
1. Increasing the softupdate iteration =32 and soft update TAU from 0.001 to 0.01 helped with some spikes in the few episodes. Would hard update be beneficial?
2. Having HARD-update at 1000 instead of soft-update actually helped. agent_1 which wasnt learning anything performed better than agent_0. THere were multiple spikes where the average score over 100 consecutive window was more that 0.04 which was great given the fact that the score were always closer to -0.004. This was really a good trial after a long time.
3. After converting the code to MADDPG with Centralized critic learning hard update didnt perform as before (This is weird). However soft-update with Tau-0.01 and soft-update after 32 iteration, while learning was set at every 16 iteration helped the model learn great amount amid the training where the score reached an average of 0.08 - 0.1 from (episode 400-1200). But then after many iteration the score dropped.
   -> Was this becasue we have noise decay till 0.001, should we decay noise all the way to 0.000001.
   -> Should we decrease the TAU =0.001 and let model shift slowely towards the target.  


### TODO's
1. Build my own Unity Environment: [LINK](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md)
2. Implement Centralized learning for multi-agent (MADDPG)
3. Look at how efficient in Entropy Regularizer