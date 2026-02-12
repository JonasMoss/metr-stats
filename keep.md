## Notes

* I also tried a finite-time singularity specification of the form $\theta \sim \gamma_0 + \gamma_1 x + c / (t^* - x)^\alpha$. But here the data are completely uninformative. The posterior on the singularity date $t^*$ is essentially the prior. This is not a surprise.
* The main weakness of this analysis is treating task times $t_j$ as known. METR takes the geometric mean of (usually 2-3) successful human baselines per task and treats it as a known quantity, discarding uncertainty. But we can model $t_j$ as a latent variable informed by the reported baselines. This would be straightforward in Stan and give more honest picture of what the data actually support, as all credible intervals will be widened. 
* When modelling $t_j$ I would probably try a Weibull distribution instead of log-normal, since the log-normal is more typically more heavy-tailed and the Weibull has stronger intuitive justification as a general distribution of task completion times.
