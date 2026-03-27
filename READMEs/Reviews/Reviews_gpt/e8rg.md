We thank the reviewer for the positive assessment of the theoretical development, the reverse-KL motivation, and the generality of the framework. We also appreciate the constructive comments on evaluation and presentation. Below we address the main points and indicate the revisions we will make.

**W2 No experiments on multimodal tasks.**  
We added a toy example ([link](https://github.com/dmerlicml/DMERL_Rebuttal/blob/main/MultimodalActions/multi_agents.gif)) showing that the method can learn multimodal action distributions. A 2D agent outputs actions in \([-1,1]\), corresponding to movement directions between \(-90^\circ\) and \(90^\circ\). Rewards are given by a double-well potential with optima at \(-45^\circ\) and \(45^\circ\). The state is the agent’s current orientation. The transition rounds the resulting action to the closest global minimum of the double well, while the reward is computed from the unrounded action. As a result, there are only 8 possible states for each of which we can plot a corresponding action histogram ([link](https://github.com/dmerlicml/DMERL_Rebuttal/blob/main/MultimodalActions/histo.png)) and compare it to the reward landscape. We compare REPPO, DIME-REPPO, and DME-REPPO on this environment. REPPO exhibits mode collapse because a Gaussian policy can represent only one mode, causing the agent to move toward the lower-left corner regardless of initialization. In contrast, DIME-REPPO and DME-REPPO cover both reward maxima for each state, leading to richer behavior in all directions.

**Q1 Figure 1 visualization.**  
Thank you for this suggestion. Our current plotting scheme is organized as follows: each vanilla Gaussian-policy method is shown with a dotted line in a specific color, while the corresponding diffusion-based variant uses the same color with a solid line. REPPO-DIME and WPO currently use colors similar to their respective counterparts. We agree that it would be more consistent to also show WPO with a dotted line. At [link], we provide an updated figure in which we incorporated the reviewer’s feedback.

**Q2 ODE-based variants / faster inference.**  
This is an interesting question, and we agree that ODE-based variants are a natural direction to consider.

A maximum-entropy RL formulation for ODE-based samplers appears possible, since the process is not fully deterministic: the initial condition is still sampled from a prior distribution. The corresponding log-density can be written as
\[
\log p_\theta(x) = \log p_0(x_0) - \int_0^1 \nabla \cdot u_\theta(x_t)\, dt,
\qquad
x_t = x_0 + \int_0^t u_s(x_s)\, ds.
\]
Thus, the log-probability of a sample decomposes over flow-integration steps, in a way that is conceptually similar to gaussian or diffusion policies in RL. Based on this, we believe that a policy-gradient argument on the left-hand side of Eq. 1 could in principle also be applied directly over the flow-integration steps. However, computing \(\nabla \cdot u_\theta(x_t)\) is expensive, so one would likely need to rely on the Hutchinson trace estimators. This is our current intuition, but we think that the mathematical details would need to be worked out more carefully.

ODE-based formulations may also be possible through approaches such as (Tian et al. 2024), where flow samplers are trained with physics-informed losses (Raissi et al. 2019). At the same time, such flow-based samplers have empirically been found to perform weakly even on low-dimensional tasks (He et al. 2025). 

Finally, we would also like to note that, after training, it is already possible to sample from the diffusion model by simulating the corresponding probability-flow ODE ( Song et al. 2021 (Eq. 7)).

We thank the reviewer again for the helpful comments. We believe these revisions will improve the presentation and better clarify both the current scope of the experiments and the broader design space opened by the proposed framework.

Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational physics 378 (2019): 686-707.

Tian, Yifeng, Nishant Panda, and Yen Ting Lin. "Liouville flow importance sampler." Proceedings of the 41st International Conference on Machine Learning. 2024.

He, Jiajun, et al. "No Trick, No Treat: Pursuits and Challenges Towards Simulation-free Training of Neural Samplers." Frontiers in Probabilistic Inference: Learning meets Sampling. 2025.

Song, Yang, et al. "Maximum likelihood training of score-based diffusion models." Advances in neural information processing systems 34 (2021): 1415-1428.