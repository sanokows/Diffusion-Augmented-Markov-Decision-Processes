We thank the reviewer for the positive assessment of the theoretical development, the reverse-KL motivation, and the generality of the framework. We appreciate the constructive comments on evaluation and presentation. Below we address the main points.

**A1: W2 - No experiments on multimodal tasks.**  
To address this concern, we added a toy multimodal control example that directly tests whether the compared methods can learn multimodal action distributions. A visualization of the learned behaviors is provided here ([gif](https://github.com/dmerlicml/DMERL_Rebuttal/blob/main/MultimodalActions/multi_agents.gif)), and the corresponding per-state action histograms are shown here ([histograms](https://github.com/dmerlicml/DMERL_Rebuttal/blob/main/MultimodalActions/histo.png)).

In this environment, a 2D agent outputs actions in $[-1,1]$, corresponding to movement directions between $-90^\circ$ and $90^\circ$. The reward is a double-well potential with two optima at $-45^\circ$ and $45^\circ$, making the optimal policy inherently multimodal. The state is the agent’s current orientation. After an action is taken, the transition maps it to the nearest global maximum of the double-well reward potential, while the reward is still computed from the original, unrounded action. This yields only 8 possible states, allowing us to visualize for each state the learned action histogram together with the reward landscape.

We compare **REPPO, DIME-REPPO, DPPO, DME-REPPO, DA-MDP WPO, and DME-PPO** on this task. **REPPO** suffers from mode collapse because its Gaussian policy can represent only a single mode; consequently, the agent consistently moves toward the lower-left corner regardless of initialization. **DPPO** [2], which corresponds to DME-PPO at temperature zero, also fails to represent the multimodal action distribution, leading the agent to move in cycles. By contrast, **DIME-REPPO** and our methods, namely **DME-REPPO, DA-MDP WPO, and DME-PPO**, all assign probability mass to both reward maxima for each state, demonstrating successful learning of multimodal action distributions and substantially richer behavior.

While this experiment is intentionally simple, it provides direct evidence that our approach can model and exploit multimodal action structure, unlike unimodal baselines and DPPO, which is another diffusion-based baseline.

**A2: Q1 - Figure 1 visualization.**  
We thank the reviewer for this suggestion and have incorporated the feedback in the updated figure: [link](https://github.com/dmerlicml/DMERL_Rebuttal/blob/main/IQM/all_envs_methods_iqm_eval_return.png).

**A3: Q2 - ODE-based variants / faster inference.**  
This is an interesting question, and we agree that ODE-based variants are a natural direction to consider.

A maximum-entropy RL formulation for ODE-based samplers appears possible, since the process is not fully deterministic: the initial condition is still sampled from a prior distribution. The corresponding log-density can be written as
$
\log p_\theta(x) = \log p_0(x_0) - \int_0^1 \nabla \cdot u_\theta(x_t) dt,
\quad
x_t = x_0 + \int_0^t u_s(x_s) ds.
$
Thus, the log-probability of a sample decomposes over flow-integration steps, conceptually similar to how standard Gaussian policies decompose over time or the joint probability distribution of our diffusion policies decomposes over diffusion steps. Based on this, we believe that a policy-gradient argument on the left-hand side of Eq. 1 of our paper could, in principle, also be applied directly over the flow-integration steps. However, computing $\nabla \cdot u_\theta(x_t)$ is expensive, so one would likely need Hutchinson trace estimators. This is our current mathematical intuition, but the details would need to be worked out more carefully.

ODE-based formulations may also be possible through approaches such as in [3], where flow samplers are trained with physics-informed losses [1]. At the same time, such flow-based samplers have empirically been found to perform weakly even on simple low-dimensional tasks [4].

Finally, we would also like to note that, after training, it is already possible to sample from the diffusion model by simulating the corresponding probability-flow ODE [5].

We thank the reviewer again for the helpful comments. We believe these revisions improve the updated manuscript of this paper.

**References**  
[1] Raissi, M., et al., and Karniadakis, G. E. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* JCP, 2019.  

[2] Ren, A. Z., et al. *Diffusion Policy Policy Optimization.* ICLR, 2025.  

[3] Tian, Y., et al. *Liouville flow importance sampler.* ICML, 2024.  

[4] He, J., et al. *No Trick, No Treat: Pursuits and Challenges Towards Simulation-free Training of Neural Samplers.* Frontiers in Probabilistic Inference: Learning meets Sampling, 2025.  

[5] Song, Y., et al. *Maximum likelihood training of score-based diffusion models.* NeurIPS, 2021.