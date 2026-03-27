
We thank the reviewer for the careful reading and constructive feedback. We are encouraged that the reviewer finds the problem relevant and the paper easy to follow. Below we address the main concerns and clarify the relation to prior work. At [[link](https://github.com/dmerlicml/DMERL_Rebuttal)], we provide a one-to-one comparison between the equations of REPPO-DIME and DME-REPPO; we refer to these as (T-Eq. X).

**W1 Length of Sec. 2 / log-variance discussion.**  
We agree that Secs. 2/2.1 can be condensed. Our goal was to make the paper self-contained and establish notation, but the standard max-ent RL background can be presented more compactly. However, this section is not only background: unlike the standard derivation often used in max-ent RL, we use the data processing inequality (DPI), which is particularly convenient for deriving the MaxEnt diffusion-MDP formulation.

**W2, Q1, Q2 Difference to DIME / REPPO-DIME.**  
We agree that the distinction to DIME should be stated clearly in the main text, not mainly in the appendix.

The key difference is that DIME / REPPO-DIME optimize objectives over the **full reverse diffusion trajectory**. This is visible in the policy loss and KL regularizer, defined over $a_t^{0:K}$ in (T-Eqs. 1, 5). Hence both losses require simulating the whole reverse chain, giving $\mathcal{O}(K)$ time and memory. In contrast, our method defines both losses at a **single sampled reverse step** conditioned on the augmented state $\tilde s_{\tilde t}=(s_t,a_t^k,k)$; see (T-Eqs. 3, 7). This enables subsampling over diffusion steps and yields $\mathcal{O}(1)$ cost per sampled step and $\mathcal{O}(\kappa)$ memory for a minibatch of $\kappa$ sampled diffusion steps. If we train on all rollout data, we perform roughly $K/\kappa$ more updates, so the runtime advantage can disappear, while the memory advantage remains.

Thus the main algorithmic advantage is **flexibility and scalability**. The benefit of the step-wise formulation becomes more important for larger policies, longer diffusion horizons, or settings such as fine-tuning large diffusion/VLA-style policies, where full-chain backpropagation becomes increasingly costly.

**Q3 Comparison to other on-policy diffusion/flow RL methods.**  
We agree that broader comparison would strengthen the paper. At the same time, we already compare against a diffusion-policy baseline, REPPO-DIME, in Figs. 1 and 2. Moreover, as explained in L.076ff, DME-PPO reduces to DPPO at temperature 0; thus DPPO is a special case of our formulation, while our method generalizes it to the max-ent setting for arbitrary temperatures.

**Q4 Why does the Q-function depend on latent actions?**  
We justify this in Sec. 3. Starting from Eq. 8, we apply the DPI to obtain a tractable upper bound for diffusion policies. From Eq. 10 to Eq. 11, we then apply the policy gradient theorem (derived in App. G), which yields the surrogate loss in Eq. 11 together with the value- and Q-function definitions in Eq. 12. Intuitively, under a diffusion policy, the joint reverse process decomposes into a product of reverse steps. Each reverse step can therefore be interpreted as a noisy intermediate policy, so the resulting Q-function must depend on latent actions.

**Q3 Why is the diffusion MDP necessary?**  
The diffusion MDP is a direct consequence of applying the policy gradient theorem to Eq. 10. While the reward reduces to the environment reward at the last diffusion step, the value function additionally contains the log-ratios between forward and reverse diffusion transitions (see Eq. 12 and L.280ff). Since $Q = R + \mathbb{E}[V]$, one could equivalently absorb this log-ratio term into the reward, as in App. G, Eq. 39. We chose the present formulation because in max-ent RL the entropy term is usually incorporated into the value function, and we mirror that convention here. The augmented diffusion-MDP formulation is also precisely what enables step-wise optimization and diffusion-step subsampling.

**Q5 How is the Q-function learned?**  
We agree this should be explained more clearly. In the table at [[link](https://github.com/dmerlicml/DMERL_Rebuttal)], we provide the corresponding equations. In our formulation, the critic is trained on augmented tuples $(\tilde s_{\tilde t}, a_t^{k-1})$ via the loss in (T-Eq. 14), with TD-$\lambda$-style targets in (T-Eqs. 17, 18). This is the DME analogue of the DIME critic loss in (T-Eq. 13) with targets in (T-Eqs. 15, 16). We will make this clearer in the revision and add pseudocode.

**Q5 How is the intractable entropy term handled?**  
As explained around L.231ff, applying the DPI to the intractable KL yields an upper bound in which all terms are tractable; in particular, the marginal distribution is no longer required.

**W4 Statistical reporting.**  
At the link above, the reviewer can find updated figures with IQM results. The results are very similar.

We thank the reviewer again for the helpful comments. We will incorporate these clarifications in the revised manuscript to improve clarity.