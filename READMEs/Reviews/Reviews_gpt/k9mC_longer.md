# 1) Rebuttal: Maximum Entropy RL for Diffusion-Based Policies

We thank the reviewer for the careful reading and constructive feedback. We are encouraged that the reviewer finds the problem relevant and the paper easy to follow. Below, we address the main concerns and clarify the relation to prior work; we will incorporate these clarifications in the revision. At [link], we provide an overview table with a one-to-one comparison between the equations of REPPO-DIME and DME-REPPO, and we refer to these equations below as (T-Eq. X).

**W1 Length of Sec. 2 / log-variance discussion.**  
We agree that Secs. 2/2.1 can be condensed. Our goal was to make the paper self-contained and establish notation, but we agree with the reviewer that the standard max-ent RL background can be presented more compactly. However, this section is not only background: unlike the standard derivation often used in max-ent RL (e.g., Levine, 2018), we use the data processing inequality (DPI), which is particularly convenient for deriving the MaxEnt diffusion-MDP formulation.

**W2, Q1, Q2 Related work / difference to DIME and REPPO-DIME / runtime.**  
We agree that the distinction to DIME should be stated clearly in the main text rather than mainly in the appendix, and we will add a compact related-work paragraph after the method section.

The key difference is that DIME / REPPO-DIME optimize objectives over the **full reverse diffusion trajectory**. This is visible directly in the policy loss and KL regularizer, which are defined over $a_t^{0:K}$ in (T-Eq. 1) and (T-Eq. 5). As a result, both the policy-training loss and the KL-regularization loss require simulating the whole reverse chain, giving $\mathcal{O}(K)$ time and memory. In contrast, our method defines both losses at a **single sampled reverse step** conditioned on the augmented state $\tilde s_{\tilde t}=(s_t,a_t^k,k)$; see (T-Eq. 3) and (T-Eq. 7). This is what enables subsampling over diffusion steps and yields $\mathcal{O}(1)$ cost per sampled step and $\mathcal{O}(\kappa)$ memory for a minibatch of $\kappa$ subsampled diffusion steps. However, if we train on all collected rollout data, then we also perform roughly $K/\kappa$ more update steps, so the runtime advantage can disappear, while the memory advantage remains.

This is the main algorithmic advantage: flexibility and scalability of optimization. In our experiments, runtime is also affected by experimental design choices (e.g., more gradient steps with smaller diffusion-step batches). The benefit of the step-wise formulation becomes more important for larger policies, longer diffusion horizons, or settings such as fine-tuning large diffusion/VLA-style policies, where full-chain backpropagation becomes increasingly costly.

**Q2 What is the advantage of not backpropagating through the whole diffusion chain?**  
For 8 diffusion steps, as used in our experiments, this is likely not a major issue. However, in the diffusion-sampler literature, where often hundreds of diffusion steps are used, the reparameterization trick can suffer from vanishing and exploding gradients because it is applied repeatedly over a long chain. In such settings, full-chain backpropagation can become problematic and can contribute to severe mode collapse (TODO cite papers). This is precisely the regime where the advantage of step-wise optimization should become more significant.

**Q3 Comparison to other on-policy diffusion/flow RL methods.**  
We agree that broader comparison would strengthen the paper. At the same time, we would like to emphasize that we already compare against a diffusion-policy baseline, namely REPPO-DIME, in Figures 1 and 2 of our paper. Importantly, as we explain in L.076ff (left column), DME-PPO reduces to DPPO at temperature 0; thus DPPO is a special case of our formulation, while our method generalizes it to the max-ent setting for arbitrary temperatures.

**Q4 Why does the Q-function depend on latent actions?**  
In the paper, we justify this in Sec. 3. Starting from Eq. 8, we apply the DPI in order to obtain a tractable upper bound for diffusion policies. From Eq. 10 to Eq. 11, we then apply the policy gradient theorem, which we derive in detail in App. G, and show that this leads to the surrogate loss in Eq. 11 together with the corresponding value- and Q-function definitions in Eq. 12.
Intuitively, the dependence on latent actions arises because, under a diffusion policy, the joint reverse diffusion process decomposes into a product of reverse steps. Each reverse diffusion step can therefore be interpreted as a noisy intermediate policy. This is why the resulting Q-function depends on latent actions.

**Q3 Why is the diffusion MDP necessary?**  
The diffusion MDP is a direct consequence of applying the policy gradient theorem to Eq. 10. While the reward indeed reduces to the environment reward at the last diffusion step, it is important to note, as we explain around L.280ff (left column), that the value function now includes the log-ratios between the forward and reverse diffusion transition probabilities. In principle, since $Q = R + \mathbb{E}[V]$, one could equivalently absorb the log-ratio term from the value function in Eq. 12 into the reward, as we do for example in App. G, Eq. 39. This is an equivalent formulation, but we chose the present one because in max-ent RL (e.g., SAC) the entropy term is usually incorporated into the value function, and we wanted to mirror that convention here.
Also the augmented diffusion-MDP formulation is precisely what enables step-wise optimization and diffusion-step subsampling.

**Q5 How is the Q-function learned in general?**  
We agree that this should be explained more clearly. In the table at [link], we present the corresponding equations. In our formulation, the critic is trained on augmented tuples $(\tilde s_{\tilde t}, a_t^{k-1})$ via the loss in (T-Eq. 14), with TD-$\lambda$/GAE-style targets given in (T-Eqs. 17, 18). This is the DME analogue of the DIME critic loss in (T-Eq. 13) with targets in (T-Eqs. 15, 16). We will make this much clearer in the revision and add pseudocode.

**Q5 How is the intractable entropy term handled?**  
As we explain around L.231ff (right column), by applying the DPI to the intractable KL, we obtain an upper bound in which all terms become tractable. In particular, the marginal distribution is no longer required.

**W4 Statistical reporting.**  
At the link above, the reviewer can find updated figures where we report IQM results. The results are very similar, which is consistent with the fact that in highly parallel on-policy RL, sensitivity to random seeds is often smaller than in small-batch on-policy RL.

We thank the reviewer again for the helpful comments. We will incorporate these clarifications and discussions into an updated manuscript in order to improve clarity. We believe these revisions will substantially strengthen the presentation of the paper.