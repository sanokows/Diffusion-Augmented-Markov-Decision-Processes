# Review Response Plan: Maximum Entropy RL for Diffusion-Based Policies

---

## Summary
The paper introduces a maximum entropy reinforcement learning framework for diffusion-based policies, proposing a tractable lower bound for the KL divergence and a Q-function aware of latent actions. The method performs well on simulated control benchmarks.

---

## Strengths
- **Relevant problem in reinforcement learning**
- **Well-written and easy to follow**

---

## Weaknesses

### Weakness 1:
> While I appreciate the long and detailed problem description (Section 2) for clarity, I believe relative to the method's text, it is too long, such that the paper does not provide the necessary information to understand the proposed method's details. For example, the introduction of Section 2 and Section 2.1 are well-known in the maximum entropy RL literature and can be formulated in a more compact format. Similarly, although the connection to the log variance loss is interesting, I don't really understand why it is relevant to discuss the connection here.

> **My Plan:**
> Say that we will update the manuscript to have these sections more condensed. emphasize that these sections are also improtant to provide notation and context. alos usually RL is derived using jensen inequality and we derive it differently using the data processing inequality which is then more convenient when we derive the diffusion MDP. 

---

### Weakness 2:
> The paper lacks a related work section that distinguishes itself clearly from prior works in the main text. For example, the paper's title and the algorithm's name reads very similarly to a prior work [1]. It should be clearly clarified what exactly the differences are, so the reader can easily assess the tackled problem and the contributions. While information is provided in the Appendix, it is important to provide the most important differences in a compact way in the main text.

> **My Plan:**
> Thank you for pointing this out. Indeed as the reviewer has noticed the tiltle names are very similar. we will look into it whether we can change the title. However we want to emphasize that our algo is different from DIME and we will explain this more clearly in a related work section after the method section in an updated manuscript. We will also explain the difference now: So basically what DIME does is to apply the data processing inequality on L_ME (page 3, right column) to have a tracktable loss for diffusion models. Thus they arrive at $D_{KL}(q_\theta(a_{t}^{0:K}, s_t) || p_\theta(a_{t}^{0:K}, s_t) exp(\beta Q(a_t^K, s_t))/Z)$ (Eq.1) (where q is the reverse diff process and p the forward diff process) as a policy loss and the Value function is defind as $V(s_t) = \mathbb{E}_{a_t} \left [  Q(a_t^K, S_t)  - \Tau \log \frac{q_\theta(a_{t}^{0:K}, s_t)}{q_\theta(a_{t}^{0:K}, s_t)}\right ]$. Thus in order to train the policy they have to minimize Eq. 1 which requires simulating the whole reverse diffusion process for each update step and then backprop through the whole reverse diffusion process which has a memory and inference footprint of $\mathcal{O}(K)$. Furthermore, in order to compute the KL penality wiht respect tot he old policy REPPO-DIME also has to compute $D_{KL}(q_\theta_{old}(a_{t}^{0:K}, s_t)||q_\theta(a_{t}^{0:K}, s_t))$, which also has a memory and inference footprint of \mathcal{O}(K). In contrast to that the loss of our method is given by $\mathcal{L}_\text{DME}(\theta, \tilde{s}_{\tilde{t}}) =
D_{\mathrm{KL}}^\Tau\!\left(
q_\theta(\cdot | \tilde{s}_{\tilde{t}})
\;\Big\|\;\pi_\theta(a_t^k|\cdot, s_t)
\frac{\exp\big(\alpha\, Q^{q_{\theta^*}}_\text{DME}(\tilde{s}_{\tilde{t}}, \cdot)\big)}{Z(\tilde{s}_{\tilde{t}})}
\right).$ (Eq. 11 in our paper) which only requires sampling the action from the policy at the diffusion step $k$. Thus our method does not only allow for subsampling the state but also the diffsuion step $k$ (see \hat{\mathcal{L}}_{DME} page 5 right column of our paper). Thus the memory footprint is of $\mathcal{O}(\kappa)$, where $\kappa$ is the minibatch size for diffusion steps. Furhtermore the inference cost for this step is only $\mathcal{O}(1)$ in contrast to reppo DIME which is $\mathcal{O}(K)$. The same applies for computing the D_KL penalit which is $\mathcal{O}(1)$ in our case because in our case it is defined as the deviation of a reverse diffusion step and for dime it is $\mathcal{O}(K)$.  Thus this can be done in parallel for a batch of samples (See) (TODO add where we mention this in our paper but say that thaks to the reviewer we have noticed that we have not explained this well enough and we will do so better in a updated manuscript)

---

### Weakness 3:
> The paper lacks comparison to other on-policy flow/diffusion-based RL methods [2,3].

> **My Plan:**
> Look into if flow based Rl method results are comparable and can be taken. emphasize DPPO is DME-PPO at tau = 0. Make an ablation on one hard problem where we tune the temperature for DME-PPO and comapre to DPPO.

---

### Weakness 4:
> Minor, but important for readers to assess the performance: The paper reports 7 independent seeds for the tasks, but in RL, it is generally important to run on more seeds, and it has been shown beneficial to report the interquartile mean with 95% bootstrapped confidence intervals as reported in [4]. This has become common in the RL community. I highly recommend following the procedure.

> **My Plan:**
> TODO look into IQM score and report, optionally run more seeds.

---

## Key Questions for Authors

### Question 1:
> What exactly are the differences to DIME [1]? When I skim their equations, it looks very similar to what I see in this work.

> **My Plan:**
> TODO explain in detail the difference.

---

### Question 2:
> Is REPPO-DIME a variant of DIME? In this case, the paper states that DIME requires backpropagating through the whole diffusion chain, but it seems that REPPO-DIME's performance is similar to the proposed method here, while being faster in terms of run time according to the reported values in the appendix. Where is the bottleneck in terms of the runtime? What is the benefit of not backpropagating through the chain here?

> **My Plan:**
> TODO explain the difference. Write that while we do not observe performance benefits in our experiments our method gives more flexibility by subsampling diffusion steps (add reference to paper line). This would be relevant if larger policy networks are used in training (e.g. when finetuning VLAs). Or when using more diffusion steps, subsampling can be important when 100 diff steps or more are used. However using so many diff steps in rl nowadays would be too computationally demanding but might be useful in the future.

---

### Question 3:
> The paper introduces an extended MDP that incorporates the diffusion process's MDP and defines an augmented reward (Eq.13), which boils down to the environment reward, as there are no rewards during the denoising process. In this case, why exactly is introducing the MDP necessary?

> **My Plan:**
> TODO stress that importantly the logratio between forward and reverse process is present in the value function (add reference). One could alternatively pull the log ratio into the reward as we do in the appendix. add reference. but we wanted to align this more with the maxent rl formulation, where the reward is used in the value function definition. Intoducing the diffusion MDP is necessary such that we do not ahve to backprop through the whole diff process but we can subsample in diff steps k.

---

### Question 4:
> Also related to the previous question, I don't understand the justification for why the Q-function depends on the latent actions at all? How is this mathematically justified?

> **My Plan:**
> This is mathematically justified, we derive this in (add references). This is exactly our contribution.

---

### Question 5:
> Related to the previous question, the paper does not state how the Q-function is learned in general. Given that it depends on latent actions, this needs clarification. Additionally, how is the entropy term handled, as the paper correctly states that the marginal distribution of the policy is intractable?

> **My Plan:**
> we will add more clarity. add pseudocode for each algo. add a comparison to previous methods also to dime. 

---

# Rebuttal Draft

We thank the reviewer for the careful reading and constructive feedback. We are encouraged that the reviewer finds the problem relevant and the paper easy to follow. Below we address the main concerns and clarify the relation to prior work; we will incorporate these clarifications in the revision.

**(1) Length of Sec. 2 / log-variance discussion.**  
We agree that Secs. 2/2.1 can be condensed. Our goal was to make the paper self-contained and establish notation, but we can present the standard max-ent RL background more compactly. We will shorten this part and move non-essential discussion out of the main text. That said, this section is not only background: unlike the standard Jensen-style derivation often used in max-ent RL (e.g., Levine, 2018), we use the data processing inequality, which is particularly convenient for deriving our diffusion-MDP formulation and the resulting tractable surrogate objective.

**(2) Related work / difference to DIME and REPPO-DIME / runtime.**  
We agree that the distinction to DIME should be stated clearly in the main text rather than mainly in the appendix, and we will add a compact related-work paragraph after the method section. We will also reconsider the title to avoid confusion.

The key difference is that DIME applies DPI to the max-ent RL objective over the **full diffusion trajectory**, yielding a training objective defined on the entire reverse chain. As a result, policy optimization and KL control are performed over full reverse trajectories, requiring simulation/backpropagation through all diffusion steps, with memory/inference cost scaling as $\mathcal{O}(K)$. In contrast, our method derives a **diffusion-step-wise** surrogate objective on an augmented diffusion MDP. The loss is defined for a sampled augmented state $\tilde{s}_{\tilde t}=(s_t,a_t^k,k)$ and a single reverse step $a_t^{k-1}$ with a diffusion-aware Q-function. This enables subsampling not only over environment steps but also over diffusion steps, so training does not require backpropagation through the full reverse chain.

This is the main algorithmic advantage: flexibility and scalability of optimization, not necessarily a guaranteed wall-clock speedup in every small-scale benchmark. In our experiments, runtime is also affected by implementation choices (e.g., more gradient steps with smaller batches). The benefit of our formulation becomes more important for larger policies, more diffusion steps, or settings such as fine-tuning large diffusion/VLA-style policies, where full-chain backpropagation becomes increasingly costly. We will make this complexity-vs-runtime distinction explicit.

**(3) Comparison to other on-policy diffusion/flow RL methods.**  
We agree that broader comparison would strengthen the paper and will expand this where feasible. Importantly, DME-PPO reduces to DPPO at $\tau=0$; thus DPPO is a special case of our formulation, while our method generalizes it to the max-ent setting for arbitrary $\tau$. We will make this connection more prominent in the main text and, if possible within the rebuttal timeline, add an ablation comparing tuned-temperature DME-PPO against the $\tau=0$ special case.

**(4) Diffusion MDP / latent-action Q-function / critic learning / entropy term.**  
These questions are closely related. The diffusion MDP is needed not because it changes the environment reward itself, but because it changes the **state-action structure**. Once diffusion is incorporated into the control problem, the relevant state is the augmented state $\tilde{s}_{\tilde t}=(s_t,a_t^k,k)$ and the action is the next denoising step $a_t^{k-1}$. Therefore, the Q-function must depend on latent actions: this is not an extra modeling choice, but a direct consequence of the augmented MDP. Although the environment reward is only obtained at the final denoising step, the value function additionally contains the reverse/forward log-ratio term, so the diffusion process still enters the return in an essential way. This augmented formulation is precisely what enables step-wise optimization and diffusion-step subsampling. Likewise, we never need the intractable marginal over final actions: the entropy term is handled through the tractable reverse/forward log-ratio at the sampled diffusion step. We agree that the current exposition leaves too much of this implicit; in the revision we will clarify the derivation, explain critic learning more explicitly, and add clearer pseudocode.

**(5) Statistical reporting.**  
We agree that stronger statistical reporting would improve the paper. In the revision, we will report IQM with 95% bootstrapped confidence intervals, and, if additional runs finish in time, we will include more seeds as well.

We thank the reviewer again for the helpful comments. We believe these revisions will substantially improve the clarity of the paper.