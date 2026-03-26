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
> Say that we will update the manuscript to have these sections more condensed. emphasize that these sections are also improtant to provide notation and context. alos usually RL is derived using jensen inequality and we derive it differently using the data processing inequality which is then more convenient when we derive the diffusion MDP. https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html

---

### Weakness 2:
> The paper lacks a related work section that distinguishes itself clearly from prior works in the main text. For example, the paper's title and the algorithm's name reads very similarly to a prior work [1]. It should be clearly clarified what exactly the differences are, so the reader can easily assess the tackled problem and the contributions. While information is provided in the Appendix, it is important to provide the most important differences in a compact way in the main text.

> **My Plan:**
> Thank you for pointing this out. Indeed as the reviewer has noticed the tiltle names are very similar. we will look into it whether we can change the title. However we want to emphasize that our algo is different from DIME and we will explain this more clearly in a related work section after the method section in an updated manuscript. We will also explain the difference now: So basically what DIME does is to apply the data processing inequality on L_ME (page 3, right column) to ahve a tracktable loss for diffusion models. Thus they arrive at $D_{KL}(q_\theta(a_{t}^{0:k}) || exp()/Z)$

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