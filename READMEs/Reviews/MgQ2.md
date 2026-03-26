# Review Response Plan: DME-RL Framework

---

## Summary
The paper proposes DME-RL, a general RL framework where the policy is parameterized by a diffusion model and trained under maximum entropy RL. The authors reformulate the problem as an augmented MDP, integrating diffusion sampling with RL objectives. Variants like DME-PPO, DME-REPPO, and DME-WPO are introduced, with experiments showing performance improvements over base algorithms on continuous-control benchmarks.

---

## Strengths
- **General framework unifying diffusion policies with RL**
- **Novel formulation of diffusion MDP**
- **Demonstrated performance improvements on benchmarks**

---

## Weaknesses

### Weakness 1:
> This paper introduces an augmented time index, diffusion MDP, and multiple state/action transformations, which makes the algorithm difficult to follow. Much of this complexity arises from rewriting the diffusion sampling procedure as an MDP, which may obscure the underlying intuition. Furthermore, the notation in the paper is sometimes confusing; for example, the definition of the unnormalized target distribution in the left column of line 082–087 is not clearly explained.

> **My Plan:**
> TODO write that we will explain in more detail, pseudoalgorithms, mroe text. we will aslo explain the unnormalized target distribution in more detial.

---

### Weakness 2:
> Most components are adaptations of existing methods, and the primary novelty appears to lie mainly in the unified formulation.

> **My Plan:**
> Stress that it is not an adaptation. MaxEnt-WPO is new. and the maximum entropy diff MDP is also new. From this formulation we can basically derive a diffusion based variant for each existing maxent RL algorithm (as we do DME-PPO, DME-REPPO, DME-WPO). Only DPPO is an exisiting algorithm which is a special case of our DME-PPO algo at tau = 0 (as we state int the paper see ...).

---

### Weakness 3:
> Although the paper presents a formal framework, it does not clearly address whether diffusion improves exploration or whether it effectively captures multimodal optimal actions.

> **My Plan:**
> TODO look into multimodal toy example

---

### Weakness 4:
> The training of DME-RL requires value function evaluation at each diffusion step, which introduces significant computational overhead. The experiments indicate that the runtime can be up to 5.5x slower than baseline algorithms.

> **My Plan:**
> Yes but as we state in L ... we wanted to show tha tour algorithm yields more flexibility and thus we can subsample in diff time steps which leads to smaller batches, but more update steps and thus longer training time. TOOD add experiment where we do not do that (Kaustubh).

---

### Weakness 5:
> It remains unclear whether the method scales well or provides substantial practical benefits, as the reported improvements in the experimental studies are sometimes modest.

> **My Plan:**
>

---

## Key Questions for Authors

### Question 1:
> The augmented diffusion MDP significantly increases complexity. Is there a simpler formulation that avoids introducing diffusion steps as MDP states?

> **My Plan:**
> Yes, but we do this because this yields a novel algorithm. The simplest thing would be to do this as in DIME but then you ahve to backprop through the whole diffusion chain. This is not necessary with out algorithm and thus we have a more flexible algorithm. (add references to paper)

---

### Question 2:
> The diffusion policy is sampled without value guidance. Could the authors discuss whether incorporating value-guided diffusion sampling (e.g., Q-guidance) would further improve performance or reduce the number of diffusion steps?

> **My Plan:**
> would be possible. 

---

### Question 3:
> The proposed framework introduces an augmented diffusion MDP where diffusion steps are treated as environment transitions. However, since the diffusion process is internal to the policy and the environment state remains fixed during these steps, it is unclear whether this reformulation is strictly necessary. Could the authors clarify whether similar training objectives could be derived without redefining the MDP, and what practical advantages the diffusion MDP formulation provides?

> **My Plan:**
> Yes it is possible, this would be DIMe (add line references). but then we have to backprop though the whole diff chain which can become infeasible for many diff steps or when large policy networks are used.

---

### Question 4:
> Diffusion policies are motivated by their ability to model complex and multimodal action distributions. However, the experiments focus on standard continuous control benchmarks where optimal policies are often unimodal. Could the authors provide experiments or analysis demonstrating scenarios where the additional expressiveness of diffusion policies provides a clear advantage?

> **My Plan:**
> TODO work on toy example. also stress that we do not only motivate by multimodal action distributions but also in amxent RL it is also possible that the optimal action dsitribution is not gaussian but e.g. laplacian. also write that the reward metric often does not detect multimodality. thus we believe workign on better metrics for Rl is very immportant for diffusion policyies in RL.

---

### Question 5:
> Could the authors provide a more detailed analysis of the performance–compute tradeoff, such as improvements normalized by training time or number of environment interactions?

> **My Plan:**
> We plot the performance over env interactions. Add table with runtime performance on each task. Stress that implementation details also matter (jax.lax.cond is a overhead in our method, implementationd etail because we used the reppo repository which uses jax.lax.scan everywhere)

---