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
> We will provide a toy example where we show that our method is able to learn multimodal action distributions. TODO add github link to this.

---

### Weakness 4:
> The training of DME-RL requires value function evaluation at each diffusion step, which introduces significant computational overhead. The experiments indicate that the runtime can be up to 5.5x slower than baseline algorithms.

> **My Plan:**
> Yes but as we state in L ... we wanted to show that our algorithm yields more flexibility and thus we can subsample in diff time steps which leads to smaller batches, but more update steps and thus longer training time. TOOD add experiment where we do not do that (Kaustubh).

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
> would be possible. we tried it and it is unstable unfortunately. more research in this diection would be needed.

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
> We plot the performance over env interactions. Add table with runtime performance on each task. 

---

We thank the reviewer for the thoughtful and constructive feedback. We are glad that the reviewer sees the value of the general framework, the diffusion-MDP formulation, and the empirical gains. Below we address the main concerns and clarify what we view as the core contributions and trade-offs of the paper.

**(1) Complexity of the formulation / notation clarity.**  
We agree that the current presentation is too dense in places. The augmented time index, diffusion MDP, and associated notation were introduced to make the derivation precise, but this likely obscures the intuition in the current draft. In the revision, we will simplify the exposition, improve the notation, add pseudo-code, and explain the unnormalized target distribution more carefully. We will also add more intuition for why the diffusion MDP is introduced and how it leads to the final training objective.

**(2) Is the contribution mainly a unified formulation?**  
We respectfully disagree that the paper is only a re-packaging of existing methods. The maximum-entropy diffusion MDP itself is new, and it provides a principled recipe for deriving diffusion-policy variants of max-ent RL algorithms. This is not merely an adaptation of a single prior method: from this formulation we derive three entirely novel algorithms namely DME-PPO, DME-REPPO, and DME-WPO. The only direct connection to an existing algorithm is that DPPO appears as the $\tau=0$ special case of DME-PPO, as already noted in the paper. We will state this more explicitly so the novelty of the framework and its algorithmic consequences are clearer. (TODO add references to our paper)

**(3) Why introduce the diffusion MDP at all? Could one avoid it?**  
These questions are closely related. Yes, one can derive diffusion-RL objectives without redefining the problem as an augmented MDP; in essence, this is what DIME does. However, that route leads to objectives defined over the **full reverse diffusion chain**, which require simulating and backpropagating through all diffusion steps during training. The practical advantage of our diffusion-MDP formulation is that it yields a **step-wise** objective: training can be performed at sampled diffusion steps with the corresponding diffusion-aware value function, without backpropagating through the entire chain. This makes the algorithm more flexible and better suited to settings with many diffusion steps or larger policy networks. We agree that this motivation should be stated more clearly and will emphasize it in the revision.

**(4) Compute overhead / scaling / practical benefit / performance-compute tradeoff.**  
We agree that the current experiments should better disentangle algorithmic flexibility from wall-clock cost. The framework does introduce extra computation because values are evaluated at diffusion steps. At the same time, our implementation also intentionally uses smaller diffusion-step batches with more updates to demonstrate the flexibility of step-wise training; this increases runtime and does not fully reflect the scaling advantage of avoiding full-chain backpropagation. We will clarify this distinction in the paper. We also plan to expand the performance-compute discussion by reporting results over environment interactions and adding a task-wise runtime/performance table. If available in time, we will also include an experiment without this small-batch/more-update regime to give a cleaner comparison.

Regarding practical benefit, we do not claim that diffusion policies must dominate on every standard control benchmark. Rather, our claim is that the framework gives a principled and scalable way to train expressive diffusion policies under max-ent RL. The benefit is expected to be larger when policies are more expressive, the diffusion horizon is longer, or the action distribution is more complex than what simple parametric families can capture. We will temper and clarify this claim in the revision.

**(5) Exploration / multimodality / expressiveness.**  
We agree that the current experiments do not fully isolate whether the gains come from improved exploration or from the ability to represent more complex action distributions. We will add a toy example demonstrating that the method can learn multimodal action distributions, and we will provide the corresponding implementation link. More broadly, our motivation is not limited to strict multimodality: in max-ent RL, the optimal action distribution need not belong to a simple Gaussian family even when it is not visibly multimodal. Diffusion policies provide a richer policy class for such cases. We also note that standard return metrics may fail to reveal whether a learned policy captures multimodal or otherwise non-Gaussian action structure, which is why better evaluation protocols for expressive RL policies are an important direction for future work. We will make this discussion more explicit.  (TODO add that the learning task also gets harder once multimodal actions are covered, because this leads to novel states)

**(6) Value-guided diffusion sampling / Q-guidance.**  
We agree this is an interesting direction. In principle, value-guided diffusion sampling could further improve performance or reduce the required number of diffusion steps. In our preliminary experiments, however, this was unstable, so we did not include it in the current paper. We will mention this explicitly and note it as an important direction for future work.

We thank the reviewer again for the helpful comments. We believe the revision will significantly improve the clarity of the framework, make the practical trade-offs more transparent, and better highlight when and why the proposed formulation is useful.