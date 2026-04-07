We apologize that we were unable to address all of the reviewer’s previous questions in our first rebuttal due to the character limit. We appreciate the opportunity to clarify these points here.

**A1: Role of the unnormalized target distribution in L.82--87**

The unnormalized target distribution is
$\tilde{\pi}(a_{0:T}) = \int_{s_{0:T+1}}
 \prod_{t=0}^{T}
   p(s_{t+1}| s_t,a_t)\, \tilde{\pi}(a_t| s_t)\,
 p(s_0)\, d s_{0:T+1},$
where
$\tilde{\pi}(a_t| s_t) \propto \exp(\alpha R_\text{env}(s_t,a_t)).$
This can be rewritten as
$
\tilde{\pi}(a_{0:T}) =  \int_{s_{0:T+1}} 
 ( p(s_0) \prod_{t=0}^{T}
   p(s_{t+1}| s_t,a_t) )\,
 \exp(\alpha \sum_{t=0}^{T} R_\text{env}(s_t,a_t))\,
 d s_{0:T+1}.
$

Thus, as explained in L.88, this is a reward-weighted trajectory distribution: trajectories with higher cumulative reward receive exponentially more probability mass. The weighting is by
$\prod_{t=0}^{T} \tilde{\pi}(a_t| s_t)=\exp(\alpha \sum_{t=0}^{T} R_\text{env}(s_t,a_t)),$
not only by the instantaneous term $\exp(\alpha R_\text{env}(s_t,a_t))$ as suggested in the review comment. We will state this more clearly in our revision.

**A2: Role of the assumption that $
\tilde{\pi}(a_t|s_t) \propto \exp(\alpha R_\text{env}(s_t,a_t))
$**

This is a common intermediate assumption in variational-inference views of RL. In particular, it is closely related to Eq. 8 and the equation above Eq. 11 in [1], where the trajectory is weighted by $\exp(\sum_{t=1}^{T} r(s_t,a_t))$. In our notation,
$
\prod_{t=1}^{T} \tilde{\pi}(a_t|s_t)
\propto
\exp(\alpha \sum_{t=1}^{T} R_\text{env}(s_t,a_t)).
$

Thus, we fully agree with the reviewer that the optimal policy is not characterized purely by immediate reward.

It is important to note, however, that as stated in L.100ff, the unnormalized target is intractable and is not the final object being optimized. Instead, we apply the data processing inequality to Eq. 1, yielding the tractable reverse-KL between the joint target and variational distributions in L.110--117, directly analogous in spirit to Eq. 11 in [1]. Applying the policy gradient theorem then gives the surrogate loss in Eq. 4 (L.148ff), which aligns with maximum-entropy RL.

We therefore agree that the optimal policy $q^*(a_t|s_t)$ of the surrogate loss is proportional to $\exp(\alpha Q(s_t,a_t))$, i.e., it depends on future states through the Q-function and not only on immediate reward. This should not be confused with our $\tilde{\pi}(a_t|s_t)$, which plays the role of $p(\mathcal{O}_t |s_t,a_t)$ in Eq. 3 of [1]. Unlike in [1] we do not introduce optimality variables because we derive the surrogate loss via the data processing inequality instead, which is more convenient for deriving the diffusion-based maximum-entropy RL MDP. We thank the reviewer for pointing out that this distinction should have been made more explicit.

**A3: Clarification regarding the claimed efficiency advantage**

We want to emphasize that we do not claim that our method is more efficient in total wall-clock time. Our claim is instead a memory/flexibility benefit.

As discussed with reviewer k9mC in A2, our formulation writes the objective at a single sampled diffusion step in the augmented MDP, thereby providing additional flexibility in how computation is distributed across diffusion steps. This yields a memory/runtime trade-off rather than a runtime improvement. Concretely, the inference cost of our loss is $O(1)$ in the number of diffusion steps used in one update, and the memory cost is $O(\kappa)$. However, if $\kappa < K$, one requires $K/\kappa$ more update steps, so there is no overall time-efficiency improvement.
As also noted noted in L.422ff of our paper, the longer runtime in our experiments is partially due to our choice of $\kappa = K/2$ in order to explicitly demonstrate the diffusion-step subsampling feature.
We will make sure to state this compute/memory trade-off clearly in the revised manuscript.

**A4: Practical significance**

We believe that the practical benefit is partially addressed in our response A2 to reviewer **k9mC**. The diffusion-based maximum-entropy MDP yields a formulation with additional memory flexibility while matching REPPO-DIME performance.

Our framework also allows us to combine diffusion policies with PPO without requiring importance weights over the entire reverse diffusion process between the old and current policy, which would likely lead to high-variance updates. Instead, within our framework, the importance weights are defined over a single reverse diffusion steps.

Moreover, to the best of our knowledge, the only diffusion-RL method with such flexibility is DPPO, corresponding to DME-PPO at zero temperature. However, as shown in A3 of our first rebuttal, DPPO fails to model multimodal action distributions even on simple toy tasks, which makes the temperature extension practically meaningful.

We thank the reviewer again for the useful comments and will incorporate this feedback into the revised manuscript.

[1] Levine, Sergey. *Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review*