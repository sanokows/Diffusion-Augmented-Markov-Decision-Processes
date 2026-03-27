# Review Response Plan: Maximum Entropy RL as Diffusion-Based Sampling

---

## Summary
The paper frames Maximum Entropy RL as diffusion-based sampling from the optimal Boltzmann policy, minimizing reverse KL between a diffusion policy and the optimal policy. The core contribution is a tractable upper bound obtained by applying the data processing inequality twice, decomposing over environment and diffusion denoising steps. This framework integrates with common algorithms like PPO and REPPO with minor modifications.

---

## Strengths
- **Clean theoretical development and motivation for reverse KL**
- **Loss decomposes per diffusion step, enabling efficient sampling and backpropagation**
- **Generality across algorithms (DME-PPO, DME-REPPO, DME-WPO)**

---

## Weaknesses

### Weakness 1:
> Runtime overhead. Experimental gains over non-diffusion baselines are moderate on standard benchmarks, while computational costs are significant.

> **My Plan:**
> This is due to our choice of having more mini batches.

---

### Weakness 2:
> Limited benchmarks. Evaluations cover only DeepMind control tasks, with no high-dimensional or explicitly multimodal tasks where diffusion's expressiveness would be most justified. Also no ablation on the number of diffusion steps, which directly affects runtime, bound tightness, and Q-function learning difficulty.

> **My Plan:**
> We will provide ultimodal toy example

---

## Key Questions for Authors

### Question 1:
> Minor suggestion on visualization. The line colors in Figure 1 are difficult to distinguish. Also, currently line styles are split into two groups. Grouping into three categories—proposed algorithms, their corresponding baselines, and other baselines—would make comparisons much easier to parse.

> **My Plan:**
> Try to provide this

---

### Question 2:
> This is out of curiosity. ME-RL explicitly requires stochastic policies, and SDE naturally provides this. The KL decomposition in Eq. 10 relies on forward and reverse processes having well-defined stochastic transition kernels, making the KL between Gaussian transitions tractable, a structure ODE flows lack. However, ODE offers significantly faster inference. Would an ODE-based variant be feasible, perhaps by injecting stochasticity only at the final step or through a hybrid formulation?

> **My Plan:**
> Add that during inference we can also sample using the probabiltiy flow ODE. A policy gradient theorem for ODEs is also possible, add math, but we are not aware that this has ever been tried. could be difficult because hudington trace estimator has to be used. Using ODEs would be possible, CITE PINN Flow papers, explain them. But also write that this performs poorly even in low dimensions (cite no tricks no treat). But it would be an interesting variant.

---

We thank the reviewer for the positive assessment of the theoretical development, the reverse-KL motivation, and the generality of the framework. We also appreciate the constructive comments on evaluation and presentation. Below we address the main points and indicate revisions we will make.

**(1) Runtime overhead / moderate gains on standard benchmarks.**  
We agree that computational cost is an important consideration. The current runtime overhead is influenced not only by the diffusion formulation itself, but also by our implementation choice to use more, smaller diffusion-step minibatches in order to demonstrate the flexibility of the step-wise training objective. This increases wall-clock time. We will clarify this more explicitly in the paper and better separate the algorithmic advantage of the method—namely, step-wise optimization without backpropagating through the full diffusion chain—from the particular training configuration used in our experiments. We will also expand the discussion of the performance–compute trade-off.

**(2) Benchmark scope / multimodality / diffusion-step ablations.**  
We agree that the current benchmark suite does not fully stress the regimes where diffusion policies are most strongly motivated. The main goal of the current experiments was to validate that the framework works reliably in standard continuous-control settings and integrates with common RL algorithms with only small changes. That said, we agree that it would be valuable to include settings where expressive or multimodal action distributions matter more directly. To address this, we plan to add a toy multimodal example showing that the method can learn multimodal action distributions. We also agree that an ablation over the number of diffusion steps would be informative, since it affects runtime, approximation quality, and critic learning difficulty. We will add such an ablation if feasible within the revision timeline, or otherwise discuss this limitation explicitly.

**(3) Figure 1 visualization.**  
Thank you for this suggestion. We agree that the current styling can be improved. In the revision, we will update the figure to use more distinguishable colors and organize line styles into clearer categories, e.g., proposed methods, their direct non-diffusion counterparts, and other baselines. This should make the comparison substantially easier to parse.

**(4) ODE-based variants / faster inference.**  
This is an interesting question. We agree that ODE-based variants are a natural direction to consider, especially because probability-flow ODEs can reduce inference cost. In principle, one could use ODE-based sampling at inference time, and it may also be possible to develop an ODE-based analogue of the policy-gradient derivation. However, the current framework relies heavily on the stochastic forward/reverse-kernel structure, which makes the KL decomposition tractable and gives a clean maximum-entropy RL interpretation. Extending this to deterministic ODE flows would require a different treatment, likely involving continuous-time change-of-variables machinery and trace-estimation techniques, which would make the derivation and implementation substantially more involved. So while we believe an ODE or hybrid stochastic/ODE variant is possible in principle, it is beyond the scope of the current paper. We will mention this as an interesting future direction and clarify that probability-flow ODE sampling could also be considered at inference time.

We thank the reviewer again for the helpful comments. We believe these revisions will improve the presentation and better clarify both the current scope of the experiments and the broader design space opened by the proposed framework.



Walker RUn: 1h30
HopperHop: 2h06
FishSwim: 2h10