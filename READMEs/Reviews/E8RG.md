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
>

---

### Weakness 2:
> Limited benchmarks. Evaluations cover only DeepMind control tasks, with no high-dimensional or explicitly multimodal tasks where diffusion's expressiveness would be most justified. Also no ablation on the number of diffusion steps, which directly affects runtime, bound tightness, and Q-function learning difficulty.

> **My Plan:**
> Provide Multimodal toy example

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