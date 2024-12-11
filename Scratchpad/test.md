This was what I got

Q27 
The ridge regression minimisation problem is given by:

$$
\theta_R^*(\mathcal{D}) = \arg\min_\theta \left[ \frac{1}{2\sigma^2} \| \mathbf{y} - \Phi \theta \|_2^2 + \frac{\lambda}{2} \|\theta\|_2^2 \right]
$$

Expanding the terms:

$$
\| \mathbf{y} - \Phi \theta \|_2^2 = (\mathbf{y} - \Phi \theta)^\top (\mathbf{y} - \Phi \theta) = \mathbf{y}^\top \mathbf{y} - 2\mathbf{y}^\top \Phi \theta + \theta^\top \Phi^\top \Phi \theta
$$

Thus, the objective function becomes:

$$
\mathcal{L}(\theta) = \frac{1}{2\sigma^2} \left[ \mathbf{y}^\top \mathbf{y} - 2\mathbf{y}^\top \Phi \theta + \theta^\top \Phi^\top \Phi \theta \right] + \frac{\lambda}{2} \|\theta\|_2^2
$$

Differentiating with respect to $\theta$:

$$
\frac{\partial \mathcal{L}(\theta)}{\partial \theta} = -\frac{1}{\sigma^2} \Phi^\top \mathbf{y} + \frac{1}{\sigma^2} \Phi^\top \Phi \theta + \lambda \theta
$$

Setting $\frac{\partial \mathcal{L}(\theta)}{\partial \theta} = 0$:

$$
\Phi^\top \Phi \theta + \lambda \sigma^2 \theta = \Phi^\top \mathbf{y}
$$

Factoring out $\theta$:

$$
\theta_R^* = (\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top \mathbf{y}
$$

Assume the true model is:

$$
\mathbf{y} = \Phi \theta_0 + \varepsilon
$$

where $\varepsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$ is Gaussian noise with zero mean and variance $\sigma^2$.

Substituting this into the ridge regression solution:

$$
\theta_R^* = (\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top (\Phi \theta_0 + \varepsilon)
$$

Decomposing:

$$
\theta_R^* = \underbrace{(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top \Phi \theta_0}_{\text{deterministic part}} + \underbrace{(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top \varepsilon}_{\text{random part}}
$$

The first term is the deterministic part, which we don't care about since it is a constant. The second term is the random part influenced by noise – so its covariance will be non-trivial.


For $\mathbf{z} = A \mathbf{x}$, where $A$ is deterministic, the covariance is given by:

$$
\text{Cov}[\mathbf{z}] = A \text{Cov}[\mathbf{x}] A^\top
$$ 

The covariance of the ridge estimator is derived from the random noise term:

$$
\text{Cov}[\theta_R^*] = \text{Cov}[(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top \varepsilon]
$$

Because $\text{Cov}(\varepsilon) = \sigma^2 I$, we have:

$$
\text{Cov}[\theta_R^*] = \sigma^2 (\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top \text{Cov}(\varepsilon) \Phi (\Phi^\top \Phi + \lambda \sigma^2 I)^{-1}
$$

Simplifying:

$$
\text{Cov}[\theta_R^*] = \sigma^2 (\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top \Phi (\Phi^\top \Phi + \lambda \sigma^2 I)^{-1}
$$

Define $V(\lambda)$ as the covariance matrix of the ridge regression estimator:

$$
V(\lambda) = \mathbb{E}_{\mathbf{X}_{\text{train}}} \left[ \text{Cov}[\theta_R^*] \right] = \mathbb{E}_{\mathbf{X}_{\mathrm{train}}}[\sigma^2(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}\Phi^\top\Phi(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}]
$$

When $\lambda = 0$ (no regularisation), the covariance reduces to:

$$
V(0) = \mathbb{E}_{\mathbf{X}_{\text{train}}} \left[ \sigma^2 (\Phi^\top \Phi)^{-1}  \right]
$$

Note that as we assume $\Phi^\top\Phi$ is invertible, we can rewrite

$$V(0)=\mathbb{E}_{\mathbf{X}_{\mathrm{train}}}[\sigma^2(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)(\Phi^\top\Phi)^{-1}(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}]\\=\mathbb{E}_{\mathbf{X}_{\mathrm{train}}}[\sigma^2(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}(2\sigma^2\lambda\mathbf{I}+\sigma^4\lambda^2(\Phi^\top\Phi)^{-1}+\Phi^\top\Phi)(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}]$$

For $\lambda > 0$, the variance reduction can be shown as:

$$
 V(0) - V(\lambda) = \sigma^2 (\sigma^2 \lambda I + \Phi^\top \Phi)^{-1} (2\sigma^2 \lambda I + \sigma^4 \lambda^2 (\Phi^\top \Phi)^{-1}) (\sigma^2 \lambda I + \Phi^\top \Phi)^{-1}
$$

$\sigma^2$ is nonzero as noise has nontrivial variance, and $\lambda>0$, so we know $2\sigma^2\lambda I$ is positive definite. We also assume that  $(\Phi^\top\Phi)^{-1}$ is positive semidefinite, so their positive weighted sum $(2\sigma^2 \lambda I + \sigma^4 \lambda^2 (\Phi^\top \Phi)^{-1})$ is positive definite.

Thus:

$$
V(\lambda) \preceq V(0)
$$

So using ridge regression where we have $\lambda >0$ instead of OLS (setting $\lambda = 0$), will reduce the variance.





Q28 

The bias of the ridge regression estimator is defined as:

$$
b(\theta_R^*) = \mathbb{E}_{\mathcal{D} \sim p_N}[\theta_R^*(\mathcal{D})] - \theta_0
$$


From the ridge regression solution, we have:

$$
\theta_R^*(\mathcal{D}) = (\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top (\Phi \theta_0 + \varepsilon)
$$

Taking the expectation over $\varepsilon$ (noting that $\mathbb{E}[\varepsilon] = 0$):

$$
\begin{aligned}
\mathbb{E}_{\mathcal{D} \sim p_N}[\theta_R^*(\mathcal{D})] &= \mathbb{E}_{\mathbf{X}_\text{train}}[(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \Phi^\top \Phi \theta_0]\\
& = \mathbb{E}_{\mathbf{X}_\text{train}}[(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1}(\Phi^\top \Phi + \lambda \sigma^2I - \lambda \sigma^2 I)^ \theta_0]\\
& = \mathbb{E}_{\mathbf{X}_\text{train}}[(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1}(\Phi^\top \Phi + \lambda \sigma^2I) \theta_0 - \lambda \sigma^2I(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \theta_0]\\
& = \mathbb{E}_{\mathbf{X}_\text{train}}[\theta_0 - \lambda \sigma^2I(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} \theta_0]\\
& = \theta_0 - \mathbb{E}_{\mathbf{X}_\text{train}}[\lambda \sigma^2I(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} ]\theta_0
\end{aligned}
$$


Then, the bias of our estimator is:
$$
b(\lambda) = \mathbb{E}_{\mathcal{D} \sim p_N}[\theta_R^*(\mathcal{D})] - \theta_0 = -\mathbb{E}_{\mathbf{X}_\text{train}}[\lambda \sigma^2I(\Phi^\top \Phi + \lambda \sigma^2 I)^{-1} ]\theta_0
$$


We then square the bias:

$$
\begin{aligned}
b(\lambda)b(\lambda)^\top &= 
\mathbb{E}_{\mathbf{X}_{\mathrm{train}}}[\sigma^4\lambda^2(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}]\theta_0\theta_0^\top\mathbb{E}_{\mathbf{X}_{\mathrm{train}}}[(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}]\\&\preceq\mathbb{E}_{\mathbf{X}_{\mathrm{train}}}[\sigma^4\lambda^2(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}\theta_0\theta_0^\top(\sigma^2\lambda\mathbf{I}+\Phi^\top\Phi)^{-1}]\end{aligned}
$$

We then have:
$$
\begin{aligned}&V(0)-V(\lambda) - b(\lambda)b(\lambda)^{\top} \succeq\mathbb{E}_{{\mathbf{X}_{{\mathrm{train}}}}}[\sigma^{2}\lambda(\Phi^{\top}\Phi{+}\sigma^{2}\lambda\mathbf{I})^{-1}\underbrace{{(\sigma^{2}[2\mathbf{I}-\lambda\theta_{0}\theta_{0}^{\top}+\sigma^{2}\lambda(\Phi^{\top}\Phi)^{-1}])}}_{:={\mathbf{E}}}(\Phi^{\top}\Phi{+}\sigma^{2}\lambda\mathbf{I})^{-1}]\end{aligned}
$$

If $\mathbf{E}$ is postive semi-definite, then $V(0)-V(\lambda) - b(\lambda)b(\lambda)^{\top} \succeq 0$, if $0 \leq \lambda \leq \frac{2}{\|\theta_0\|_2^2} \cdot$ . Here, $\sigma^2\lambda(\Phi^\top\Phi)^{-1}$ is positive semi-definite too by the same assumption in the previous question. So $2\mathbf{I}-\lambda\theta_0\theta_0^\top$ has to be positive semi-definite. This allows us to draw the bounds on $\lambda$.
