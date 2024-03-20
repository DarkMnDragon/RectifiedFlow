Rectified flow offers a new perspective on understanding diffusion models and their ODE variants. Distinct from the complex SDE models, rectified flow is purely ODE-based, offering a straightforward and unified framework for generative and transfer modeling. Given the infinite possibilities of ODEs/SDEs to transfer data between two distributions, rectified flow specifically advocates for ODEs with solution paths that are straight lines. By learning straight flows, it provides a principled approach to learning ODEs with fast inference, effectively training one-step models with ODEs as intermediate steps.

Given two distributions $\pi_0$ and $\pi_1$, rectified flow implicitly learns the transport map by constructing an ODE driven by a drift force in $\mathbb R^d \times [0,1]$:
$$
\mathrm d \mathbf Z_t = \mathbf v(\mathbf Z_t ,t) \, \mathrm dt, \quad t \in [0,1], \quad \text{starting from }\mathbf Z_0 \sim \mathbf\pi_0
$$
such that $\mathbf Z_1 \sim \pi_1$ when following the ODE starting from $\mathbf Z_0 \sim \pi_0$. Injecting strong priors that intermediate trajectories are straight, we can achieve both theoretical relevance for optimal transport [cite rectified flow] and computational efficiency, as ODEs with straight paths can be simulated precisely without time discretization.

Specifically, rectified flow aims find an ODE to match (the marginal distributions of) the linear interpolation between points from distributions $\pi_0$ and $\pi_1$. Given observations $\mathbf{X}_0 \sim \pi_0$ and $\mathbf{X}_1 \sim \pi_1$, the linear interpolation $\mathbf{X}_t=t\mathbf{X}_1 + (1-t)\mathbf{X}_0, t\in [0,1]$ yields a trivial solution $\mathrm{d}\mathbf{X}_t = (\mathbf{X}_1 - \mathbf{X}_0)\mathrm{d}t$, which is not causally simulatable without $\mathbf{X}_1$. To address this, $\mathbf{X}_t$ is "projected" into a space of causally simulatable ODEs, $\mathrm{d}\mathbf{Z}_t = \mathbf{v}(\mathbf{Z}_t, t)$, by minimizing the least squares loss with respect to the direction $\mathbf{X}_1 - \mathbf{X}_0$:
$$
\min_{\mathbf v} \int_0^1  \mathbb{E}\left [\lVert{(\mathbf X_1-\mathbf X_0) - \mathbf v(\mathbf X_t, t)}\rVert^2\right] \,\mathrm{d}t.
$$
This process ensures that the $\mathbf{Z}_t$ trajectories from rectified flow closely mirror the density map of $\mathbf{X}_t$ trajectories but *reroute* at intersections to maintain causality, effectively transforming any arbitrary coupling into a deterministic one without increasing convex transport costs.

A distinctive aspect of rectified flow is its capability for "reflow," which straightens the trajectory of ODE paths. Denote the rectified flow $\boldsymbol Z^0 = \{ \mathbf Z_t: t\in[0,1]\}$ induced from $(\mathbf X_0,\mathbf X_1)$ as $\boldsymbol Z^0 = \mathsf{Rectflow}((\mathbf X_0,\mathbf X_1))$. Recursively applying this $\mathsf{Rectflow}(\cdot)$operator generates a series of rectified flows $\boldsymbol Z^{k+1} = \mathsf{Rectflow}((\mathbf Z_0^k, \mathbf Z_1^k))$, starting with $(\mathbf Z_0^0,\mathbf Z_1^0)=(\mathbf X_0,\mathbf X_1)$, where $\boldsymbol Z^k$ is the $k$-th iteration of rectified flow induced from $(\mathbf X_0,\mathbf X_1)$. This "reflow" process not only reduces transport costs but also straightens the paths of rectified flows, making $\boldsymbol Z^k$ paths straighter with increasing $k$.

Nearly straight paths offer computational benefits by reducing the time-discretization error in numerical simulations. Specifically, for an ODE \(\mathrm{d}Z_t = v(Z_t,t) \mathrm{d}t\) with straight paths, it simplifies to \(Z_t = Z_0 + t v(Z_0, 0)\), allowing for straightforward solutions.

