# Idea

Building an optimizer that takes into account the Hessian information instead of making it time-based (like cosine annealing etc) and changes the LR based upon the curvature. We estimate the Hessian using the following:

$$
\nabla L(x + \Delta) \approx \nabla L(x) + \epsilon H(\Delta)
$$

$\Delta \to 0$

The above is the Taylor series linearization.

$$
H(x) = \frac{\nabla L(x + \Delta) - \nabla L(x)}{\Delta}
$$

But remember that the Hessian is actually a matrix — we can't find any curvature information of an area using a matrix alone. Hence to actually "understand" the curvature, we use the Rayleigh quotient to quantify it, as it is nothing but the eigenvalue and is not affected by the twisting of the vector space due to the linear transformation occurring due to the Hessian. A higher eigenvalue translates to a higher rate of change of slope, and the sign of the eigenvalue dictates the maxima/minima/plane:

- +ve eigenvalue → MINIMUM
- -ve eigenvalue → MAXIMUM
- close to 0 eigenvalue → PLANE

$$
\kappa = \frac{\nabla L^T(x)\, H(x)\, \nabla L(x)}{\nabla L^T(x)\, \nabla L(x)}
$$

But now, we don't have any data on the direction in which we need to flow, so let

$$
\Delta = \epsilon \nabla L(x)
$$

where $\epsilon \to 0$.

This is equivalent to Nesterov momentum "foresight" (check Part 3: CNN PureMath).

$$
H(x)\,\nabla L(x) = \frac{\nabla L(x + \epsilon \hat{\nabla L}(x)) - \nabla L(x)}{\epsilon}
$$

$\hat{\nabla L}(x)$ is nothing but the direction vector of the gradient:

$$
\hat{\nabla L}(x) = \frac{\nabla L(x)}{\lVert \nabla L(x) \rVert}
$$

Hence overall, the expression becomes:

$$
\kappa = \frac{\nabla L^T(x) \left\{ \dfrac{\nabla L(x + \epsilon \hat{\nabla L}(x)) - \nabla L(x)}{\epsilon} \right\}}{\nabla L^T(x)\, \nabla L(x)}
$$

Let $\nabla L(x) = g$.

$$
\kappa = \frac{g^T \left\{ \dfrac{\nabla L(x + \epsilon \hat{g}) - g}{\epsilon} \right\}}{g^T g}
$$

$$
\kappa = \frac{g^T \nabla L(x + \epsilon \hat{g}) - \lVert g \rVert^2}{\lVert g \rVert^2}
$$

Note that we capture both curvature and gradient data ($\lVert g \rVert$), and both these parameters have a say on the final LR.

Now, suppose we are working with batched data — meaning our $g$ values are not the actual, true gradients of the entire dataset, but only a batch/sample of the data, for which the loss landscape might differ from the actual dataset. Why? Because in a particular batch, the number of items in each label of the data might be at different concentrations, which affects the updates of the learnable parameters.

For example, we have 1000 images of a husky and 1000 images of a golden retriever in our data, but in one batch we have 20 images of husky and 70 images of the other class — this uneven distribution of labels directly reflects in the parameters the model wants to update, which might differ from the true gradients of the complete dataset.

Let $g_{true}$ = gradients of the entire dataset.

Hence, to overcome this problem we implement an EWMA (check out the Adam section in Part 2: Logistic Regression):

$$
\bar{\kappa}_t = \beta \bar{\kappa}_{t-1} + (1-\beta)\kappa_t
$$

Now, we need to create a function of the LR using this quantity $\bar{\kappa}_t$ — but how?

What we want to do is:

| κ | ‖g‖ | Optimal η value change |
| :--- | :---: | :--- |
| +ve High | High | decrease |
| -ve High | Low | increase |
| +ve Low | High | increase |
| -ve Low | Low | increase |
| -ve High | High | decrease |
| +ve High | Low | decrease |
| -ve Low | High | increase |
| +ve Low | Low | increase |

Define a function:

$$
risk(\bar{\kappa}, \lVert g \rVert) = \frac{1}{\text{ReLU}(\bar{\kappa}) + \dfrac{1}{\lVert g \rVert}} + \text{RevReLU}(\lVert g \rVert \cdot \bar{\kappa})
$$

Where RevReLU is defined as:

$$
\text{RevReLU}(x) =
\begin{cases}
-1 \cdot x & \text{for } x < 0 \\
0 & \text{for } x \geq 0
\end{cases}
$$

This custom risk function still has an issue with the second term: $\text{RevReLU}(\lVert g \rVert \cdot \bar{\kappa})$. Here $\lVert g \rVert \cdot \bar{\kappa}$ can grow arbitrarily large when $\bar{\kappa}$ is huge and negative. To tackle this issue we can introduce a squashing function — $\ln(1 + \text{RevReLU}(\lVert g \rVert \cdot \bar{\kappa}))$, for example, or even $\sigma(\text{RevReLU}(\lVert g \rVert \cdot \bar{\kappa}))$.

This gives us a final (v2) function of:

$$
risk(\bar{\kappa}, \lVert g \rVert) = \frac{1}{\text{ReLU}(\bar{\kappa}) + \dfrac{1}{\lVert g \rVert}} + \sigma(\text{RevReLU}(\lVert g \rVert \cdot \bar{\kappa}))
$$

For some reason, an EMA of $\kappa^2$ works better than an EMA of $\kappa$, for which I am unable to find the reason — but it beats other optimizers on N-dimensional defined terrains, even though that completely destroys the purpose of the curvature-based LR calculation.

Further, the function I created works very well with terrains that have a positive-curvature character, and its efficacy drops while dealing with negative curvature and sinusoidal terrains.

---

# Muon

## SVD: Singular Value Decomposition (in short)

Suppose we have a matrix that exists in $\mathbb{R}^2$ and another one in $\mathbb{R}^3$. To go from one dimension to the other, we use matmul with a rectangular matrix — a rectangular matrix possesses the power to erase/add dimensions.

Symmetric matrices have eigenvectors that are perpendicular to each other, and the matrix of the normalized eigenvectors rotates the vector space so that the eigenvectors are the new basis vectors. If we take the transpose of the normalized eigenvector matrix, we rotate the eigenvectors back to the original basis vectors.

To force symmetry out of other types of matrices (which can be rectangular), we do this — for example, let $A$ be a rectangular matrix:

$A^TA$ and $AA^T$ are both symmetric.

Let $A$ be $m \times n$, then $A^TA$ is $n \times n$ and $AA^T$ is $m \times m$.

Now obviously $A^TA$ has $n$ eigenvectors and $AA^T$ has $m$. A matrix of the eigenvectors of $A^TA$ is known as the Right-Singular Vectors, and that of $AA^T$ is known as the Left-Singular Vectors. Also, $A^TA$ and $AA^T$ are PSD (Positive Semi-Definite) matrices, meaning the eigenvalues of both matrices are positive and numerically equal. In case $n$ does not equal $m$, the lower value between $n$ and $m$ is the total number of eigenvalues, and the other extra $(m-n)$ eigenvalues are 0.

Also, the square roots of these eigenvalues are the "singular values" of matrix $A$.

### SVD

SVD is just a way to rewrite a matrix into 3 different matrices:

$$
A = U \Sigma V^T
$$

where $U$ is the matrix of eigenvectors of the Left-Singular Vectors, written in descending order left to right based on singular value (square root of eigenvalue).

$\Sigma$ is the rectangular matrix composed of only the singular values of the matrix along its diagonal (or half-diagonal, since it's not square).

$V^T$ is the transpose of the Right-Singular Vector matrix, written in descending order left to right based on singular value.

Geometrically: first we apply the $V^T$ transform, which rotates the eigenvectors of the matrix under transform to the basis vectors of the vector space, such that the eigenvector with the largest singular value gets transformed to the x-axis first, and so on. Then we apply the $\Sigma$ transform, which scales the vector by a factor of the singular values of the matrix and reduces/adds dimensions being transformed. Finally we apply the $U$ transform to rotate the basis vectors along the eigenvectors of the matrix.

## Note: fractional powers of a matrix

Using the spectral theorem:

$$
A = Q \Lambda Q^T
$$

where $Q$ is the orthonormal matrix of eigenvectors as columns, and $\Lambda$ is the diagonal matrix of eigenvalues.

$$
A^n = Q \Lambda^n Q^T
$$

Since $\Lambda$ is a symmetric (diagonal) matrix:

$$
\Lambda = \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix}
$$

$$
\Lambda^n = \begin{pmatrix} \lambda_1^n & 0 \\ 0 & \lambda_2^n \end{pmatrix}
$$

But why do $Q$ and $Q^T$ not get transformed?

Because of a pattern:

$$
AA = Q \Lambda Q^T Q \Lambda Q^T
$$

Now $Q^T Q = I$ since $Q$ is orthonormal:

$$
A^2 = Q \Lambda^2 Q^T
$$

In general:

$$
A^n = Q \Lambda^n Q^T
$$

## Relation between SVD and Muon

In Muon, we decompose the gradient matrix ($G$) into $U$, $\Sigma$, and $V^T$. But remember that $\Sigma$ is a matrix of singular values which scales the matrix $G$ — this can be a problem. Suppose our $\Sigma$ is (if $G$ is symmetric):

$$
\begin{pmatrix} 1000 & 0 \\ 0 & 1 \end{pmatrix}
$$

This is a problem — we are essentially preferring the direction with singular value 1000 much more than the other, leading to "ill-conditioning." We explore the direction with the larger singular value far more than the other.

To tackle this, Muon deletes the $\Sigma$ matrix entirely, so the decomposition after deletion becomes:

$$
\tilde{G} = UV^T
$$

But the problem is, this logic seems almost too obvious — SVD was popularized in computer science as early as 1970, and Muon arrived as late as 2024. Probably because SVD itself is expensive to compute directly.

### Removing $\Sigma$

$$
G = U \Sigma V^T
$$

$$
G^T G = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T
$$

Since $U$ is orthonormal, $U^TU = I$:

$$
G^T G = V \Sigma^T \Sigma V^T
$$

Since $\Sigma$ is diagonal, $\Sigma^T = \Sigma$:

$$
G^T G = V \Sigma^2 V^T
$$

Taking power $-1/2$ on both sides:

$$
(G^T G)^{-1/2} = V \Sigma^{-1} V^T \quad \text{(by spectral theorem)}
$$

Pre-multiplying by $G$:

$$
G (G^T G)^{-1/2} = U \Sigma V^T V \Sigma^{-1} V^T
$$

Since $V$ is orthonormal, $V^TV = I$:

$$
G (G^T G)^{-1/2} = U \Sigma \Sigma^{-1} V^T
$$

$$
G (G^T G)^{-1/2} = U V^T
$$

Hence we have simplified $UV^T$ in terms of $G$ (gradient). Now we just need a way to estimate $(G^T G)^{-1/2}$ without again using SVD. We have a technique to do so called the "Newton-Schulz method" — though it's a derivative of the original method rather than its exact textbook form.

Suppose we have a matrix:

$$
A = G^T G
$$

We need $A^{-1/2}$, meaning:

$$
AX^2 = I
$$

where $X = A^{-1/2}$. So if we create a general equation to estimate $X$:

$$
E = AX^2 - I
$$

As $X \to A^{-1/2}$, $E \to 0$.

Creating and isolating $A^{-1/2}$:

$$
E + I = AX^2
$$

Taking power $-1/2$ on both sides:

$$
(E+I)^{-1/2} = (AX^2)^{-1/2}
$$

$$
(E+I)^{-1/2} = A^{-1/2} X^{-1}
$$

Post-multiplying by $X$:

$$
(E+I)^{-1/2}X = A^{-1/2}
$$

Now we want to estimate $A^{-1/2}$ iteratively — define $A^{-1/2}$ to be the result of an iteration of the operation $(E+I)^{-1/2}X$, so $A^{-1/2} = X_{new}$.

Also note that with each iteration, $E$ gets smaller, so we can use a binomial expansion to estimate $(E+I)^{-1/2}$:

$$
(E+I)^{-1/2} \approx I - \frac{1}{2}E \quad \text{(linearization)}
$$

$$
(I - \frac{1}{2}E)X = A^{-1/2}
$$

Using the definition of $E$:

$$
\left(I - \frac{1}{2}(AX^2 - I)\right)X = A^{-1/2}
$$

$$
\frac{1}{2}X_{old}(3I - AX_{old}^2) = X_{new}
$$

Hence, after $N \to \infty$ iterations, $X_{new} = A^{-1/2}$. This equation is a really smart and effective way to estimate $(G^T G)^{-1/2}$.

That's it — that's Muon. Now we just update the parameters:

$$
P_n = P_{n-1} - \eta\, UV^T
$$

---

# BFGS & L-BFGS (Broyden-Fletcher-Goldfarb-Shanno and limited-memory BFGS)

## Newton's method

$$
P_{n+1} = P_n - H^{-1} \nabla L(x)
$$

with the variables having their usual meanings.

### Derivation

We want to explore our landscape using small steps: $x + h$.

Taylor expansion of $f(x)$ about a small $h$, where $f(x)$ is our landscape function:

$$
f(x + h) \approx f(x) + \nabla f(x)^T h + \frac{1}{2} h^T H h
$$

The above function is an estimation of our landscape $f(x)$. If we differentiate the above equation with respect to $h$, we find the stationary points (maxima/minima/plateau) of the landscape at that point. Keep in mind our main motivation is to find an expression for $h$.

$$
0 = \nabla_h \left[ f(x) + \nabla f(x)^T h + \frac{1}{2} h^T H h \right]
$$

Using the chain rule (not expanding the sums and differentiating term-by-term here, like in the PureMath series):

$$
0 = \nabla f(x) + \frac{1}{2}\left(h^T (\nabla_h Hh)^T + Hh\, \nabla_h h^T\right) \quad \text{(transposed to match dimensions)}
$$

$$
0 = \nabla f(x) + \frac{1}{2}(2Hh)
$$

$$
0 = \nabla f(x) + Hh
$$

Let $\nabla f(x) = G$:

$$
-G = Hh
$$

Pre-multiplying by $H^{-1}$:

$$
-H^{-1}G = h
$$

Hence, going by our earlier statement, $x_{new} = x_{old} + h$:

$$
x_{n+1} = x_n - H^{-1}G
$$

## DFP

The theory explained below is of DFP (Davidon-Fletcher-Powell), where we estimate the inverse of $H$.

Calculating the Hessian for trillions of parameters is impossible, so we estimate the inverse Hessian $B \approx H^{-1}$ such that:

$$
x_{n+1} = x_n - B_n G_n
$$

Define certain values:

$$
s_n = x_{n+1} - x_n
$$

$$
y_n = G_{n+1} - G_n
$$

Using the Taylor expansion of $G_n$ around $s_n$:

$$
G(x_n + s_n) \approx G(x_n) + Hs_n
$$

$$
G(x_n + x_{n+1} - x_n) - G(x_n) \approx Hs_n
$$

$$
y_n = Hs_n \quad \text{(this condition is known as the secant equation)}
$$

$$
H^{-1}y_n = s_n
$$

Now we need to estimate $H^{-1}$ recursively, so let's have a variable $B_n$ such that:

$$
B_{n+1}y_n \approx s_n
$$

$$
(B_n + \delta)y_n = s_n
$$

We know that the Hessian is symmetric, hence to ensure symmetry in our $B$, we must ensure $\delta$ is symmetric (we initialize $B$ from the identity matrix, so it is already symmetric). Hence we initialize $\delta$ to be a rank-2 tensor using outer products:

$$
\delta = \alpha u u^T + \beta v v^T
$$

where $\alpha$ and $\beta$ are constants, and $u$ and $v$ are vectors with shapes matching $y_n$ that we need to find. Overall, this should satisfy the inverse secant equation:

$$
(B + \alpha u u^T + \beta v v^T)y_n = s_n
$$

$$
By_n + \alpha u (u^T y_n) + \beta v (v^T y_n) = s_n
$$

Now $(v^T y_n)$ and $(u^T y_n)$ are constants ($c_1$ and $c_0$) since they have the same shape as $u$, $v$, and $y_n$:

$$
c_0 \alpha u + c_1 \beta v = s_n - By_n
$$

The reason we are using rank-2 tensors and not a rank-1 tensor is because rank-1 tensors produce vectors that are always parallel to $y_n$, whereas our $s_n - By_n$ term can be anything. So to have more freedom we use rank-2 tensors — rank 3, 4, ..., N are also possible but computationally heavy.

Continuing the algorithm, we can set $u$ and $v$ to be $s_n$ and $B_n y_n$, since those are the vectors we know, and they make the LHS and RHS comparable — so we only need to find the scalar values now by direct comparison:

$$
c_0 \alpha s_n + c_1 \beta B_n y_n = s_n - By_n
$$

By direct comparison, we can infer that:

$$
\alpha = \frac{1}{c_0}, \quad \beta = -\frac{1}{c_1}
$$

$$
\alpha = \frac{1}{s_n^T y_n}, \quad \beta = -\frac{1}{y_n^T B_n y_n} \quad (B_n^T = B_n \text{, by symmetry})
$$

$$
\delta = \frac{s_n s_n^T}{s_n^T y_n} - \frac{B_n y_n y_n^T B_n^T}{y_n^T B_n y_n}
$$

$$
\delta = \frac{s_n s_n^T}{s_n^T y_n} - \frac{B_n y_n y_n^T B_n}{y_n^T B_n y_n} \quad \text{(by symmetry)}
$$

$$
B_{n+1} = B_n + \frac{s_n s_n^T}{s_n^T y_n} - \frac{B_n y_n y_n^T B_n}{y_n^T B_n y_n}
$$

That's it for the estimation.

### BFGS

Instead of approximating the inverse Hessian directly, we can approximate the Hessian itself — which is what is done in the actual BFGS algorithm:

$$
M_n \approx H
$$

Using the secant equation:

$$
M_{n+1}s_n = y_n
$$

and following the same procedure:

$$
M_{n+1} = M_n + \frac{y_n y_n^T}{y_n^T s_n} - \frac{M_n s_n s_n^T M_n}{s_n^T M_n s_n}
$$

But this is the Hessian, not the inverse, so we need to invert the entire thing:

$$
M_{n+1}^{-1} = \left( M_n + \frac{y_n y_n^T}{y_n^T s_n} - \frac{M_n s_n s_n^T M_n}{s_n^T M_n s_n} \right)^{-1}
$$

We can invert it directly, costing us a time complexity of $O(n^3)$, or we can use the Sherman-Morrison-Woodbury formula.

### Sherman-Morrison formula

Suppose we have a matrix $A$ and we know its inverse $A^{-1}$, but then we perturb $A$ by a rank-1 matrix, $A + uv^T$, and now we want to find the inverse of this new matrix efficiently. For this we have the Sherman-Morrison (SM) formula.

$$
M = A + uv^T
$$

$$
M^{-1} = A^{-1} + X \quad \text{(where $X$ is a rank-1 matrix)}
$$

We only have the matrices $A$, $A^{-1}$, $u$, and $v$ in our collection, so we must construct a general rank-1 matrix out of these. First attempt:

$$
X = \alpha u v^T
$$

$$
(A + uv^T)(A^{-1} + \alpha u v^T) = I
$$

$$
I + \alpha A u v^T + u v^T A^{-1} + \alpha u v^T u v^T = I
$$

$$
\alpha A u v^T + u v^T A^{-1} + \alpha u v^T u v^T = 0
$$

We cannot proceed further here, since nothing factors out cleanly. So let's try a different ansatz for $X$:

$$
X = \alpha A^{-1} u v^T A^{-1}
$$

$$
(A + uv^T)(A^{-1} + \alpha A^{-1} u v^T A^{-1}) = I
$$

$$
I + \alpha u v^T A^{-1} + u v^T A^{-1} + \alpha u v^T A^{-1} u v^T A^{-1} = I
$$

$$
\alpha u v^T A^{-1} + u v^T A^{-1} + \alpha u v^T A^{-1} u v^T A^{-1} = 0
$$

$$
u\left(\alpha + 1 + \alpha v^T A^{-1} u\right) v^T A^{-1} = 0
$$

$$
\alpha + 1 + \alpha v^T A^{-1} u = 0
$$

$$
\alpha = \frac{-1}{1 + v^T A^{-1} u}
$$

$$
X = \frac{-(A^{-1} u v^T A^{-1})}{1 + v^T A^{-1} u}
$$

$$
M^{-1} = A^{-1} - \frac{A^{-1} u v^T A^{-1}}{1 + v^T A^{-1} u}
$$

Applying this exact logic twice on our DFP result yields the BFGS update:

$$
B_{n+1} = \left(I - \frac{s_n y_n^T}{y_n^T s_n}\right) B_n \left(I - \frac{y_n s_n^T}{y_n^T s_n}\right) + \frac{s_n s_n^T}{y_n^T s_n}
$$

where $M_n^{-1}$ is renamed to $B_n$.

And in L-BFGS, we don't store $B$ but instead store $s$ and $y$, and directly approximate the Hessian from those.

## Wolfe's conditions and changing LR

(Doesn't seem that intuitive at first, but bear with it.)

The BFGS algorithm works given that the Hessian approximant stays positive-definite, so to guard this we have two conditions that need to be satisfied:

**1. Armijo condition:**

$$
f(x+\eta p) \leq f(x) + a\eta \nabla f(x)^T p
$$

where $p = -B\nabla f(x)$ and $a \in (0, 1)$. This prevents the learning rate from dying out.

**2. Curvature condition:**

$$
\nabla f(x+\eta p)^T p \geq b \nabla f(x)^T p
$$

This forces us to move in a direction where the slope is decreasing.

---

# AdaHessian

Just like Adam, but instead of the velocity being an EWMA of gradient squares, it's an EWMA of Hessian squares.

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) G_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) H_t^2
$$

$$
p_{t+1} = p_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

And, of course, we estimate the Hessian.

But here — unlike BFGS and DFP — we only want the diagonal entries of the Hessian, because all the other terms act more as a correlation than a degree of curvature along the axis of a given parameter. To extract the diagonal terms, we have the Hutchinson estimator.

## Hutchinson's estimator

In this estimation we use a Rademacher vector. A vector sampled from the Rademacher distribution is defined as a Rademacher vector ($z$):

$$
\mathbb{E}[z_i] = 0
$$

$$
\mathbb{E}[z_i z_j] = 0 \quad \text{for } i \neq j
$$

$$
\mathbb{E}[z_i^2] = 1
$$

meaning the mean is 0, the covariance is 0 (completely independent terms), and the variance is 1.

For an operation such as $z \odot (Hz)$, the expectation for a particular row $i$ yields:

$$
\mathbb{E}\left[z \odot (Hz)\right]_i = \mathbb{E}\left[z_i \sum_{j=1}^n H_{ij} z_j\right] = \mathbb{E}\left[H_{ii} z_i^2 + \sum_{j \neq i} H_{ij} z_i z_j\right]
$$

In matrix format:

$$
\begin{bmatrix}
\mathbb{E}\left[H_{11}z_1^2 + \sum_{j \neq 1} H_{1j}z_1z_j\right] \\
\mathbb{E}\left[H_{22}z_2^2 + \sum_{j \neq 2} H_{2j}z_2z_j\right] \\
\vdots \\
\mathbb{E}\left[H_{nn}z_n^2 + \sum_{j \neq n} H_{nj}z_nz_j\right]
\end{bmatrix}
$$

Since the covariance is 0:

$$
\mathbb{E}\left[z \odot (Hz)\right]_i = \mathbb{E}\left[H_{ii} z_i^2\right] = H_{ii}\,\mathbb{E}[z_i^2] = H_{ii}
$$

Hence we have proved that $\mathbb{E}\left[z \odot (Hz)\right]$ gives us the diagonal elements of $H$.

Now the problem lies in finding $Hz$, which again can be found using the HVP (Hessian-Vector Product).

$$
\nabla f(x) =
\begin{bmatrix}
\dfrac{\partial f}{\partial x_1} \\
\dfrac{\partial f}{\partial x_2} \\
\vdots \\
\dfrac{\partial f}{\partial x_n}
\end{bmatrix}
$$

$$
g(x) = \nabla f(x)^T z = \sum_{i=1}^n \frac{\partial f}{\partial x_i} z_i
$$

where $z$ is the Rademacher vector. Now, a thing to observe is what happens when we differentiate $g(x)$ again:

$$
\nabla_x\left(\nabla f(x)^T z\right) =
\begin{bmatrix}
\dfrac{\partial}{\partial x_1} \displaystyle\sum_{i=1}^n \dfrac{\partial f}{\partial x_i} z_i \\
\dfrac{\partial}{\partial x_2} \displaystyle\sum_{i=1}^n \dfrac{\partial f}{\partial x_i} z_i \\
\vdots \\
\dfrac{\partial}{\partial x_n} \displaystyle\sum_{i=1}^n \dfrac{\partial f}{\partial x_i} z_i
\end{bmatrix}
$$

Notice that this is nothing but the Hessian multiplied by the Rademacher vector.

Expanding the first row:

$$
\frac{\partial^2 f}{\partial x_1^2} z_1 + \frac{\partial^2 f}{\partial x_1 \partial x_2} z_2 + \dots + \frac{\partial^2 f}{\partial x_1 \partial x_n} z_n
$$

And so on for all the rows — hence it is nothing but our $Hz$:

$$
\nabla_x\left(\nabla f(x)^T z\right) = Hz
$$
