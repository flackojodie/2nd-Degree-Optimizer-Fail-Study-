# Idea:

Building an optimizer that takes into account the hessian information instead of making it on time (like cosine annealing etc) and changes the LR based upon the curvature, and we estimate the hessian using the following:

$$
∇L(x + \Delta) ≈ ∇L(x) + ϵ H(\Delta)
$$

$Δ \to 0$

the above is the taylor series linearization,

$$
H(x) = \frac{∇L(x + \Delta) - ∇L(x)} \Delta
$$

But remember that hessian is actually a matrix, we cant find any curvature information of an area using a matrix hence to actually "understand" the curvature, we use the rayleigh quotient to actually quantify the curavture as it is nothing but the eigenvalue and is not affected by the twisting of the vector space due to the linear transformation occuring due to the hessian, a higher eigenvalue translates to a higher rate of change of slope, and the sign of the eigenvalue dictate the maxima/minima/plane

- +ve eigenvalue $\to$ MINIMUM
- -ve eigenvalue $\to$ MAXIMUM
- close to 0 eigenvalue $\to$ PLANE

$$
κ = \frac{\nabla L^T(x) H(x) \nabla L(x)}{\nabla L^T(x) \nabla L(x)}
$$

But now, we dont have any data of the direction in which we need to flow, so let
$$
\Delta = ϵ∇L(x)
$$
where $ϵ \to 0$

This is equivalent to nesterov momentum "foresight" (check Part 3: CNN Puremath)

$$
H(x)\nabla L(x) = \frac{∇L(x + ϵ \hat ∇L(x)) - ∇L(x)} ϵ
$$

$\hat ∇L(x)$ is nothing but the direction vectors of the gradient, $\hat ∇L(x) = \frac{∇L(x)}{ ||∇L(x)||}$

hence overall, the expression becomes:

$$
κ = \frac{\nabla L^T(x) \{\frac{∇L(x + ϵ\hat ∇L(x)) - ∇L(x)} ϵ\}}{\nabla L^T(x) \nabla L(x)}
$$

let $\nabla L(x) = g$

$$
\begin{aligned}
κ &= \frac{g^T \{\frac{∇L(x + ϵ\hat g) - g} ϵ\}}{g^T g} \\
κ &= \frac{g^T∇L(x + ϵ\hat g) - ||g||^2}{||g||^2}
\end{aligned}
$$

Note that we capture both curvature and gradient data ($||g||$) and both these parameters have a say on the final LR

Now, suppose we are working with batched data meaning our g values are not the actual, true grads of the entire dataset but only a batch/sample of the data, for which the loss landscape might be different than the actual dataset why? because in a particular batch the number of items in each label of the data might be at different concentrations which affect the updates of the learnable params,

for example we have 1000 images of a husky in our data and 1000 images of a golden retreiver but in one batch we have 20 images of husky and 70 images of the other class, and this uneven distribution of labels directly reflect in the parameters the model wants to update, which might be different that the true gradients of the actual complete dataset.

$g_{true}$ = gradients of the entire datset,

Hence to overcome this problem we implement an EWMA (check out Adam section in Part 2: Logistic Regression)

$$
\bar κ_{t} = \beta \bar κ_{t-1} + (1-\beta)κ_t
$$

Now, we need to create a function of the LR using this quantity $\bar \kappa_{t}$ but how?

what we want to do is:

| κ | ||g|| | Optmal η value change |
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
risk(\bar \kappa, ||g||) = \frac{1}{\text{ReLU}(\bar \kappa) + \frac 1 {||g||}} + \text{RevReLU}(||g|| \cdot \bar \kappa)
$$

Where RevReLU is defined as:

$$
\text{RevReLU}(x) = \begin{cases}
      -1 \cdot x & \text{for } x < 0 \\
      0 & \text{for } x \geq 0
      \end{cases}
$$

This custom risk function still has an error which is the second term : $\text{RevReLU}(||g|| \cdot \bar \kappa)$, here $||g|| \cdot \bar \kappa$ can grow arbitrarily large when $\bar \kappa$ is huge and -ve, to tackle this issue we can introduce a squashing function, $ln(1 + \text{RevReLU}(||g|| \cdot \bar \kappa))$ for example or maybe even $\sigma(\text{RevReLU}(||g|| \cdot \bar \kappa))$

giving us a final (v2) function of:

$$
risk(\bar \kappa, ||g||) = \frac{1}{\text{ReLU}(\bar \kappa) + \frac 1 {||g||}} + \sigma(\text{RevReLU}(||g|| \cdot \bar \kappa))
$$

For some reason, an EMA of $\kappa ^2$ works better than an EMA of $\kappa$, for which I am unable to find the reason, but it beats other optimizers on N-Dimensional defined terrains, but that completely destroys the purpose of the curvature based LR calculation

Further, the function I created works too well with terrains with a character of positive curvature and its efficacy drops while dealing with negative curvature and sinusodial terrains

# Muon

## SVD: Single Value Decomposition: (in short)

suppose we have a matrix that exists in $\mathbb{R}^2$ and another one in $\mathbb{R}^3$, go from one dimension to the other, we use matmul with a rectangular matrix, so a rect matrix posses the power to erase/add dimensions

symmetric matrices have eigenvectors that are perpendicular to each other and the matrix of the normalized eigenvectors rotate the vectors space so that the eigenvecs are the new basis vecs and if we take the transpose of the normalized eigenvector matrix, we rotate the eigenvecs to the original basis vecs

To force symmetry out of other types of matrix (can be rect) we do this:

for example A is a rect matrix:
$A^TA$ and $AA^T$ are both symmetric

let A be mxn then $A^TA$ is nxn and $AA^T$ is mxm

now obviously $A^TA$ has n eigenvecs and $AA^T$ has m, a matrix of the eigenvectors of $A^TA$ is known as the Right-Singular Vector and $AA^T$ is known as the Left-Singular Vector, also $A^TA$ and $AA^T$ are PSD (Positive Semi-Definite) matrices meaning the eigenvalues of both matrices are positive and numerically equal, in case n does not equal m, then the lowest value from n and m are the total number of eigenvalues and the other extra (m-n) eigenvalues are 0

ALso the square root of these eigenvalues are the "singular-values" of matrix A

SVD:

SVD is just a way to rewrite a matrix into 3 different matrices:

$$
A = U \Sigma V^T
$$

where U is the eigenvectors of the Left-Singular Vector written in descending order left to right based on its singular value (square root of eigenvalue)

$\Sigma$ is the rectangular vector composed of only the singular values of the matrix along its diagonal (or half diagonal since not square)

and $V^T$ is the transpose of the Right-Singular Vector written in descending order left to right based on its singular value

Ina geometric sense, first we apply the $V^T$ transform where it rotates the eigenvectors of matrix under transform to the basis vectors of the vector space, in such a way that the eigenvector with largest singular value first gets transformed to the x-axis and so on, then we apply the $\Sigma$ transform which scales the vector with a factor of the singular values of the matrix and reduces/adds dimensions being transformed and then we apply U transform to rotate the basis vectors along the eigenvectors of the matrix.

## Note: fractional powers of a matrix:

using spectral theorem:
$$
A = Q Λ Q^T
$$

where Q is the orthonormal matrix of eigenvectors as column,
$Λ$ diagonal matrix of eigenvalues

$$
A^n = Q Λ^n Q^T
$$

since $Λ$ is symmetric (diagonal) matrix so:
$$
\begin{aligned}
Λ &= \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix} \\
Λ^n &= \begin{pmatrix} \lambda_1^n & 0 \\ 0 & \lambda_2^n \end{pmatrix}
\end{aligned}
$$

but why do $Q$ and $Q^T$ not get transformed?

because of a pattern:
$$
AA = Q Λ Q^T Q Λ Q^T
$$
now $Q^T Q = I$ since Q is orthonormal
$$
A^2 = Q Λ^2 Q^T
$$
in general
$A^n = Q Λ^n Q^T$

## relation between svd and muon

in muon, we
