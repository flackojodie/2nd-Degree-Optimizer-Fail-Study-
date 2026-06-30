# Idea:

Building an optimizer that takes into account the hessian information instead of making it on time (like cosine annealing etc) and changes the LR based upon the curvature, and we estimate the hessian using the following:

$$
\nabla L(x + \Delta) \approx \nabla L(x) + \epsilon H(\Delta)
$$

$\Delta \to 0$

the above is the taylor series linearization,

$$
H(x) = \frac{\nabla L(x + \Delta) - \nabla L(x)}{\Delta}
$$

But remember that hessian is actually a matrix, we cant find any curvature information of an area using a matrix hence to actually "understand" the curvature, we use the rayleigh quotient to actually quantify the curavture as it is nothing but the eigenvalue and is not affected by the twisting of the vector space due to the linear transformation occuring due to the hessian, a higher eigenvalue translates to a higher rate of change of slope, and the sign of the eigenvalue dictate the maxima/minima/plane

- +ve eigenvalue $\to$ MINIMUM
- -ve eigenvalue $\to$ MAXIMUM
- close to 0 eigenvalue $\to$ PLANE

$$
\kappa = \frac{\nabla L^T(x) H(x) \nabla L(x)}{\nabla L^T(x) \nabla L(x)}
$$

But now, we dont have any data of the direction in which we need to flow, so let
$$
\Delta = \epsilon\nabla L(x)
$$
where $\epsilon \to 0$

This is equivalent to nesterov momentum "foresight" (check Part 3: CNN Puremath)

$$
H(x)\nabla L(x) = \frac{\nabla L(x + \epsilon \hat \nabla L(x)) - \nabla L(x)}{\epsilon}
$$

$\hat \nabla L(x)$ is nothing but the direction vectors of the gradient, $\hat \nabla L(x) = \frac{\nabla L(x)}{||\nabla L(x)||}$

hence overall, the expression becomes:

$$
\kappa = \frac{\nabla L^T(x) \{\frac{\nabla L(x + \epsilon\hat \nabla L(x)) - \nabla L(x)}{\epsilon}\}}{\nabla L^T(x) \nabla L(x)}
$$

let $\nabla L(x) = g$

$$
\begin{aligned}
\kappa &= \frac{g^T \{\frac{\nabla L(x + \epsilon\hat g) - g}{\epsilon}\}}{g^T g} \\
\kappa &= \frac{g^T\nabla L(x + \epsilon\hat g) - ||g||^2}{||g||^2}
\end{aligned}
$$

Note that we capture both curvature and gradient data ($||g||$) and both these parameters have a say on the final LR

Now, suppose we are working with batched data meaning our g values are not the actual, true grads of the entire dataset but only a batch/sample of the data, for which the loss landscape might be different than the actual dataset why? because in a particular batch the number of items in each label of the data might be at different concentrations which affect the updates of the learnable params,

for example we have 1000 images of a husky in our data and 1000 images of a golden retreiver but in one batch we have 20 images of husky and 70 images of the other class, and this uneven distribution of labels directly reflect in the parameters the model wants to update, which might be different that the true gradients of the actual complete dataset.

$g_{true}$ = gradients of the entire datset,

Hence to overcome this problem we implement an EWMA (check out Adam section in Part 2: Logistic Regression)

$$
\bar \kappa_{t} = \beta \bar \kappa_{t-1} + (1-\beta)\kappa_t
$$

Now, we need to create a function of the LR using this quantity $\bar \kappa_{t}$ but how?

what we want to do is:

| $\kappa$ | $||g||$ | Optmal $\eta$ value change |
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
risk(\bar \kappa, ||g||) = \frac{1}{\text{ReLU}(\bar \kappa) + \frac{1}{||g||}} + \text{RevReLU}(||g|| \cdot \bar \kappa)
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
risk(\bar \kappa, ||g||) = \frac{1}{\text{ReLU}(\bar \kappa) + \frac{1}{||g||}} + \sigma(\text{RevReLU}(||g|| \cdot \bar \kappa))
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
A = Q \Lambda Q^T
$$

where Q is the orthonormal matrix of eigenvectors as column,
$\Lambda$ diagonal matrix of eigenvalues

$$
A^n = Q \Lambda^n Q^T
$$

since $\Lambda$ is symmetric (diagonal) matrix so:
$$
\begin{aligned}
\Lambda &= \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix} \\
\Lambda^n &= \begin{pmatrix} \lambda_1^n & 0 \\ 0 & \lambda_2^n \end{pmatrix}
\end{aligned}
$$

but why do $Q$ and $Q^T$ not get transformed?

because of a pattern:
$$
AA = Q \Lambda Q^T Q \Lambda Q^T
$$
now $Q^T Q = I$ since Q is orthonormal
$$
A^2 = Q \Lambda^2 Q^T
$$
in general
$A^n = Q \Lambda^n Q^T$

## relation between svd and muon

in muon, we decompose the gradient matrix ($G$) into $U$, $\Sigma$ and $V^T$ but remember that $\Sigma$ is a matrix of the singular values which scales the matrix $G$, but this can be a problem, suppose our $\Sigma$ is: (if G is symmetric)
$$
\begin{pmatrix} 1000 & 0 \\ 0 & 1 \end{pmatrix}
$$

now this is a problem, we are kind of preferring the direction with singular value 1000 more than the other value leading to "ill-conditioning", we explore this direction with 1000 SV more, to tackle this, muon deletes the $\Sigma$ matrix entirely so the decompostion after deletion becomes something like this:

$$
\tilde G = UV^T
$$

But the problem is, this logic is kinda too obvious SVD was popularized in the field of computer science in the year 1970, and muon arrived as late as 2024

probably because SVD is expensive,

## Removing $\Sigma$:
$$
\begin{aligned}
G &= U \Sigma V^T \\
G^T G &= (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^TU^TU \Sigma V^T \\
\text{Since U is orthonormal} &\to U^TU = I \\
G^T G &= V \Sigma^T \Sigma V^T \\
\text{Since Σ is diagonal} &\to \Sigma^T = \Sigma \\
G^T G &= V \Sigma^2 V^T, \ \text{taking power -1/2 on bts} \\
(G^T G)^{-1/2} &= V \Sigma^{-1} V^T \ \text{by spectral theorem} \\
\text{pre multiplying by G} & \\
G (G^T G)^{-1/2} &= U \Sigma V^T V \Sigma^{-1} V^T \\
\text{Since V is orthonormal} &\to V^TV = I \\
G (G^T G)^{-1/2} &= U \Sigma \Sigma^{-1} V^T \\
G (G^T G)^{-1/2} &= U V^T
\end{aligned}
$$

Hence we have simplified $UV^T$ in terms of $G$ (Gradient), now we just need a way to estimate $(G^T G)^{-1/2}$ without again using SVD, we have a technique to do so called the "Newton-Schulz method" but id argue we are using the newton-schulz in its true form, this method is just a derivative of the original method:

suppose we have a matrix:
$$
A = G^T G
$$

but we need $A^{-1/2}$ meaning:
$$
AX^2 = I
$$
where $X = A^{-1/2}$, so if we create a general equation to estimate X:
$$
E = AX^2 - I
$$
as X $\to A^{-1/2},\ E \to 0$

creating and isolating $A^{-1/2}$:
$$
\begin{aligned}
E+I &= AX^2 \\
(E+I)^{-1/2} &= (AX^2)^{-1/2} \ \ \text{taking power -1/2 on bts}, \\
(E+I)^{-1/2} &= A^{-1/2} X^{-1} \\
(E+I)^{-1/2} &= A^{-1/2} X^{-1} \ \ \text{post multiply by X}, \\
(E+I)^{-1/2}X &= A^{-1/2}
\end{aligned}
$$

now we want to estimate $A^{-1/2}$, as in iteratively get close to the value, hence define $A^{-1/2}$ to be the result of an iteration of the operation $(E+I)^{-1/2}X$ so $A^{-1/2} = X_{new}$

also another thing to notice is that with each iteration $E$ gets smaller and we can also use binomial expansion to estimate $(E+I)^{-1/2}$

$$
\begin{aligned}
(E+I)^{-1/2} &\approx I - \frac{1}{2}E, \ \ \text{linearization}, \\
(I - \frac{1}{2}E)X &= A^{-1/2}, \\
(I - \frac{1}{2}(AX^2 - I))X &= A^{-1/2}, \ \ \text{using defination of E}, \\
\frac{1}{2}X_{old}(3I - AX_{old}^2) &= X_{new}
\end{aligned}
$$

hence after $N \to \infty$ iterations $X_{new} = A^{-1/2}$
hence this equation is a really smart and effective way to estimate $(G^T G)^{-1/2}$

Thats it, thats muon, now we just update the variables:

$$
P_n = P_{n-1} - \eta UV^T
$$

# BFGS & L-BFGS (Broyden-Fletcher-Goldfarb-Shanno and limited-BFGS)

## Newton's method:
$$
P_{n+1} = P_{n} - H^{-1} \nabla L(x)
$$

with the variables having their usual meanings

### Derivation:
we want to explore our landscape using small steps: $x + h$

Taylor expansion of $f(x)$ about a small h, where f(x) is our landscape function

$$
f(x + h) \approx  f(x) + \nabla f(x)^Th + \frac{1}{2}h^THh
$$

the above function is an estimation of our landscape f(x) and if we differentiate the above equation wrt h, we find the stationary points (maxima/minima/plateau) of the landscape at that point, keep in mind our main motivation is to find an expression for $h$

$$
0 = \nabla_h \left[f(x) + \nabla f(x)^Th + \frac{1}{2}h^THh \right]
$$

I am not expanding the sums and differentiating here like how I did in the puremath series so using the chain rule:

$$
\begin{aligned}
0 &= \nabla f(x) + \frac{1}{2} (h^T (\nabla_h Hh)^T  + Hh \nabla_h h^T) \ \ \text{transposed to match dimensions} \\
0 &= \nabla f(x) + \frac{1}{2} (2 Hh) \\
0 &= \nabla f(x) + Hh \\
\text{let } \nabla f(x) &= G \\
-G &= Hh \\
\text{pre multiplying by } &H^{-1} \\
-H^{-1}G &= h
\end{aligned}
$$

Hence going by our earlier statement
