# Curvature-Aware Optimizer: (ARCHIVED)
![Status: Abandoned](https://img.shields.io/badge/Status-Archived-red)
![CIFAR-10 Accuracy](https://img.shields.io/badge/Benchmark-58%25-orange)
![Baseline](https://img.shields.io/badge/Baseline_Adam%2BCosine-68%25-blue)

**This repository contains my own study and tests of an optimizer idea I had that ended up not working that well**
This repo is frozen as for `01-07-2026` and will get updated when I have the time to resume my research on the optimizer.

## Abstract
This is an experimental, second degree, curvature aware optimizer that calculates its own learning rate based on the eigenvalues of the hessian matrix, which is approximated using taylor linearization

The optimizer was meant to follow this table: (later not followed through)

$$
\begin{array}{|l|c|r|}
\hline
\text{κ} & \text{||g||} & \text{Optmal η value change} \\ \hline
\text{+ve High}       & \text{High}               &\text{decrease}                 \\ \hline
\text{-ve High}       & \text{Low}               & \text{increase}               \\ \hline
\text{+ve Low}       & \text{High}               & \text{increase}               \\ \hline
\text{-ve Low}       & \text{Low}               & \text{increase}                \\ \hline
\text{-ve High}       & \text{High}               &\text{decrease}                 \\ \hline
\text{+ve High}       & \text{Low}               & \text{decrease}                \\ \hline
\text{-ve Low}       & \text{High}               & \text{increase}                \\ \hline
\text{+ve Low}       & \text{Low}               & \text{increase}                \\ \hline
\end{array}
$$

**Versions:** I have documented 6 versions of the optimizer, each having different architectures, and I have tried about 5 more architectures which I deemed to be failures and hence did not document them. out of all the versions, V3 performs the best on terrains with positive curvature but it lacks on negative curvature terrains and periodic terrains. MNISTVersion is the version of the optimizer I used on the digit MNIST dataset in the file `CurvatureOptimizerTestRunsTerrains`.

**Performance**: V3 beat the optimizers I put it up against on the N-dimensional terrains coupled with learning rate callbacks but it fails to converge on spherical terrains and Rastrigin. (check Graphs). 
MNISTVersion scored a final of **97.97% validation accuracy** in 3 epochs while Adam + CosineAnnealing scored a final of **98.87% validation accuracy** in the same number of epochs, on the digit MNIST dataset. 
V3 scored a final of **57.85% validation accuracy** in 10 epochs and Adam + CosineAnnealing scored a final of **68.19%  validation accuracy** in the same number of epochs on CIFAR-10.

I suspect, the reason for failure is the intense condensation of information that I did with the optimzer logic, condensing the entire curvature + gradient information into a single scalar - Learning rate is not the way to go about this problem, hence in the future, I will work on a "per-parameter" scaling and explore further possibilities




