# Level Crossings in the Spectrum of the Laplacian on Calabi-Yau (CY) Manifolds

This repository contains the code to generate all the plots in [2304.00027](https://arxiv.org/pdf/2304.00027), which investigates level crossings in the spectrum of the laplacian on Calabi-Yau (CY) Manifolds.

CY manifolds are important in both physics and math contexts. In particular, they are a solution to the string theory equations of motion, since they have been shown to always admit a Ricci-flat
metric. However, upt until recently, no such explicit metric had been written down, except for low-dimensional CYs. This changed with advancements in machine learning techniques and their applications to math and physics, and so numerical Ricci-flat metrics for these manifolds were found using neural networks. 
With the metric in hand, the spectrum of the Laplacian on these manifolds was computed and plotted as a function
of the complex structure. It was noticed that the spectrum crossed at certain values of the complex structure, which indicated an enhanced degeneracy and hence an enhancement in the symmetries of the CY.
However, all the symmetry had been taken into account, and so the goal of our work was to find the source of this enhanced degeneracy in the spectrum.

In the paper, we give evidence that these crossings are related to special number-theoretic properties that arise at particular values of the complex structure. We use the [cymetric](https://github.com/pythoncymetric/cymetric) package to learn the Ricci-flat metric for Fermat type CY manifolds in dimensions 2,4, and 6.
