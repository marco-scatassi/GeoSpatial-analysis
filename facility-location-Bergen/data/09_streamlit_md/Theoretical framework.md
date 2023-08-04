# Theoretical framework

## Introduction
In this section, we provide a general description of the emergency facility problem and discuss the reasons why such facilities are crucial. 

An emergency is “something dangerous or serious, such as an accident, that happens suddenly or unexpectedly and needs fast action in order to avoid harmful results” (https://dictionary.cambridge.org/dictionary/english/emergency). The practice of dealing with emergency situations goes under the name of *emergency management*, and it includes a 4-phase cycle [1] :
1. mitigation;
2. preparedness;
3. response;
4. recovery.

Evidently, a zero step can be included: *prevention*.

![image](https://github.com/marco-scatassi/GeoSpatial-analysis/assets/96434607/85c72bf8-3036-4793-a3d1-32acc18909e7)

In this general framework, the issue of selecting the number and the locations of emergency facilities lies both in the *mitigation* and *response* phases especially, when the uncertainty is considered. Indeed, In the context of mitigation, it involves determining the placement of facilities to minimize the detrimental effects of unavoidable emergencies (where should we place the facilities?). On the other hand, the *response* phase regards every course of actions undertaken after the occurrence of an emergency (which demanding point is associate to which serving point?). 

## Emergency facility location problem
### A taxonomy
A first classification can be performed, distinguishing *continuous* and *discrete* models. Specifically:
- **continuous** model are those in which facilities can be located in any point within the feasibility region;
- **discrete** model are those in which facilities can be located in a finite subset of pre-selected points within the feasibility region.

Continuous model are extremely computational demanding, therefore the most used approach in literature is the discrete formulation. Indeed, it is the formulation used in this work.

Then, following [2], discrete facility location problems can be divided in the following 3 broad categories:
- **covering based**, in which a maximum distance is fixed. And it is required that the distance between every couple (destination, facility) is below that value;
- **median based**, in which the objective is to minimize the weighted average distance between points and the associated facility;
- other problems.

Further classification can be done within each of the 3 previous categories. However, the one adopted in this work is the **p-center location problem**. This formulation as well as its stochastic version will be described in the following sections.

### Deterministic Formulation 
Referring to [5], the **p-center location** problems aim to **minimize the maximum distance** among demand points and their associated facility. It is assumed that all demand points are covered. 

This model, that is well known in literature, is particularly suitable to approach **emergency response** facility location problems. As already mentioned, an **emergency** is an event that occurs suddenly, unexpectedly and that requires an immediate intervention. This means that wherever an emergency event is originated, the **response time** should be as low as possible. The presence of some demand hardly reachable points represents a potential critical situation and should be avoided. Therefore, the **p-center** model is a valid solution able to take into account those aspects.

A possible formulation for the **p-center problem**, as described in [5], is provided below.

##### Formulation
$$
\begin{aligned}
& \min L \\
& \text { subject to } \\
& \sum_{j \in N_i} y_{i j}=1, i \in I \\
& \sum_{j \in J} x_j=p \\
& \sum_{j \in N_i} d_{i j} y_{i j} \leq L, i \in I \\
& y_{i j} \leq x_j, i \in I, j \in N_i \\
& y_{i j} \in\{0,1\}, i \in I, j \in N_i \\
& x_j \in\{0,1\}, j \in J \\
& L \geq 0 .
\end{aligned}
$$

#### Sets:
- $I$ : The set of demand points.
- $J$ : The set of candidate locations.
- $N_i$ : The set of all candidate locations which can cover demand point $i \in I, N_i=\{j \in J: d_{i j} \leq D_i\}$.

#### Input parameters:
- $d_{i j}$ : The **travel time** from demand point $i \in I$ to candidate location $j \in J$.
- $D_i$ : The maximum acceptable **travel time** from demand point $i \in I$ (the cover distance or time).
- $p$ : The number of candidate locations to be established.

#### Decision variables:
- $x_j$ : 1, if a facility is established at candidate location $j \in J$; 0 otherwise.
- $y_{i j}$: 1, if demand point $i$ is assigned to a facility at candidate location $j \in N_i ; 0$ otherwise.

### Stochastic Formulation 
The deterministic formulation is not able to take into account the stochastic nature of **travel time**, that depends on several factors (hour of the day, weather conditions, changes in the road viability, …). Many methodologies and approaches exist in order to embrace the stochastic nature of the model previously described, such as chance constraint optimization, stochastic programming, robust optimization, queuing theory.

The two-stage optimization model has been used in this work. This decision is justified observing that the location of the facilities can be viewed as a first stage decision, that need to be robust with respect to changing scenarios. While, once the uncertainty has been disclosed, it is possible to decide which facility should serve a specific demand point [6].

Specifically, the deterministic model can be reformulated as follows:

$$
\begin{aligned}
& \min L \\
& \text { subject to } \\
& \sum_{j \in N_i} y_{i j}^k=1, i \in I, k\in{K}\\
& \sum_{j \in J} x_j=p \\
& \sum_{j \in N_i} d_{i j}^k y_{i j}^k \leq L, i \in I, k\in K \\
& y_{i j}^k \leq x_j, i \in I, j \in N_i, k\in K \\
& y_{i j}^k \in\{0,1\}, i \in I, j \in N_i, k\in K \\
& x_j \in\{0,1\}, j \in J \\
& L \geq 0 .
\end{aligned}
$$

#### Sets:
- $I$ : The set of demand points.
- $J$ : The set of candidate locations.
- $K$: The set of possible scenarios.
- $N_i$ : The set of all candidate locations which can cover demand point $i \in I, N_i=\{j \in J, k\in K: d_{i j}^k \leq D_i\}$.

#### Input parameters:
- $d_{i j}^k$ : The **travel time** from demand point $i \in I$ to candidate location $j \in J$ in the $k$-th scenario.
- $D_i$ : The maximum acceptable **travel time** from demand point $i \in I$ (the cover distance or time).
- $p$ : The number of candidate locations to be established.

#### Decision variables:
- $x_j$ : 1, if a facility is established at candidate location $j \in J$; 0 otherwise.
- $y_{i j}^k$: 1, if demand point $i$ is assigned to a facility at candidate location $j \in N_i$ in the $k$-th scenario; $0$ otherwise.

### Observation 
The formulation adopted in this work assumes $D_i = +\infty$. This allows every solution to be feasible, and it's motivated by the absence of a specific requirement.

#### Bibliography
[1] R. Z. Farahani, M. M. Lotfi, A. Baghaian, R. Ruiz, and S. Rezapour, ‘Mass casualty management in disaster scene: A systematic review of OR&MS research in humanitarian operations’, _European Journal of Operational Research_, vol. 287, no. 3, pp. 787–819, Dec. 2020, doi: [10.1016/j.ejor.2020.03.005](https://doi.org/10.1016/j.ejor.2020.03.005).

[2] M. S. Daskin, ‘What you should know about location modeling’, _Naval Research Logistics (NRL)_, vol. 55, no. 4, pp. 283–294, 2008, doi: [10.1002/nav.20284](https://doi.org/10.1002/nav.20284).

[3] Y. Liu, Y. Yuan, J. Shen, and W. Gao, ‘Emergency response facility location in transportation networks: A literature review’, _Journal of Traffic and Transportation Engineering (English Edition)_, vol. 8, no. 2, pp. 153–169, Apr. 2021, doi: [10.1016/j.jtte.2021.03.001](https://doi.org/10.1016/j.jtte.2021.03.001).

[4] W. Wang, S. Wu, S. Wang, L. Zhen, and X. Qu, ‘Emergency facility location problems in logistics: Status and perspectives’, _Transportation Research Part E: Logistics and Transportation Review_, vol. 154, p. 102465, Oct. 2021, doi: [10.1016/j.tre.2021.102465](https://doi.org/10.1016/j.tre.2021.102465).

[5] A. Ahmadi-Javid, P. Seyedi, and S. S. Syam, ‘A survey of healthcare facility location’, _Computers & Operations Research_, vol. 79, pp. 223–263, Mar. 2017, doi: [10.1016/j.cor.2016.05.018](https://doi.org/10.1016/j.cor.2016.05.018).

[6] P. Beraldi and M. E. Bruni, ‘A probabilistic model applied to emergency service vehicle location’, _European Journal of Operational Research_, vol. 196, no. 1, pp. 323–331, Jul. 2009, doi: [10.1016/j.ejor.2008.02.027](https://doi.org/10.1016/j.ejor.2008.02.027).