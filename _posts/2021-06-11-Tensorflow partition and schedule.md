---
layout: post
title: The TensorFlow Partitioning and Scheduling Problem --- It's the Critical Path! 阅读笔记
comments: True
author: liangpeng

---

作者：梁鹏

# The TensorFlow Partitioning and Scheduling Problem: It's the Critical Path! 阅读笔记

2017年11月 arXiv:1711.01912v1 德国Stuttgart大学的文章
## ABSTRACT

SOTA数据流系统比如tensorflow需要将大的计算图划分到异构的CPU、GPU和TPU上。然而，划分不能单独看待，每一个设备都需要选择下一个被执行的图顶点，也就是要完成本地的调度决策。划分(partition)和调度(scheduling)都是本身NP完全问题，但是需要结合在一起解决，以优化迭代的执行时间。

在这篇论文中，作者提出了在TensorFlow中不同的启发式策略来解决划分和调度的问题。模拟了提出方法在异构环境中执行communication-intense任务的性能。发现最优解是专注于最小化计算图中critical path（关键路径）的执行时间的启发式划分和调度方法。这一方法和与关键路径无关的策略(如基于hash的划分和FIFO调度)相比，这些策略提供了高达4倍的加速。


## 1. Introduction
相关工作还很少

贡献：
1. formulate了tensorflow的划分和调度问题，并证明NP-hard
2. 基于直觉，开发了几种partition和schedule的启发式方法
3. 评估了启发式方法在不同的计算图中的性能。
发现，最好的划分调度启发式方法是基于最小化critical path的执行时间，效果比hash-based partition and FIFO schedule 快4倍

##  2. Problem Formulation
令$$G=(V,E)$$为有向无环数据流图，$$E$$在TensorFlow中相当于tensor，其$$e_i$$的权重是通信量$$t_i$$，$$V$$中$$v_i$$的权重是计算复杂度$$c_i$$，只有在其所有的入边都available的时候才标记为schedulable。

令$$D$$为设备集合，集合中的每一个设备$$dev_i$$都有计算速度$$s_i$$。并且，$$dev_i$$还有对应的最大内存容量$$C_i$$。例如，数据从源顶点流出到边时需要消耗设备的内存。任意两个设备通过物理或者虚拟网络连接。带宽矩阵$$B \in \mathbb{R}^{k\times k}$$，设备$$dev_i$$和$$dev_j$$之间 的通信带宽为$$B_{i,j}$$ Bytes/second.

搭配约束(collocation constraints)$$\mathbb{C \in V\times V}$$ 指示了需要放在同一设备上的vertice间的对称性。并且，现实世界中的数据流设备还有隐形或者显性的计算操作的placement constraints，记为$$\mathbb{D}\in V\times D$$。

在图一中给了例子，有向无环图（DAG）由12个顶点组成，他们被划分到了三个设备上。这一划分是全局的。划分之后，每一个设备需要执行一个集合的顶点。这可能会有多种调度策略，比如$$dev_1$$可以选择调度$$v_1,v_2$$或者$$v_3$$。
{% include figure.html src="/figures/tensorflow_partition_schedule/1.png" caption="图1"%}

划分函数$$p:V \to D$$将顶点映射到设备上，调度函数$$f:V \to N$$将顶点映射到顶点被执行的time slots上。则目标变为 
$$min_f(max_{v\in V}f(v))\tag{1}$$ 
Note: 函数f的返回值是顶点执行的开始时间，但是我们需要研究的是最小化最大完成时间。我们可以通过将所有顶点连接到一个没有出边的sink 顶点上来，并且连接边权重均为0，来解决这一问题。

我们需要保证，内存约束是完全实现的。将$$dev_j$$上在时间$$l$$时的活跃边记为$$E_{active}(l,j)$$，则有：
$$\forall dev_j \in D, l \in \mathbb{N}:\sum_{e_i \in E_{active(l,j)}}t_i \lt C_j \tag{2}$$
且要满足collocation constraints, 即他们需要分在同一设备上，
$$\forall v_i,v_j \in V: (v_i,v_j)\in \mathbb{C}\to p(v_i)=p(v_j) \tag{3}$$
最后，需要满足设备约束，即如果$$v_i$$在$$dev_j$$上，需要满足
$$\forall v_i \in V, dev_j \in D: (v_i,dev_j)\in \mathbb{D}\to p(v_i)=dev_j \tag{4}$$

### 2.1 NP-completeness
**Theorem 2.1** 划分和调度问题是 NP-完全的。证明略

### 2.2 Challenges
**Scalability**: 现有的调度算法全局选择处理器和顶点的执行时间。这种方法对task scheduling可行，因为任务是粒度比较粗的计算单位。但是对与大量的细粒度图顶点来说，计算全局调度的时间太长了。因此，作者决定先用可扩展的划分启发方法完成图的划分，再在设备上局部解决调度问题。

**Heterogeneity**: TensorFlow中，有顶点计算复杂度的异构，边通信量的异构，设备内存容量的异构，设备计算速度的异构，设备间带宽的异构。

## 3 Partitioning Approaches
### 3.1 Hashing
随机assign
### 3.2 Path-Based Heuristics
#### 3.2.1 Batch Split 
避免昂贵的关键路径计算，但仍然优先考虑位于长路径上的顶点的最优位置。
#### 3.2.2 Critical Path
试图将完整的critical path分配给最快的设备，使用这一策略时，对于critical path就没有额外的通信延迟。
### 3.3 Multi-Objective Heuristics
#### 3.3.1 MITE
MITE(Memory, Importance, Traffic, Execution time)。考虑四个优化目标，用一个启发函数来将顶点映射到设备上。
$$ mite(v_i,dev_l) = mem(dev_l)\times imp(v_i, dev_l)\\\times traffic(v_i,dev_l)\times execTime(v_i,dev_l) \tag{8}$$
具体计算看文章吧
#### 3.3.2 Depth First Search
$$dfsScore(v-i,dev_l) = traffic(v_i,dev_l)\\ \times execTime(v_i,dev_l) \tag{11}$$

## 4 Scheduling
### 4.1 PCT Scheduling
暂略
### 4.2 MSR Scheduling
暂略

## 5 EVALUATIONS
使用了基于事件的仿真。
### 5.1 Experimental setup
**Simulation Parameters.** 在评估中仿真了50台设备。参考AWS服务器设置了不同的参数。

**TensorFlow Networks.** 模拟了CNN，动态RNN，RNN。

**Baselines** Heterogeneous Earliest Finish Time(HEFT)算法解决了相关的*任务调度问题*，但不能直接用到TF问题中。作者稍微修改了一下以作为baseline。

第二个scheduling的baseline是FIFO。

### 5.2 Experiments and Results
看图吧。
{% include figure.html src="/figures/tensorflow_partition_schedule/3.png" caption="图3"%}
Hash划分和FIFO调度结果差的原因是不适应TF问题的特点。Hash划分虽然负责均衡很好，但是这不是减少数据流计算时间的重要方法。FIFO调度不支持关键路径的快速执行。

**很明显，分区和调度策略的重点都是减少关键路径的计算时间，这是最
重要的。**


## 6 Related Work
将图划分问题作为预处理中的一步已经吸引了很大的注意力。然而，图处理中的图划分策略专注于 最小化设备之间的traffic，并保持负载均衡。而TF问题处理的是从源顶点到sink顶点的数据流。因此，图划分问题中的划分策略不能泛化到TF问题上来用。

在这篇论文里分析的HEFT调度算法在数据流执行上相较于其他20种启发式策略表现出了更好的性能。然而，这种调度算法将顶点在运行过程中动态地放在设备上。我们没有考虑动态过程。我们不认为顶点在运行过程中的动态调整是一个有效的TF问题的解决方法。取而代之的是，更直观的方法是先将图划分好，然后在各个设备上进行自己那一部分的调度。

## 7 Conclusion
证明了NP-完全。在模拟实验中将目标放在最小化关键路径执行时间的方法比其他方法块4倍。计划文莱在tensorflow框架中验证模拟结果。
