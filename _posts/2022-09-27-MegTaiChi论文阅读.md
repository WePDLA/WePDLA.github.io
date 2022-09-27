---
layout: post
mathjax: true
catalog: true
comments: true
top-tags-list: true
header-img: "img/post-bg-universe.jpg"
header-mask: 0.4
title: MegTaiChi论文阅读
author: 杨智琳
tags: [内存管理]
---

由于现有的tensor partition和rematerialization的方法大多数忽视了**动态计算图的不变特征**，还有**不同内存位置上大小相同张量之间的变化**，所以仍有优化空间。由于在神经网络训练过程中对张量的访问模式是有规律的，所以本文**基于运行时追踪到的动态张量访问模式**进行内存管理决策，有效协调了**tensor partition和rematerialization**，实现了启发式、自适应和细粒度的内存管理。

### （一）现有的内存优化的两种主要技术：

- **tensor partition：**对不同层的模型参数进行划分，给每个分区分配一个machine进行训练。该方法是在**多台设备**上训练一个大型模型，并**实现负载平衡，提高并行性，最小化网络通信代价**。
- **tensor rematerialization：**在前向传播中释放中间张量，在反向传播中重新生成。该方法主要是针对**单台设备**进行内存优化的，通过利用外存或者增加额外的计算来节省设备上的内存空间。主要包括两种通用的方法：**swap和重计算**。

{% include figure.html src="/figures/MegTaiChi/1.PNG" caption=""%}

虽然上述两种优化技术都可以减少内存占用，但它们是根据**不同的应用需求**开发的。

### （二）背景：

在实际应用中，通常会在**中等规模的集群上**训练大规模模型或者batch size较大的DNN模型，比如在GPU较少的私有云平台上训练BERT模型。在这些场景下，可以**将tensor partition和rematerialization两种方法结合起来**，从而在上述应用场景下进行内存管理和优化。

但是在tensor rematerialization的过程中，需要动态地释放和频繁地分配内存，这将导致严重的**内存碎片化**。如图1(b)所示。所以需要对内存碎片进行管理，但碎片管理需要主存的帮助，会严重地影响性能。

### （三）面临的问题与挑战：

1. **如何在运行时进行张量划分的动态调整？**
2. **如何确定在当前张量划分的计划下，驱逐和重新生成哪些张量？**
3. **如何为每个张量分配内存空间来提高内存的利用率？**

### （四）本工作基于的两个observations：

1. 深度学习训练过程中张量访问的**数据重用性和固定的访问模式**。
2. **动态计算图中的一些不变特征**。

### （五）本文的贡献：

- 提出了一种动态张量内存管理优化模块MegTaiChi用于DNNs的训练，该模块实现了tensor partition和rematerialization的有效结合。

- 设计了一种**动态张量划分策略**，在模型训练过程中实现了**动态调整**。

- 设计了一种**动态张量管理策略**，有效地**结合了swap和重计算**机制。

- 提出了一种**最优的内存分配方案**，利用DCGs的不变特性，缓解了**内存碎片问题**。

# 一、本文的方法

如图2所示，MegTaiChi是一个虚拟机（VM）模块，通过基本的**原语操作**控制张量的访问，可以在**张量粒度上**管理动态内存访问。首先，利用**空间局部性**提出了**dynamic tensor partition (DTP)**；其次，利用**时间局部性**设计了**dynamic tensor evicting (DTE)**；最后，**结合空间局部性与时间局部性**，提出了接近最优的**tensor memory allocation (TMA)**方法。

{% include figure.html src="/figures/MegTaiChi/2.PNG" caption=""%}

## （一）**Dynamic Tensor Partition (DTP)**

DTP是一种针对张量划分的**启发式策略**。MegTaiChi会根据operator的输入和输出信息，为参与OP的所有张量**生成一个划分的策略**。如下图，在要执行OP𝑖+1时，DTP会根据启发式策略来为OP𝑖+1生成一个张量划分策略，可能会改变OP𝑖的输出划分维度和在执行OP𝑖+1之前权重的划分维度。

{% include figure.html src="/figures/MegTaiChi/3.png" caption=""%}

## （二）**Dynamic Tensor Evicting (DTE)**

DTE是一种在运行时释放张量的**自适应策略**。MegTaiChi会**追踪OPs的执行顺序，并记录执行信息**。当发生OOM时，DTE会根据这些信息**决定应该释放哪些张量，以及是执行swap还是重计算**。如下图，在执行OP4之前发生了OOM，则Autoe_vict（）指令会被插入到指令队列中，并基于DTE自适应策略释放一些张量。如果发生内存访问失败的情况，缺失张量会重新生成。

{% include figure.html src="/figures/MegTaiChi/4.PNG" caption=""%}

## （三）**Tensor Memory Allocation (TMA)**

TMA是一种**细粒度的方法**，用于为所有张量分配内存地址。在前五次的训练迭代过程中，MegTaiChi通过追踪指令序列来获得**DCG重要的不变特征**。TMA基于这些特征建立**一个细粒度的内存分配策略**，并应用于接下来的训练过程中。如下图，TMA会模拟内存分配过程，并调整某些张量的起始地址，得到一个关于所有张量在时间和空间维度上依赖关系的拓扑映射，并利用排序算法生成**一个接近最优的内存分配策略**。TMA成功地降低了内存峰值，并**避免了内存碎片的产生**。

{% include figure.html src="/figures/MegTaiChi/5.PNG" caption=""%}

# （二）**IMPLEMENTATION OF MEGTAICHI**

## （一）**Dynamic Tensor Partition**

#### 四种常用的张量划分模式：

{% include figure.html src="/figures/MegTaiChi/6.PNG" caption=""%}

如果两个相邻的OPs之间使用不同的张量划分模式，需要额外的通信进行**模式转换**。所需的通信如表1所示：

{% include figure.html src="/figures/MegTaiChi/7.PNG" caption=""%}

为了在训练过程中实现较高的吞吐量，需要在**执行成本和内存消耗**之间做一个平衡。设$p_i^k $为第𝑖个操作符𝑜𝑖的划分模式，表示使用模式𝑘对𝑜𝑖进行划分，bs表示batch size，$m_{inter} $表示模式转换所需的额外内存空间。则总执行时间$𝑡(𝑜_𝑖，𝑝^𝑘_𝑖，𝑏𝑠) $和内存成本$m(𝑜_𝑖，𝑝^𝑘_𝑖，𝑏𝑠) $可以表示为：

{% include figure.html src="/figures/MegTaiChi/8.PNG" caption=""%}

{% include figure.html src="/figures/MegTaiChi/9.PNG" caption=""%}

则可以通过求解下面的优化问题来选择划分模式$k^*_i $：

{% include figure.html src="/figures/MegTaiChi/10.PNG" caption=""%}

## （二）**Dynamic Tensor Evicting**

DTE主要解决了两个问题： (1)如何管理多个设备上的内存；(2)如何确定应该swap或重计算哪些张量，如何避免产生内存碎片。

#### **Memory Management Mechanism：**

在多个设备上进行训练时，不同层张量到达GPU的顺序通常是**无序的**，使得内存开销的预测变得很困难。本文提出了一种**内存管理机制**：

1. 首先将每个设备上的内存划分为**两个部分𝑚=𝑚1+𝑚2**。第一部分用于**设备之间的通信以梯度的存储**，另一个部分用于当前设备上的**计算过程**。

1. 接下来训练时接收到的梯度被存储在第一部分上，该区域存储了**所有参数的梯度和跨𝑝个设备的层输入和输出总张量的1/𝑝**。

1. 最后为第二部分定义了一个**内存阈值𝑚∗**。如果第二部分的内存消耗大于𝑚∗，将**选择并驱逐一些张量**，保留了𝑚2−𝑚∗大小的空间用于执行当前的operator。

#### **Tensor Evicting Mechanism：**

每次迭代时，追踪每个张量𝑡的信息，比如**Staleness 𝑠(𝑡)**（自上次访问以来的时间）、**Memory 𝑚(𝑡)**（张量大小）、**Cost 𝑐(𝑡)**（基于其计算路径从其父张量计算𝑡所需的时间）**Recomputing times 𝑟𝑒𝑡(𝑡)**（重计算时间）。

张量t的重计算开销：

{% include figure.html src="/figures/MegTaiChi/11.PNG" caption=""%}

张量t的swap开销（γ表示带宽）：

{% include figure.html src="/figures/MegTaiChi/12.PNG" caption=""%}

DTE优先选择释放那些**驱逐后不会产生新的内存碎片的张量**，如下图6所示。用$𝑀_{𝑙𝑒𝑓𝑡}(𝑡) $和$ 𝑀_{right}(𝑡) $分别表示为𝑡的左右两边空闲内存的大小，则**优先释放**$ 𝑚(𝑡)+𝑀_{𝑙𝑒𝑓𝑡}(𝑡)+𝑀_{right}(𝑡) $**更大的张量t**。DTE将通过最小化下面的代价函数来释放张量：

{% include figure.html src="/figures/MegTaiChi/13.PNG" caption=""%}

{% include figure.html src="/figures/MegTaiChi/14.PNG" caption=""%}

## （三）**Tensor Memory Allocation**

用一个四元组𝑟𝑖=（𝑠𝑖，𝑒𝑖，𝑚𝑖，𝑎𝑖）表示第𝑖个内存的分配请求。如果两个请求𝑟𝑖和𝑟𝑗满足**𝑠𝑖≤𝑒𝑗或𝑠𝑗≤𝑒𝑖**，则它们**在时间上会相互冲突**。如果𝑟𝑖和𝑟𝑗满足**𝑎𝑖≤𝑎𝑗+𝑚𝑗或𝑎𝑗≤𝑎𝑖+𝑚𝑖**，则它们存在**空间冲突**。TMA的设计目的是**在任何两个时间冲突的请求没有空间冲突的条件下，最小化最大内存的占用**。

1. 在开始训练之前，建立了下图所示的**6条内存管理规则**。 (1) Alloc; (2) Alloc & Division; (3) Swell; (4) Free; (5) Free & Merge; (6) Overwrite & Free。
1. 在前几次迭代过程中，追踪张量的访问信息。
1. 基于追踪到的信息，模拟张量的访问过程，并应用六种规则进行张量位置的重排，得到每个张量的最优内存块地址。

{% include figure.html src="/figures/MegTaiChi/15.PNG" caption=""%}

下图是TMA内存模拟访问的一个例子，首先根据追踪收集到的信息得到一个**内存访问序列Alloc/Free Sequence**。**Memory List和Idle List**分别记录了内存使用情况和空闲内存情况，**橙色和绿色**分别代表alloc和free**。**在模拟过程中，通过**拓扑图**记录内存块的相对位置。可以根据拓扑图确定每个请求的起始内存地址。

{% include figure.html src="/figures/MegTaiChi/16.PNG" caption=""%}

## （四）**Overhead Analysis**

#### DTP的开销：

主要的开销是将当前operator的划分模式从一种变为另一种的开销。如下表所示：

{% include figure.html src="/figures/MegTaiChi/17.PNG" caption=""%}

#### DTE的开销：

主要开销包括搜索需要驱逐的张量的开销，以及追踪张量的信息的开销。前者需要遍历所有的候选张量，非常耗时。后者的开销可以忽略。

#### TMA的开销：

主要成开销是对拓扑图进行排序，可以忽略不计。

# 三、实验结果

## （一）**Evaluation of DTP**

DTP在ResNet-50上性能改进较小，而对其他模型实现了性能加速。主要原因是ResNet-50每个卷积层通常都很小，更适合使用数据并行。

{% include figure.html src="/figures/MegTaiChi/18.PNG" caption=""%}

## （二）**Evaluations of DTE and TMA**

### 单个GPU上的测试

DTE减少了内存碎片的产生，可以降低张量驱逐的频率，从而实现了性能加速。另外通过协调DTE和TMA，通过内存细粒度的分配，可以实现更高的性能加速。

{% include figure.html src="/figures/MegTaiChi/19.PNG" caption=""%}

### 多个GPU上的测试

下图展示了在8个gpu上的测试结果。可以看到单GPU和多GPU，在吞吐量随着batch size大小的增加的变化趋势上相似，但8个GPU的吞吐量明显高于单个GPU。另外SPOS和MoE是动态模型，动态模型操作符的执行顺序不是固定的。与DTR相比，MegTaiChi将动态张量重计算扩展到了多个gpu上，减少了内存碎片。

{% include figure.html src="/figures/MegTaiChi/20.PNG" caption=""%}

## （三）**On Memory Usage**

下图展示了使用TMA和Pytorch-DTR训练时的内存使用情况。可以看到TMA的内存碎片更少，内存使用峰值更低。

{% include figure.html src="/figures/MegTaiChi/21.PNG" caption=""%}

## （四）Runtime Overhead of MegTaiChi

{% include figure.html src="/figures/MegTaiChi/22.PNG" caption=""%}