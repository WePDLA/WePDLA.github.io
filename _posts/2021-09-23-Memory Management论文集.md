---
layout: post
title: 内存管理论文集
comments: True
author: 唐宇
---

作者： 唐宇


# Memory Management论文集

分布式训练通信密集型

- [Memory management论文集]()
  - [MemoryManagement综述](#memorymanagement综述)
    - [先验知识](#先验知识)
    - [关键技术](#关键技术)
    - [研究热点](#研究热点)
  - [Checkpoint技术](#checkpoint技术)
    - [研究背景](#研究背景)
    - [核心思想](#核心思想)
    - [checkpoint算法](#checkpoint算法)
  - [GPU集群的高效训练](#gpu集群的高效训练)
    - [Problem](#problem)
    - [KeyInsight](#keyinsight)
      - [通信优化](#通信优化)
      - [计算优化](#计算优化)
    - [有趣的点](#有趣的点)
  - [DL模型的内存消耗评估](#dl模型的内存消耗评估)
    - [Motivation](#motivation)
    - [文章的核心思路](#文章的核心思路)
  - [DTR](#dtr)
    - [研究现状](#研究现状)
      - 静态图显存优化
      - 动态图显存优化
    - [核心想法](#核心想法)
      - [实施路线](#实施路线)
      - [DTR策略](#dtr策略)
    - [工程实现](#工程实现)
      - [动态图核心](#动态图核心)
      - [释放和恢复Tensor](#释放和恢复tensor)
      - [引入DTR后的算子执行](#引入dtr后的算子执行)
      - [Tensor的删除](#tensor的删除)
    - [碎片问题和优化方法](#碎片问题和优化方法)
      - [参数原地更新](#参数原地更新)
      - [改进估价函数](#改进估价函数)
      - [静态分配策略](#静态分配策略)
    - [DTP执行策略](#dtr执行策略)
    - [展望](#展望)
    - [有关DTR的想法](#有关dtr的想法)
  - [Capuchin](#capuchin)
    - [问题](#问题)
      - [内存优化的两种方式](#内存优化的两种方式)
    - [设计Capuchin](#设计capuchin)
      - [设计目标](#设计目标)
      - [设计思路](#设计思路)
      - [Swap和重计算的benefit](#swap和重计算的benefit)
      - [Swap和重计算的选择](#swap和重计算的选择)
    - [Capuchin实验](#capuchin实验)
      - [Swap](#swap)
      - [Recompute](#recompute)
    - [个人的一些看法](#个人的一些看法)
  - [SuperNeurons](#superneurons)
    - [现状](#现状)
    - [设计核心](#设计核心)
        - [Liveness Analysis](#livenessanalysis)
        - [UTP](#utp)
            - [内存的offloading和prefetching](#内存的offloading和prefetching)
            - [CachingTensors](#cachingtensors)
        - [Cost-aware的重计算](#cost-aware的重计算)
    - [实验评估](#实验评估)
        - [有关内存的优化](#有关内存的优化)
        - [有关速度的优化](#有关速度的优化)
    - [Individuals](#individuals)
  - [一些自己的想法](#一些自己的想法)
  - [Pytorch中节省内存的小技巧](#Pytorch中节省内存的小技巧)

## Memory management in deep learning: a survey

来自华中科技大学计算机学院的研究，一篇综述

### 先验知识
1、深度学习模型训练不论模型结构如何，现有的内存分配策略没有考虑到DNN分层训练的特点，造成内存资源浪费
2、MP频繁通信，通信量大
3、GPU内存消耗的三个主要部分：
* FW中的activation，降低这部分的内存消耗也是目前大多数工作的主要目标，activation在FW和BW之间的两次计算中存在很大的时间间隔，为内存管理带来了很多可能；
* BW中的gradient 
* CNN所需的额外空间
**后两个部分时临时空间，可以在当前计算完成后立即释放**

### 关键技术
1、内存交换：在GPU内存和主存之间交换数据（GPU和CPU之间，<font color=purple>offload</font>，就是用时间换空间）具有代表性的工作：虚拟深度神经网络（virtualized deep neural network， vDNN），在FW的过程中就把activation交换到CPU上，但是GPU需要等待计算和传输全部完成才会进行下一层的计算，<font color=blue>数据在主存和GPU内存之间的传输时间不能完全与GPU计算时间重叠</font>
2、重计算：将FW中的activation及时释放，在BW的计算需要用到时再通过重新计算的方式生成（利用计算来换取内存空间）。但是现有的工作基于一个<font color=blue>假设：计算图中所有的activation的内存开销时相同的</font>，工作都局限于一些特定的神经网络中。代表性的工作：Training Deep Nets with Sublinear Memory Cost和Cost-Aware的重计算技术
3、内存共享：对不同变量生命周期进行分析，在不同变量之间重复使用同一块内存空间。两种方式：
* 置换操作：将输出结果直接存储在输入数据的物理地址上
* 内存复用：在生命周期不重叠的变量之间共享同一块内存

4、压缩：适用于压缩技术的数据需要具有高度稀疏性的特征

{% include figure.html src="/figures/memory-management/9.png" caption="图1：当前的技术比较分析。"%}

### 研究热点
1、内存管理策略
2、压缩技术的广泛性
3、编译器优化



## Checkpoint技术 

[pytorch实现](https://github.com/Lyken17/pytorch-memonger)
### 研究背景
1、memory内存墙仍然是目前深度学习领域中一个亟需解决的问题
2、共享内存只能在生命周期不overlap的变量中进行使用

### 核心思想
1、在FW时丢弃一些activation，BW的时候重新计算
2、inplace operation：将新变量直接覆盖不再需要的变量的内存
3、memory sharing

### checkpoint算法
图中的temp是临时的buff
{% include figure.html src="/figures/memory-management/10.png" caption="算法。"%}

对于一个N层的神经网络，每隔$\sqrt(n)$个层存一个activation。

## GPU集群的高效训练
paper: Efficient Large-Scale Language Model Training on GPU Clusters
### Problem 
1、GPU内存墙
2、训练大模型的计算操作会有很长的训练时间

### KeyInsight 

#### 通信优化
1、<font color=orange>scatter/gather communication optimization</font>：**DGX A100使用8个InfiniBand（IB）卡**。PP中所有的8卡做单点通信很难平衡，可以用tensor MP和pipeline MP减少跨节点之间的通信。对于大模型，使用大小为8的tensor-model-parallel，也就是说需要把通信张量在GPU之间send 8次。针对这个问题，在send节点把通信张量分成大小相同的chunk，然后使用IB卡仅把划分之后的chunk send到下一个节点的对应rank上（<font color=purple>这样的话，send就受限于IB</font>）。在receiver节点，**在NVLink做all-gather（<font color=purple>同理，all-gather为了重建tensor</font>），这要比IB快**。

#### 计算优化
1、把数据layout从[b,s,a,h]改成[s,b,a,h]
b：batchsize;
s：sequence;
a：attention-head;
h：hidden-size dimensions。

2、fused kernel：bias+GELU和bias+dropout+add（element-wise的操作）

3、new kernel： fusion of scale, mask and softmax function。

### 有趣的点

模型的参数量P计算

```
P=12lh^2(1+\frac{13}{12h}+\frac{V+S}{12lh})
```

FLOPS:
```
F=96BSlh^2(1+\frac{S}{6h}+\frac{V}{16lh})
```

## DL模型的内存消耗评估
Paper: Estimating GPU memory Consumption of Deep Learning Models
哭泣 这篇文章读的好累
### Motivation
1、已有的解决内存的方法不能直接应用到DL模型中，主要有以下三个原因：
* DL模型封装性，内部执行不易跟踪精确的GPU内存使用
* 一些底层实现的算子，比如Conv2d，分析这些算子的GPU内存占用比较困难
* 在框架运行中存在一些隐藏的位置因素

{% include figure.html src="/figures/memory-management/11.png" caption="算法。"%}

### KeyInsight
1、实现了一个front-end parser，从disk中读入DL model，这个parser负责从disk中读取model然后重建为一个对应的计算图
2、定义了两个分配内存的策略：ALLOC_ON_START（初始化阶段使用）和ALLOC_ON_DEMAND
两个释放内存的策略：RELEASE_ON_EXIT和RELEASE_ON_DEATH


## DTR

DTR:DYNAMIC TENSOR REMATERIALIZATION

一种动态图的显存优化技术——在前向计算时释放保存中间结果的tensor，反向求导时根据计算历史恢复之前释放的tensor。其实是一种<font color=red>动态重算</font>。

[知乎专栏](https://zhuanlan.zhihu.com/p/375642263)

[有关代码](https://github.cgs.me/uwsampl/dtr)
### 研究现状
1、静态图显存优化，可以分为三个方向：
* 静态内存分配。由于获得了整张计算图，所以可以去分析每一个 tensor 和每个算子的生命周期。对于生命周期没有重叠的算子，它们是可以共享显存的。
* 梯度检查点（用计算换显存）。设置一些梯度检查点，剩下的中间结果就先释放掉，如果将来在反向传播的过程中发现前向结果不在显存中，就找到最近的梯度检查点，恢复出被释放的tensor。
* 内存交换（用带宽换显存）。把暂时不用的数据从GPU上交换到 CPU 上，到了需要的时候，再把它交换回来。

2、动态图显存优化
<font color=blue>动态图无法提前获得全局的计算图信息</font>。因为无法得到每个 tensor 的生命周期，所以静态显存分配不再可用；梯度检查点还是可行的，且依然可以寻找最优的检查点；内存交换在动态图中仍然也是可用的。
所以动态图显存优化有两个方向：
* <font color=red>用计算换显存</font>，也就是动态图版的Sublinear显存优化（主要的优化方向）
* 用带宽换显存，在GPU和CPU之间交换内容

### 核心想法
#### 实施路线
1、基础设施：记录产生每个 tensor 的计算路径，使框架支持释放和恢复 tensor；
2、用户策略：提供释放tensor的接口，释放后框架会自动重算
3、自动策略：框架自动寻找策略并执行它，不需要用户的干预，做到用户对显存优化完全无感知。
 
#### DTR策略
是完全动态的启发式策略。它的核心就是当显存超过一个阈值的时候，动态地选择一些tensor将其释放掉，直到显存低于阈值。选择时会根据三方面对tensor进行估价：
```
1、重计算的开销越小越好；
2、占用的显存越大越好；
3、在显存中停留的时间越长越好。
```

除了重计算带来的开销之外，其他的额外开销主要用于寻找应该被释放掉的最优tensor。因为在显存中，tensor停留的时长是不断在变化的，所以只能在需要释放的时候现场计算最优的tensor。
论文中提出了两个运行时的优化技巧：
```
1、不考虑小的 tensor，当 tensor 大小小于候选集中的 tensor 的平均大小的 1% 时，不加入候选集；
2、每次在需要释放 tensor 的时候，随机采样 sqrt(N) 个 tensor 进行遍历（N 为目前可释放的 tensor 候选集的大小）
```
### 工程实现

#### 动态图核心
动态图核心——**Tensor Interpreter**
把python代码翻译成下面这四种基础操作，依次解释执行：
* `Put`：把外部数据从 host 端加载进显存中，得到一个 tensor
* `ApplyOp`：执行一个算子，它的参数是 op（算子）和输入 tensor，返回输出tensor
* `Del`：删除一个tensor，释放它在显存中占用的空间
* `GetValue`：获取一个tensor的值，需要把数据从显存中加载到host端

#### 释放和恢复Tensor
用户并不知道他访问的 tensor 当前是否在显存中，但是框架能保证当用户想获得tensor的内容时，就算它不在显存中，也可以立即恢复出来。
{% include figure.html src="/figures/memory-management/2.jpg" caption=""%}

如上图，若框架要释放掉当前这个 tensor 的显存，**reset 它的指针就可以把最底层的显存释放掉**。为了将来能够恢复出该 tensor，需要在 `tensorInfo` 中维护一些信息，如果使用 drop（用计算换显存）就需要记录计算历史；如果使用 swap（用带宽换显存），就需要把它先交换到 cpu 上记录一个 host tensor。将来如果用户访问了该 tensor，框架会检查它对应的 `tensorInfo`，如果发现已经不在显存上了，就根据计算历史或 host tensor 在显存中恢复出 tensor 的内容返回给用户。

#### 引入DTR后的算子执行

{% include figure.html src="/figures/memory-management/3.jpg" caption=""%}
上图是 DTR 核心的伪代码，对于`ApplyOp`方法，以往只需要执行**黄色**的代码，表示对 input 输入执行op算子。现在由于我们引入了 DTR 技术，这些输入 tensor 有可能已经不在显存中了。因此，执行前首先需要给它们打上标记，在这个算子执行完之前不能释放掉这些输入 tensor。然后调用 AutoEvict()，控制当前的显存占用不超过阈值。AutoEvict()方法将检查当前的显存占用，如果一直超过阈值就不断地调用FindBestTensor()算法，再根据启发式估价函数找出最优的 tensor 释放掉。

做完 AutoEvict() 之后，当前的显存占用已经低于阈值了，此时检查输入的每个 tensor 是否在显存中，如果不在显存中就调用 Regenerate()把它恢复出来，然后才能执行当前算子。Regenerate(x)的过程就是重计算 x 的过程，重计算的时候读取 x 的计算历史—— op 和 inputs，然后递归调用 ApplyOp 就可以恢复出 x。

#### Tensor的删除
{% include figure.html src="/figures/memory-management/4.jpg" caption=""%}
注意到，由于这里的 Elemwise 算子都是加法（scalar add），所以它的输入（两个红色的 tensor）在求导的时候都不会被用到。因此，求导器不需要保留住两个红色的 tensor，在前向计算完之后它们实际上是会被立即释放掉的。但在引入 DTR 技术之后，如果真的删掉了这两个红色的 tensor（把tensorInfo也删掉了），就会导致图中绿色的 tensor 永远不可能被释放，因为它们的计算源（红色 tensor）已经丢失了，一旦释放红色 tensor，绿色的 tensor 就再也恢复不出来了。解决方案是在前向的过程中用**Drop**来代替删除，不删除tensorInfo，也就是“假删除” —— 保留tensorInfo，只是释放掉tensorInfo下面对应的显存。这样只需要保留 9MB 的 tensor 就可以释放掉后面 4 个 25MB 的 tensor，并且可以在将来的任意时刻恢复出它们。
![](media/16299430196071/16301348831295.jpg)
上图就是 MegEngine 中对 tensor 的删除的伪代码实现，在解释器收到 Del 指令时，会对 tensorInfo 调用 Free()函数，如果是前向则为**假删除**，否则则尝试**真删除**，但只有在“某个Tensor既被用户删除，且没有任何tensor依赖它（ref_cnt = 0）的时候可以真正删除”。

假删除的实现很简单，打上删除标记，释放掉 tensorInfo 管理的显存即可；

### 碎片问题和优化方法
{% include figure.html src="/figures/memory-management/5.jpg" caption=""%}

#### 参数原地更新
就是inplace

#### 改进估价函数

引入碎片相关的信息
{% include figure.html src="/figures/memory-management/6.jpg" caption=""%}
有可能一个Tensor虽然不大，但是它左右的空闲段很大，因此释放这个Tensor也会有很大的收益

### 展望
显存优化策略的抽象：拿到这个序列后，既可以解释执行，也可以静态执行

#### 静态分配策略
{% include figure.html src="/figures/memory-management/7.jpg" caption=""%}
<font color=purple>这个方法应该是最有效的</font>
使用Pushdown算法，可以降低10%，不存在碎片

### 有关DTR的想法

好像DTR没有讨论communication和computation之间的overlap。



## Capuchin

Capuchin: Tensor-based GPU memory management for deep learning 

这篇文章发表在ASPLOS‘20，针对节省内存，主要思想用io换内存

### 问题

1、DNN训练中前向过程的中间层的输出（intermediate layer）占据了GPU内存的大部分。

#### 内存优化的两种方式
* swapping：swapping就是用io来换显存。swapping利用CPU的DRAM，将其作为一个更大的外部存储器，并且异步地将数据在CPU和GPU之间进行复制。Swapping的同步开销是巨大的，看图中的同步开销指的是在层与层之间的FWD和BWD过程中出现的overhead，毕竟之间存在计算依赖。Swapping还有一个决策性质的问题就是<font color=blue>什么时候做data在CPU和GPU的in/out</font>。
{% include figure.html src="/figures/memory-management/12.png" caption=""%}

* recomputing：recomputing就是用计算换显存。在forward过程中对某些feature map打checkpoint，然后在backward的时候利用这些checkpoints进行重计算。 
但是<font color=purple>不管是swapping还是recomputing都会增加训练时间</font>。

### 设计Capuchin 
#### 设计目标
1、最小化swapping的overhead
2、设计的框架要具备通用性，如果在不同的框架中修改的代码量要足够小。

#### 设计思路
对于swap，降低开销的最好方法是将swap与计算并行化；对于compute，就是要选择最便宜的情况做（说白了就是如果compute更快就用他，不然就用swap）。
把训练分成两个阶段，measured execution和guided execution。为了保证在OOM和Access Failure的时候Capuchin还可以继续执行，文章定义了一个Passive mode（On-demand swapping），就是在访问张量B的时候超出内存，张量A就会被释放。

measured execution：从第一个iteration中获得一些tensor的特征。在measured execution阶段，Capuchin记录下访问每个张量的信息：access count（就是这个张量在目前为止访问的次数），timestamp（时间戳）以及该张量对应的操作（重生成的时候用）。

guided execution：在第一个iteration之后的训练过程，根据测试阶段观察到的tensor access模式对执行过程进行决策。

个人觉得设计的实现思路是这样的：在Passive mode下，如果访问tensor超过了内存，那就对access count小的tensor进行释放，如果之后需要的话，再进行重生成。

{% include figure.html src="/figures/memory-management/13.png" caption=""%}

#### Swap和重计算的benefit

这一部分主要是为了评估swap和recompute的性能。那么对于Swapping定量分析的话，
```
SwapTime：从系统请求数据开始，到请求的数据转移到CPU上所有经历的时间
SwapInStartTime：预取目标tensor的时刻 = back-access time -SwapTime
SwapOutEndTime：逐出访问的时间加上SwapTime。
SwapOutTime：数据已经从GPU转移到CPU上的那个时刻。
FreeTime=SwapInStartTime-SwapOutEndTime
```
对于Swapping的计算，首先是计算SwapInStartTime，然后从张量访问列表中的back-access反向遍历，寻找时间早于SwapInStartTime的第一个张量访问。
在实际中，这样的计算方式可能并不准确，in-trigger机制不应该设置在内存峰值周围，为此而引入**feedback-driven调整**策略：在运行的时候动态调整in-trigger时间，在张量的反向访问时获得运行时反馈，如果张量仍在swap状态中，则意味着in-trigger时间应该提前调整（实现也比较简单，在张量的数据结构中增加一个swap status就行了）。

Memory Saving Per Second(MSPS)用来衡量重计算
```
MSPS=$\frac{Memory Saving}{Recomutation Time}$
```
举个例子：T1->T2->T3->T4，T1、T2、T4都需要进行重计算，如果对于T3的最后一次访问是在T4FWD的时候，在重计算T4的时候T3就变得不可用，因为在T3的最后一次访问之后，T3被释放了。那么T4的重计算就需要从T2开始，即使T2也需要重计算，只要T2在内存中，就不再需要从T1开始重计算T4。如果T3也在BWD过程中被访问到，比较T3的最后一个访问时间和T4的反向访问时间，若T3的最后一个访问时间较大，就说明T3在T4访问之后再进行访问，那T3在T4的重计算中可用是一个很直观的想法，否则的话，T3也需要被重计算。
根据这个例子，对于每个tensor，有两部分信息可以在重计算的时候使用：(1)张量自身的生命周期，主要用来判断它自身能构成为别的张量重计算时候的source；(2)张量自身是否需要进行重计算，需要重计算的张量都在GPU内存中（<font color=purple>我个人理解是这些张量不是在GPU内存中，而是他们不需要进行swap，在重生成的时候直接就在GPU中了，如果本身就在GPU内存中，那其实还是对内存有消耗</font>）。对于重计算的策略就是：<font color=red>一个Tensor重计算所需的时间越小，且它所需memory越大，那这个Tensor就更有价值去做重计算</font>。
{% include figure.html src="/figures/memory-management/16.png" caption=""%}


#### Swap和重计算的选择

<font color=red>swap可以和计算进行高度的overlap</font>。在swap和重计算中，优先选择swap。
首先系统会将符合以下两者的Tensor加入被处理列表：（1）Tensor会被处理多次（2）Tensor在内存密集的时候被访问。系统先根据FT（FT = SwapInStartTime - SwapOutEndTime）作为依据，如果一个Tensor的FT非常大那么意味着这个Tensor更值得去去swap或者recompute，那就把这个大FT的Tensor放到列表的前面。如果全部都用swap的方案不合适，那么就用混合方案，实际上就是对于每个tensor而言，选择开销较小的那个方式。
{% include figure.html src="/figures/memory-management/15.png" caption=""%}


### Experiments 

#### Swap

{% include figure.html src="/figures/memory-management/17.png" caption=""%}

#### Recompute

{% include figure.html src="/figures/memory-management/18.png" caption=""%}


### 个人的一些看法 

内存优化的思路就是：(1)io换显存；(2)计算换显存。根本思路都是时间换换空间，文章的思路把两种方式结合起来，好像这篇文章在DTR之前，所以没有提到动态重构张量的策略。在数据张量频繁io的时候，带来的时间消耗是巨大的。

## SuperNeurons

Paper: SuperNeurons: Dynamic GPU Memory Management for Training Deep Neural Networks

### 现状

1、GPU的DRAM限制了神经网络的deeper和wider；（<font color=purple>DRAM就是常说的GPU内存吗？</font>）
2、在GPU给张量分配内存和释放内存（allocation/deallocation）的时候，这种频繁的分配和释放操作会导致overhead；
3、在神经网络求解中的高度数据依赖，尤其是非线性的神经网络（<font color=purple>这也是做PP的时候存在的问题</font>）

### 设计核心
SuperNeurons：一个动态GPU内存调度运行机制

原始的内存消耗为：$\sum_{i=1}^{N}l_i^f+\sum_{i=1}^{N}l_i^b$，很容易理解：就是每个层前向和后向的内存消耗和。
{% include figure.html src="/figures/memory-management/19.png" caption=""%}
#### Liveness Analysis
{% include figure.html src="/figures/memory-management/20.png" caption=""%}

1、Liveness analysis可以让在不同时间分区的不同张量重复使用同一块物理内存，在Liveness Analysis的基础上，有如下改变：
* 对每一个层都构建一个*in*和*out*集合，跟踪记录在该层之前和之后的活跃张量，空间消耗为O(N);
* 运行时通过检查后续层的依赖关系来填充层的*in*和*out*集合。如果没有后续层需要它们，则将在*in*集合的张量从*out*消除。（<font color=purple>我觉得消除这个步骤主要发生在BWD的过程</font>）
假设每个层的消耗以及在FWD和BWD的时候的消耗相同。在FWD，第k个step的内存消耗$cost_k^f=\sum_{i=1}^k l_i^f$，在BWD的时候，第k个step中，i层之后的$l_i^f$和$l)i^N$会被释放掉。
Liveness analysis将内存消耗峰值从$\sum_{i=1}^{N}l_i^f+\sum_{i=1}^{N}l_i^b$降低到了$\sum_{i=1}^{N}l_i^f+l_N^b$，节省了50%。

2、由于使用原生cudaMalloc 和 cudaFree会造成大量的overhead，所以Liveness analysis预分配了一个size比较大的GPU memory作为共享内存池。然后把这个内存池分成1KB的block作为基本存储单元。内存池包含***allocated list***和***empty list***两个列表。这两个列表中的每个节点都包含内存地址、占用的块和节点 ID。对于*allocation*请求，内存池从empty list中找到第一个有足够空闲内存的节点。之后，它更新空列表并在allocated list中创建一个新节点来跟踪当前分配。对于释放请求，内存池使用<font color=red>**ID-to-node 哈希表**</font>在allocated list中定位节点，然后池将节点放回empty list。


#### UTP
Unified Tensor Poll(UTP):异步地将张量传入/传出外部存储器，比如CPU DRAM，DRAM或者其他的GPU和远程的CPU/GPU DRAM。（<color purple>远程的CPU/GPU DRAM怎么弄？</font>）文章主要关注本地的CPU DRAM。~~文章对于UTP怎么进行管理和如何进行管理并没有很详细的介绍，只是说了UTP对张量的放置、移动、分配和释放进行管理。~~

{% include figure.html src="/figures/memory-management/21.png" caption=""%}
##### 内存的offloading和prefetching

offloading：把tensor从GPU中转移到外部物理存储
prefetching：把tensor从外部物理存储转移到GPU

UTP实现管理的核心：对合适的层进行offloading和prefetching
不适合做offloading的层：POOL，ACT，BN，LRN等占据大量内存的层并且他们的计算量比较少。<font color=purple>个人觉得这样也比较容易理解：做offloading和prefetching就有了大量的通信，因为他们占了大量内存，通信量就比其他层高，但是他们的计算量比较少，就不好做communication和computation的overlap</font>。
主要是对CONV层做offload（<font color=purple>如果是对Transformer这样的模型呢？</font>）
1、offloading：runtime将CONV层的前向输出异步传输到预分配的固定CPU内存。

2、prefetching：runtime将offloaded而且马上就要用到的tensor重新取回到GPU DRAM中。在BWD的CONV层中，异步获取前一个 CONV 层所需的张量使得CPU-to-GPU的数据transfer可以和computation进行overlap。(<font color=purple>这个地方说的马上就要用到，怎么确定呢？</font>)

Offloading and Prefetching（在liveness analysis）每个backward step的内存消耗为：cost(k)=$\sum_{i=1}^k(l_i^f \notin checkpoints) + l_k^b$。


##### CachingTensors

Tensor Cache: 利用张量的时间局部性，在GPU DRAM对张量进行cache，最大化张量的使用，最小化全局的通信，只有在GPU DRAM不够的时候才会有数据的transfer。
在建立Tensor Cache的时候，采用Least Recent Used(LRU)策略，这是因为：反向传播展示了**head-to-tail和tail-to-head的计算模式**，也就是说使最近使用的张量会最早的进行重用。
{% include figure.html src="/figures/memory-management/22.png" caption=""%}
#### Cost-aware的重计算

SuperNeurons释放了<font color=red>计算成本低的层</font>（例如 POOL）中的张量以进行重建。通常分为memory-centric和speed-centric的策略。
* speed-centric的策略：保留需要重计算的tensor，在BWD的时候可以直接使用。
* memory-centric的策略：重计算每个backward层的依赖。

Cost-Aware Recomputation：在speed-centric的策略中利用最少的重新计算
* 在所有层中找到$l_{peak}=max(l_i)$
* 在每个重计算segment中，应用speed-centric的策略
这样就将原来的$peak_m$减少到了$max_{l_i}$。

### 实验评估

#### 有关内存的优化
{% include figure.html src="/figures/memory-management/23.png" caption=""%}

#### 有关速度的优化

{% include figure.html src="/figures/memory-management/24.png" caption=""%}


### Individuals
文章以层为操作对象，分析整个模型中的层对于整个内存消耗的影响，对于内存的优化主要采用重计算的方式，速度优化就是保证计算和通信的overlap（比如CacheTensors、offloading和prefetching）。
文章比较有意思的点是
* 把研究对象变成了层，每个层有一定的特性：比如CONV层，内存消耗小，但是计算量大，这样的层不适合做offload，也就是不适合做通信，那对Transformer这样的模型呢？都是矩阵乘操作，内存占据量大，计算量也比较大。
* 对于重计算操作，采用LRU策略，这样的策略可以达到减少计算时间的目的。当然我觉得对于不同的层，也应该有一个layer-centric的重计算策略，就是说针对不同的层，要做不同的free。直观上来说：内存占据量大的层要释放内存，要重计算这些层的值也不能代价过大，比如CONV层？（内存小，计算量大，但是这些层的依赖重计算的代价不大）


## SwapAdvisor

SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping

### SwapAdvisor的Motivation

影响swap性能的两个因素：
* 内存的分配：交换只有当GPU内存已满且在特定大小类别中没有空闲对象时的时候才会进行。
* 算子的调度

### SwapAdvisor核心设计
{% include figure.html src="/figures/memory-management/25.png" caption=""%}

#### operator schedule和memory allocation
operator schedule和memory allocation都是swap planner的输入。

* operator schedule: 用3个GPU的stream：一个进行正常的GPU执行；一个做swap out；一个做swap in。这三个stream可以同时进行。而且据观察，使用多个stream同时做计算操作的话，这三种stream就不能同时进行了，这也是SwapAdvisor只能单GPU的原因。
* memory allocation: 给要执行的计算图分配一个内存池，这个内存池由许多不同的大小类组成，每个类都被分配了一定数量的固定大小的张量对象。

#### Swap planning
Swap planning决定哪些tensor要做swap以及swap的时机。


* 目标确定：使用Belady策略来确定在将来较长的时间内都不会使用到的tensor。具体来说就是，Swap planning根据调度中的顺序扫描每个算子，并跟踪由于执行算子序列而驻留在内存中的输入/输出张量对象集。

* 时机确定：最好的swap in/out的时机是不影响相应的算子执行而且能够最大化通信和计算的overlap

### 有关SwapAdvisor的个人想法

SwapAdvisor实际上是对计算图来做的，就是从框架中获取原来DNN的计算图，operator scheduler和memory allocation作为swap planner的输入，swap planner的输出是一个优化后的计算图，把这个图给到框架之后，框架拿到这个新的计算图再执行。这样的方式比较受到框架本身的限制。

但是要获得比较好的offload的效果，对算子级别的细粒度优化是比较可靠的。


## Zero-Offload

Zero-Offload主要针对大模型和多GPU情况

### Zero-Offload的核心思想

{% include figure.html src="/figures/memory-management/26.png" caption=""%}

流程：把所有的fp32的模型状态和fp16的梯度offload到CPU上面，然后在CPU上做参数的更新；fp16的参数保存在GPU上， FWD和BWD的计算也在GPU上进行。在单个GPU上也是分stream进行的


## 一些自己的想法

**有关checkpoint**：在Tianqi Chen的paper中，一个N层的神经网络，每隔$\sqrt(n)$个层存一个activation，这样做checkpoint的策略应该是可以进行改进的，比如某些层可以选择合适的时机进行checkpoint。

一个好的swap应该尽可能地重叠通信和计算。

一些方法的比较
| Methods      | Keyinsight                                                      | Pros                                                                    | Cons                           |
|--------------|-----------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------|
| Capuchin     | Swap(对象:activation) & Recompute                                 | Swap和Recompute进行了较好的overlap                                             | 针对静态方法；swap的设计比较简单，只考虑了张量的生命周期 |
| SuperNeurons | Swap(利用Unified Tensor Pool对conv层进行offload或prefetch) & Recompute | 动态的swap机制；重计算的时候加入了两个执行策略（speed-centric和memory-centric）                 | swap只对conv层，粗粒度优化，没有考虑大模型的情况         |
| SwapAdvisor  | 只做swap                                                          | 利用GPU的stream做swap，对Swap的设计比较详细(planner、scheduler)、使用了Genetic Algorithm | 不支持多GPU情况，针对特定的框架    |   
| Zero-Offload  | 针对ZeRO optimizer的offload方式| 对于不同数据类型的数据区分比较明确，自己定义了CPU的optimizer来减缓参数在CPU上更新的时间延迟 | 针对Zero的特定优化，通用性不强 |  


## Pytorch中节省内存的小技巧
知乎连接：[Pytorch中节省内存的技巧](https://www.zhihu.com/question/274635237)
1、尽可能地使用inplace操作，比如relu可以使用`inplace = True`，比如
```
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
model.apply(inplace_relu)
```
<font color=purple>题外话</font>：
apply(fn)：不断遍历model的各个模块，将fn函数递归地应用到网络模型的每个子模型中，主要用在参数的初始化。

2、比如ResNet 和 DenseNet 可以将 batchnorm 和relu打包成inplace，在bp时再重新计算。使用到了pytorch新的<font color=red>**checkpoint**</font>特性，有以下两个代码。由于需要重新计算bn后的结果，所以会慢一些
* [efficient densenet pytorch](https://github.com/gpleiss/efficient_densenet_pytorch)
* [inplace abn](https://github.com/mapillary/inplace_abn)

3、每次循环结束时删除loss，可以节约很少显存，参考[issue](https://discuss.pytorch.org/t/tensor-to-variable-and-memory-freeing-best-practices/6000/2)
使用`del`方法
4、使用float16精度混合计算
5、对于不需要bp的forward，如validation请使用torch.no_grad，注意model.eval()不等于 `torch.no_grad() `，参考[问题](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
主要区别：
* `model.eval()` will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
* `torch.no_grad()` impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).

将不需要更新的层的参数从优化器中排除将变量的requires_grad设为 False，让变量不参与梯度的后向传播(主要是为了减少不必要的梯度的显存占用)

6、使用`torch.cuda.empty_cache()`这是del的进阶版，使用nvidia-smi会发现显存有明显的变化。但是训练时最大的显存占用似乎没变。每次用反而会减慢 1~2s.

7、在**Pytorch-0.4.0**出来了一个新的功能，可以将一个计算过程分成两半，也就是如果一个模型需要占用的显存太大了，我们就可以先计算一半，保存后一半需要的中间结果，然后再计算后一半。
也就是说，新的checkpoint允许我们只存储反向传播所需要的部分内容。如果当中缺少一个输出(为了节省内存而导致的)，checkpoint将会从最近的检查点重新计算中间输出，以便减少内存使用(当然计算时间增加了)。

8、跟踪显存使用情况：使用`pynvml`这个Nvidia的Python环境库和Python的垃圾回收工具，可以实时地打印我们使用的显存以及哪些Tensor使用了我们的显存。[Usage](https://github.com/Oldpan/Pytorch-Memory-Utils)

## Pytorch中的checkpoint
`torch.utils.checkpoint`，是Training Deep Nets with Sublinear Memory Cost这篇文章的实现。
pytorch的checkpoint是一种用时间换显存的技术，一般训练模式下，pytorch每次运算后会保留一些中间变量用于求导，而使用 checkpoint的函数，则不会保留中间变量，中间变量会在求导时再计算一次，因此减少了显存占用，跟tensorflow的checkpoint是完全不同的东西。<font color=purple>不过对非训练模式（torch.no_grad）没有用，因为非训练模式不需要求导，也不会有中间变量产生</font>。
参考：
* [https://blog.csdn.net/one_six_mix/article/details/93937091](https://blog.csdn.net/one_six_mix/article/details/93937091)
* [https://pytorch.org/docs/1.3.0/_modules/torch/utils/checkpoint.html#checkpoint](https://pytorch.org/docs/1.3.0/_modules/torch/utils/checkpoint.html#checkpoint)


欢迎共同探讨问题，email: ytang95@163.com
如果觉得我写的不错，对你有所启发的话，感谢打赏：
{% include figure.html src="/figures/memory-management/cash.jpg" caption=""%}

