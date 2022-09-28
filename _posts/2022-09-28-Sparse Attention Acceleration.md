---
layout: post
mathjax: true
catalog: true
comments: true
top-tags-list: true
header-img: "img/post-bg-universe.jpg"
header-mask: 0.4
title: Sparse Attention Acceleration with Synergistic In-Memory Pruning and On-Chip Recomputation
author: 杨智琳
tags: [内存管理]
---

self-attention机制虽然在现如今的研究上可以取得良好的性能，但它在**计算pairwise correlations时的计算开销较大**。虽然最近有一些关于对低attention得分元素进行运行时剪枝的一些研究，但**忽略了self-attention的复杂度以及在on-chip memory上的内存容量需求**。本文构建了**SPRINT加速器**来解决这些问题，它利用了ReRAM交叉阵列**固有的并行性**，并使用ReRAM中的**轻量级模拟阈值电路**来对低attention得分的元素进行剪枝。另外，SPRINT**重新计算**了获取到的少数数据的attention分数，来降低对模型准确率的影响。除此之外，还利用了相邻attention操作之间的**动态空间局部性**，来减少冗余的数据存取开销。

### 背景：

Self-attention机制虽然能取得良好的性能，但它的**相关计算和内存开销较大**，尤其是近年来深度学习模型的输入序列长度增加(例如>2K)的时候，它的开销大大增加。

最近的研究表明，**在输入上下文确定的时候，每个query只与少数key embeddings中的一个动态子集密切相关**。这种剪枝方法虽然看起来有效，但并没有有效地解决**数据通信开销**的问题。特别是在**输入序列不断增加且资源受限的设备**上时，现有的芯片内存资源可能不足以存储单个head的所有embeddings。

### 实现in-memory pruning的挑战：

- **Circuit inaccuracies：**限制了in-memory computing的精度。

- **Data conversion overhead：**将in-memory computing的模拟结果转换为数字结果并与阈值进行比较的开销可能较大。

- **Selective read of unpruned embeddings：**ReRAM剪枝需要特定的数据布局，但是这种布局会限制读取未剪枝的向量的能力。

### 本文的贡献：

- 采用了**近似的in-memory compute和精确的on-chip recompute**，来减轻Circuit inaccuracies对模型精度的负面影响。

- 使用**模拟比较器与阈值进行比较**，降低了硬件成本。

- 基于现有的观察**实现了数据重用**。在硬件方面，利用**transposable ReRAMs**来选择性地读取未剪枝的embeddings。在数据重用方面，利用相邻查询的未剪枝的key vector之间的**空间局部性**提高了数据的重用性，并进一步减少了数据通信的开销。

### 数据通信优化：

在on-chip memory容量受限的情况下，计算self-attention scores需要在相邻的query vectors之间进行频繁的数据移动。下面是几个减少数据通信的方法：

#### 1) In-memory Thresholding

在on-chip资源受限的情况下，可以利用in-memory computing来消除不必要的数据通信。如下图，当q1→“the”时，q1×KT的计算只需要K2、4、5、6、11、13。由这个观察结果可以知道**可以通过只获取必要的数据来大大减少数据通信**。

{% include figure.html src="/figures/Sparse Attention Acceleration/1.png" caption=""%}

#### 2) Spatial Locality in Adjacent Queries

虽然in-memory thresholding可以减少了每个查询的数据通信量，但它**增加了数据获取的频率**。因为新的查询需要重新获取新的数据进行计算。由于观察发现**每个query中有大量的key是不需要的，并且在相邻的行之间有很高的空间局部性**。比如，比较“the”和相邻query “more”的key，只有“appear”和“in”是不同的，其余未剪枝的key都是相同的，从而避免了额外的数据通信。

#### 3) Futile Computations in Padded Regions

在transformer模型中，padded部分的计算是无效的计算，与最终模型的精度无关。如上图中灰色方块区域所示，**大量的padding会造成不必要的数据通信**。所以尽早识别padding区域，可以进一步消除这些不必要的数据通信。

### IN-MEMORY THRESHOLDING

#### **Application in run-time pruning：**

in-memory可以应用于加速attention机制。如下图所示：可以通过将**每个ki向量存储在每一列中，将查询向量qi分配给水平wordline**来实现。

{% include figure.html src="/figures/Sparse Attention Acceleration/2.png" caption=""%}

#### **Analog**↔**Digital challenges：**

现有的研究表明，**Analog**↔**Digital之间的转换，消耗了ReRAM的大量功率**。尤其是对于高精度要求的情况，功率**会随着比特数的增加而增加**。所以考虑这些转换的功率开销也很重要。

- #### **In-Memory Thresholding Challenges：**

- -  主要挑战如下：

- **模拟计算的不准确性：**

ReRAM中的模拟计算的不准确性有可能会对ReRAM计算的精度造成影响。为了评估in-memory thresholding有限的计算精度对最终模型精度的影响。作者比较了三种不同模型在不同精度量化的情况下最终的模型精度。如图所示，结果表明**4bit精度的量化误差对最终模型的精度没有影响**。因此，**运行时剪枝机制对计算误差具有鲁棒性**。

{% include figure.html src="/figures/Sparse Attention Acceleration/3.png" caption=""%}

- **ADC转换的开销：**

**ADC转换的开销与转换的精度成正比**。由于得到的剪枝向量每个key只需要一个bit，所以本文采用的方法是**使用只有1bit的ADC**，从而降低转换开销。

- **读取未剪枝向量的开销：**

由于每个K向量垂直地存储在ReRAM的每列上，所以当我们需要读取未剪枝的K个向量时，需要读入所有水平的wordline，会**造成大量的read latency**。本文使用了**transposable ReRAM**来解决这个问题。

#### Transposable ReRAM for Thresholding：

transposable ReRAM支持**in-situ computation和transposed read**，结构如下图所示，则之前的**读取未剪枝向量的问题则可以通过“transposed read”来解决**。

由于analog circuit noises限制了每个存储单元上支持的位精度，上面的实验表明**4bit精度的量化误差对最终模型的精度没有影响**，所以作者在transposable ReRAM中**每个key vector只存储了4个MSB**。剩余的LSB存储在了conventional ReRAM modules上。query和value vectors也存储了conventional ReRAM modules上。

{% include figure.html src="/figures/Sparse Attention Acceleration/4.png" caption=""%}

### OVERVIEW OF SPRINT SYSTEM

整个SPRINT的架构如下图所示，主要包括两个组件：**1) ReRAM memory和**

**2) on-chip accelerator**。

- **ReRAM memory**

ReRAM memory被分为两类，standard和transposable。**Standard ReRAM**仅用于存储（Q、V和$K_{LSB}$）；**Transposable ReRAM**既用于存储（$K_{MSB}$），也用于执行点积，in-memory thresholding，通知片上加速器要获取的向量。

- **On-chip accelerator**

SPRINT的片上加速器执行三个主要操作，**$q×K^T$、Softmax和$×V$**。如图所示，加速器从ReRAM中获取未剪枝的k/v向量以及**一个二进制向量，用于表示哪些k/v索引未被剪枝**。基于这些索引，**地址生成器**生成地址，来访问k/v缓冲区中未剪枝的向量。加速器首先执行$q_{1×d}×{K_{MSB}，K_{LSB}}$，然后使用加法树来精确地计算score。然后，**Softmax模块**将score进行归一化。**最后一个模块**将每个v向量乘以其对应的score概率，然后对加权v向量进行reduction sum，生成最终的attention score。

{% include figure.html src="/figures/Sparse Attention Acceleration/5.png" caption=""%}

### SPRINT MEMORY CONTROLLER

由于对同一行的后续列进行访问具有**行缓冲区局部性**，因此**访问时延较低**。但是**不同行之间的连续访问的访问时延较高**。内存控制器的目的是调度内存命令，以**最大化行缓冲区局部性**。

#### Data Layout Organization：

基于观察到的未剪枝的key indices之间的空间局部性结果，作者**将相邻的k vectors分配在了不同的banks/channels上**。k vector这样的分布可以**更好地利用内存带宽，减少冲突**。v vectors也适用于这样的数据布局。

#### **Scaling for embedding size：**

随着key vector的embedding size增加，在ReRAM的每个列上进行reduction sum变得不可行。这个问题可以**通过将key vector分割为多个相邻的ReRAM列**来地解决。

#### Memory Controller Execution Flow:

SPRINT的内存控制器的主要执行流如下：先向$K_{MSB}$ ReRAM banks发送一个$q_i$向量的低精度变体，然后执行in-memory thresholding并生成二进制剪枝向量（‘1’→剪枝和‘0’→未剪枝），最后处理二进制剪枝向量并发送未剪枝向量的读取请求流。

### SPRINT ON-CHIP ACCELERATOR

#### **Workload balancing across CORELETs：**

SPRINT加速器可以同时处理每个CORELET中的多个 key vector子元素，一旦一个query及其所有关联的key计算完成，下一个query的计算就可以开始了。作者**将相邻的key vector分配给了不同的CORELETs，称之为token-interleaving**。这可以在考虑空间局部性的同时**平衡了跨CORELETs的工作负载**。

#### **Handling data misses：**

为了最小化由于**数据丢失**而造成的延迟，可以**对未剪枝的key vector进行prefetch**。考虑到较高的数据重用性，当出现数据丢失的情况时，可以**继续对下一个可用key vector进行计算**，直到内存控制器处理丢失的数据为止。

### 实验结果

##### 三种实验配置：

- (1) S-SPRINT: a CORELET with 16KB, 

- (2) M-SPRINT: two CORELETs with 32KB, 

- (3) L-SPRINT: four CORELETs with 64KB total on-chip buffer capacity. 

##### **Impacts on model accuracy from in-memory pruning：**

可以看出on-chip recompute在保持模型精度方面的有很强的重要性。

{% include figure.html src="/figures/Sparse Attention Acceleration/6.png" caption=""%}

##### **Main memory data movement analysis：**

**(1) “Mask Only”** *→* sequence reduction for the padded area;

**(2) “SPRINT”***→* run-timing pruning on top of the sequence reduction of the padded area.

{% include figure.html src="/figures/Sparse Attention Acceleration/7.png" caption=""%}

##### **Performance and energy comparison：**

S-、M-和l-三种实验配置下平均分别能加速7.5×、7.4×、7.1×倍。加速的原因在于运行时in-memory剪枝导致跳过了大多数计算周期。

{% include figure.html src="/figures/Sparse Attention Acceleration/8.png" caption=""%}

##### **Energy consumption breakdown：**

Energy breakdown主要包括: 

- (1) ReRAM read/write,

- (2) in-ReRAM pruning, 

- (3) on-chip K/V buffers read/write,  

- (4) computations in QK-PU, Softmax, and V-PU.

{% include figure.html src="/figures/Sparse Attention Acceleration/9.png" caption=""%}

##### **Comparison with** *A*3 **, SpAtten, and LeOPArd：**

{% include figure.html src="/figures/Sparse Attention Acceleration/10.png" caption=""%}