---
layout: post
title: TEXT2EVENT Controllable Sequence-to-Structure Generation for End-to-end Event Extraction
comments: True
author: 阚志刚
---

作者： 阚志刚

# 一、简介

这篇文章是来自中科院软件研究所的文章[链接](https://aclanthology.org/2021.acl-long.217.pdf)，发表在ACL2021上。这篇文章使用编码解码器的方法来完成篇章级事件抽取任务。这种做法的优点是，可以不需要token-level的精细标注，只需要record-level的粗粒度标注即可。考虑到任务的输入是一段文本，是一个序列，而输出的事件是一个结构化的信息，无法直接使用解码器进行生成。因此本文还提出了一种可逆的将结构化事件信息转换成线性表示的方法。

# 二、模型介绍

## 1、编码

使用多层transformer进行编码，其实就是使用一个预训练语言模型来编码，文章中用的是T5。表示为：

$$H = Encoder(x_1,...,x_{|x|})$$

## 2、解码

在编码器编码完成之后，解码器对输出进行生成。生成的顺序从前往后，每次生成新的token都需要用到已生成的信息。第i个token（$$y_i$$）的生成公式为：

$$y_i,h_{i}^d = Decoder([H;h_{1}^d,...,h_{i-1}^d],y_{i-1})$$

其中，$$h_i^d$$是decoder第i步的状态，H是Encoder的输出。解码的过程会有一个起始符“<bos>”，结束符为“<eos>”。解码器输出序列的条件概率为：

$$p(y|x)= \prod_{i}^{|y|}p(y_{i}|y_{<i})$$

# 三、结构化事件信息的线性表示

## 1、一般事件信息表示

一般的事件信息都是这样的结构

{% include figure.html src="/figures/2021-09-29-TEXT2EVENT/event record format.png" caption="Event Record Format"%}

这种形式的表示便于人类理解，但是用编码解码器难以直接生成这样的事件信息。注意，这里是record-level的数据，如果是token-level的标注数据的话，本身就是一个由“B、I、O”组成的序列了。

## 2、树结构的事件表示

为了使计算机能够读懂上面的事件信息，本文作者将上述表格形式的事件信息用树结构来表示。

{% include figure.html src="/figures/2021-09-29-TEXT2EVENT/event tree format.png" caption="Event Tree Format"%}

其中红色的边表示“event-role”关系，蓝色线表示“label-span”关系，句子中所有的事件节点都与root节点相连。同时为了保证树状的事件表示与接下来的线性化的事件表示可以相互转换，事件树中的同一深度的节点从左到右的顺序是span在文章中出现的顺序。相较于上面表格形式的事件表示这种树结构的事件新建信息更容易被计算机理解。然而，Decoder却不能生成这样的树结构。

## 3、线性化事件表示

在树结构的事件表示基础上，作者为了让Decoder能够顺利地生成出事件，又提出了一种线性化的事件表示方式。

{% include figure.html src="/figures/2021-09-29-TEXT2EVENT/event liner format.png" caption="Event Linearized Format"%}

这种线性的事件表示以“(”和“)”作为事件间和元素间的界限,此时在record-level这种粗粒度的数据中的事件就被表示成了一个字符序列。接下来的任务就是让模型能够生成这样的字符序列。
