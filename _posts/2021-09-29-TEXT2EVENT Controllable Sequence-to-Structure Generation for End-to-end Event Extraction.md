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

H = Encoder(x_1,...,x_{|x|})

## 2、解码

在编码器编码完成之后，解码器对输出进行生成。生成的顺序从前往后，每次生成新的token都需要用到已生成的信息。第i个token（y_i）的生成公式为：

y_i,h_{i}^d = Decoder([H;h_{1}^d,...,h_{i-1}^d],y_{i-1})

其中，h_i^d是decoder第i步的状态，H是Encoder的输出。解码的过程会有一个起始符“<bos>”，结束符为“<eos>”。解码器输出序列的条件概率为：

p(y|x)= \prod_{i}^{|y|}p(y_{i}|y_{<i})

# 三、结构化事件信息的线性表示

## 1、正常的事件信息表示

<<<<<<< HEAD
一般的事件信息都是这样的结构：


{% include figure.html  height="652" width="488" src="/pictures/2021-09-29-TEXT2EVENT Controllable Sequence-to-Structure Generation for End-to-end Event Extraction/event record format.png" caption="图1：Event Record format.。"%}





