---
layout: post
mathjax: true
catalog: true
comments: true
top-tags-list: true
header-img: "img/post-bg-universe.jpg"
header-mask: 0.4
title: 优化算法测试问题集
subtitle: Test Functions for Optimization
author: 乔林波
tags: [测试, 优化]
---

为了测试优化求解算法的各方面特性，构建测试问题集，并标注优化解。

### Rastrigin
Formula
\begin{equation} \begin{array} {c}
f(x)=An+\sum_{i=1}^{n} (x_{i}^{2}-A\cos(2\pi x_{i})) \\\\  
A =10, x_{i} \in [-5.12,5.12] 
\end{array} \end{equation}

Global minimum:
\begin{equation} \begin{array} {c}
x^{\*}=(0,0,\cdots,0) \\\\ 
f(x^{\*}) = 0.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Rastrigin.png" caption="图：Rastrigin"%}




### Ackley
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x) =& -20\exp \left[-0.2{\sqrt {0.5\left(x_1^{2}+x_2^{2}\right)}}\right] \\\\  
&-\exp \left[0.5\left(\cos 2\pi x_1 +\cos 2\pi x_2\right)\right]+20+e \\\\  
&\text{where }, x_i\in [-5,5]
\end{array} \end{equation}

Global minimum:
\begin{equation} \begin{array} {c}
x_i^\*=(0,0)\\\\ 
f(x^\*) = 0.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Ackley-function.jpg" caption="图：Ackley"%}



### Rosenbrock
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=\sum_{i=1}^{n-1}\left[100\left(x_{i+1}-x_{i}^{2}\right)^{2}+\left(1-x_{i}\right)^{2}\right] \\\\  
\text{where } x_{i} \in [-\infty,+\infty]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(1,1,\cdots,1)\\\\ 
f(x^\*) = 0.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Rosenbrock.jpg" caption="图：Rosenbrock"%}



### Beale
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=\left(1.5-x_1+x_1 x_2\right)^{2} \\\\ 
+\left(2.25-x_1+x_1 x_2^{2}\right)^{2}+\left(2.625-x_1+x_1 x_2^{3}\right)^{2}\\\\ 
\text{where } x_{i} \in [-4.5,+4.5]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(3,0.5)\\\\ 
f(x^\*) = 0.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Beale.jpg" caption="图：Beale"%}


### Goldstein–Price
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=[1+\left(x_1+x_2+1\right)^{2}\\\\  \left(19-14x_1+3x_1^{2}-14x_1-2+6x_1 x_2 +3x_2^{2}\right)] \\\\  
(30+\left(2x_1-3x_2\right)^{2}\\\\ 
 \left(18-32x_1+12x_1^{2}+48x_2-36x_1 x_2+27x_2^{2}\right))\\\\ 
\text{where } x_{i} \in [-2,+2]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(0,-1)\\\\ 
f(x^\*) = 3.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Goldstein.jpg" caption="图：Goldstein–Price"%}



### Bukin
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=100{\sqrt {\left|x_2-0.01x_1^{2}\right|}}+0.01\left|x_1+10\right| \\\\ 
\text{where } x_{1} \in [-15,-5], x_{2} \in [-3,3]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(-10,1)\\\\ 
f(x^\*) = 0.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Bukin.jpg" caption="图：Bukin"%}



### Lévi
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=\sin ^{2}3\pi x_1+\left(x_1-1\right)^{2}\left(1+\sin ^{2}3\pi x_2\right)\\\\ 
 +\left(x_2-1\right)^{2}\left(1+\sin ^{2}2\pi x_2\right)\\\\ 
\text{where } x_{i} \in [-10, 10]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(1,1)\\\\ 
f(x^\*) = 0.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Levi.jpg" caption="图：Levi"%}


### Himmelblau
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=(x_1^{2}+x_2-11)^{2}+(x_1+x_2^{2}-7)^{2} \\\\ 
\text{where } x_{i} \in [-5,5]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(3,2),\\\\ 
(-2.805118,3.131312),\\\\ 
(-3.779310,-3.283186),\\\\ 
(3.584428,-1.848126)\\\\ 
f(x^\*) = 0.
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Himmelblau.png" caption="图：Himmelblau"%}


### Easom
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=-\cos \left(x_1\right)\cos \left(x_2\right)*\\\\ 
\exp \left(-\left(\left(x_1-\pi \right)^{2}+\left(x_2-\pi \right)^{2}\right)\right) \\\\ 
\text{where } x_{i} \in [-100,100]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(-\pi,\pi)\\\\ 
f(x^\*) = -1
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Easom.jpg" caption="图：Easom"%}



### Cross-in-tray
Formula
$\begin{array} {lc}
&\displaystyle f(x) = -0.0001*\\\\ 
&\left[\left|\sin x_1\sin x_2\exp \left(\left|100-{\frac {\sqrt {x_1^{2}+x_2^{2}}}{\pi }}\right|\right)\right|+1\right]^{0.1} \\\\ 
&\text{where } x_{i} \in [-10,10]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(1.34941,-1.34941)\\\\ 
(1.34941,1.34941)\\\\ 
(-1.34941,1.34941)\\\\ 
(-1.34941,-1.34941)\\\\ 
f(x^\*) = -2.06261
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Cross-in-tray.jpg" caption="图：Cross-in-tray"%}



### Eggholder
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=-\left(x_2+47\right)\sin {\sqrt {\left|{\frac {x_1}{2}}+\left(x_2+47\right)\right|}}\\\\ 
-x_1\sin {\sqrt {\left|x_1-\left(x_2+47\right)\right|}}\\\\ 
\text{where } x_{i} \in [-512,512]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(512,404.2319)\\\\ 
f(x^\*) = -959.6407
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Eggholder.jpg" caption="图：Eggholder"%}




### Hölder table
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=-\left|\sin x_1\cos x_2\exp \left(\left|1-{\frac {\sqrt {x_1^{2}+x_2^{2}}}{\pi }}\right|\right)\right| \\\\ 
\text{where } x_{i} \in [-10,10]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(8.05502,9.66459)\\\\ 
(-8.05502,9.66459)\\\\ 
(8.05502,-9.66459)\\\\ 
(-8.05502,-9.66459)\\\\ 
f(x^\*) = -19.2085
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Holder.jpg" caption="图：Holder"%}



### Schaffer
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)=0.5+{\frac {\sin ^{2}\left(x_1^{2}-x_2^{2}\right)-0.5}{\left[1+0.001\left(x_1^{2}+x_2^{2}\right)\right]^{2}}} \\\\ 
\text{where } x_{i} \in [-100,100]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x^\*=(0,0)\\\\ 
f(x^\*) = 0
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Schaffer.jpg" caption="图：Schaffer"%}



### Styblinski–Tang
Formula
\begin{equation} \begin{array} {cc}
\displaystyle f(x)={\frac {\sum_{i=1}^{n}x_{i}^{4}-16x_{i}^{2}+5x_{i}}{2}} \\\\ 
\text{where } x_{i} \in [-5,5]
\end{array} \end{equation}

Global minimum
\begin{equation} \begin{array} {cc}
x_i^\*=2.903534\\\\ 
-39.16617n\le f(x^\*) \le -39.16616n
\end{array} \end{equation}

{% include figure.html  height="300" width="400" src="/figures/2021-11-15-test-func/Styblinski-Tang.jpg" caption="图：Styblinski-Tang"%}

