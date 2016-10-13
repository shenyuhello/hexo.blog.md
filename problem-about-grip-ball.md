---
title: 抓球问题
date: 2016-10-13 12:56:25
tags: [概率,hash]
categories: 数学
---

## 问题描述

有一个箱子里装有n个白球，m个黑球。现按照如下规则去抓球，若抓到一个白球，则将其涂黑后再放入箱子，若抓到一个黑球，则直接将其放入箱子，一直抓下去直到箱子里所有的白球都变为黑球为止，问平均情况下需要抓多少次才能使箱子里所有的球都变为黑球？

<!--more-->

## 基本思路

该问题实际上就是一个求数学期望的问题，次数可能为从0到无穷大，直接计算每个次数相应的概率比较复杂，我们可以这样来想，先计算把第一个白球变成黑球用了多少次，再计算把第二个白球变成黑球用了多少次，依次下去直到所有的白球都变为黑球，最后把所有的次数加起来就等于问题的答案.

## 详细过程

设$X$为把箱子里所有的球都变为黑球时所需要抓球的次数，$Xi$为把第i个白球变为黑球所需要抓球的次数，则

$$
\begin{align}
E(X)&=E(\sum_1^nX_i)\\\\
&=\sum_1^nE(X_i)
\end{align}
$$

现在我们来计算$E(X_i)$.
此时，箱子中有$n-i+1$个白球，$m+i-1$个黑球，则

$$
\begin{align}
P(X_i=1)&=\frac {n-i+1}{m+n}\\\\
P(X_i=2)&=\frac {m+i-1}{m+n}\times\frac {n-i+1}{m+n}\\\\
...\\\\
P(X_i=k)&=(\frac{m+i-1}{m+n})^{k-1}\times\frac {n-i+1}{m+n}\\\\
\end{align} 
$$

按照期望定义可得

$$
\begin{align}
E(X_i)&=\sum_1^\infty k \times  P(X_i=k)\\\\
&=\frac {n-i+1}{m+n}\sum_1^\infty k (\frac{m+i-1}{m+n})^{k-1}
\end{align}
$$

可以看出$\sum_1^\infty k \times (\frac{m+i-1}{m+n})^{k-1}$是一个等差与等比相乘的求和，可使用高中学的错位相减法计算后，再求极限.这里使用高等数学求幂级数的方法来求，先求幂级数$\sum_1^\infty kx^{k-1}$的收敛域.

$$
\lim_{k \to \infty}|\frac {k+1}{k}|=1
$$

得其收敛半径为$R=1$，由于$(\frac {m-i+1}{m+n})^{k-1}<1$，所以上述级数必收敛.

设和函数为$s(x)$,即

$$
\begin{align}
s(x)&=\sum_1^\infty k x^{k-1}\\\\
&=\sum_1^\infty (x^k)'\\\\
\end{align}
$$

运用逐项求积公式得

$$
\begin{align}
\int_0^x s(t)~dt&=\int_0^x[ \sum_1^\infty (t^k)']~dt\\\\
&=\sum_1^\infty\int_0^x(t^k)'~dt \\\\
&=\sum_1^\infty x^k \\\\
&=\frac {x}{1-x}
\end{align}
$$

上式两边求导得

$$
\begin{align}
s(x)&=(\frac{x}{1-x})'\\\\
&=\frac{1}{(x-1)^2}
\end{align}
$$

为了使公式看起来更简便，令$p_i=\frac{m+i-1}{m+n}$，$q_i=\frac {n-i+1}{m+n}$，得

$$
\begin{align}
E(X_i)&=\frac {n-i+1}{m+n}\sum_1^\infty k  (\frac{m+i-1}{m+n})^{k-1} \\\\
&=q_i\sum_1^\infty kp_i^{k-1} \\\\
&=\frac{q_i}{(p_i-1)^2}\\\\
\end{align}
$$

现在来求总的期望

$$
\begin{align}
E(X)&=\sum_1^nE(X_i)\\\\
&=\sum_1^n\frac{q_i}{(p_i-1)^2}\\\\
&=\sum_1^n\frac{m+n}{n-i+1}\\\\
&=(m+n)\sum_1^n\frac{1}{n-i+1}\\\\
&=(m+n)\sum_1^{n}\frac{1}{n}
\end{align}
$$

于是经过$(m+n)\sum_1^{n}\frac{1}{n}$次抓取之后可把全部白球变为黑球.

## 总结

上述问题可以把它看成一个问题模型，从而应用到多种具体的情况。如在某种hash算法的实现中，将某一新项存入hash表时，随机选择hash表中的某一项，若该项为空则插入新的项，若不为空，则再随机选择下一项，直到找到一个空项插入为止，问该hash算法中，平均情况下需选择多少次才可以将hash表填满？这个问题就可直接运用的抓球问题模型进行求解.又比如我经常用默默背单词，我选了一本3000单词量的书，我每天都会随机从这3000个单词里面随机选择50个来背，我想知道，我大概需要被多少天才能把这3000个单词背完.这个问题经过小小的转化后同样可以用抓球问题模型来解决。