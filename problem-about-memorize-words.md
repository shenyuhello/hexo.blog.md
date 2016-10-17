---
title: 背单词问题
date: 2016-10-15 22:24:28
tags: [概率,数学]
categories：数学
---

## 问题描述

有一本英语单词书共包含3000个单词，每天随机从中选择50个不重复的单词来背（可能包含已经背过的单词），问按照这种背法平均情况下需要多少天才能把3000个单词全部背完？

<!--more-->

## 基本思路

如果每天选择的50个单词允许重复，则可以按照抓球模型来计算（抓球模型详见我另一篇文章“[抓球问题](/2016/10/13/problem-about-grip-ball/)”），由于每天选择的50个单词不允许重复，这样就使问题变得不那么简单了。假设$X$为把3000单词背完所需要的天数，那么该问题就是要算$X$的期望$E(X)$，$X$范围可以从60到无穷大，直接算每个$X$对应的概率比较复杂，我们可以仿照抓球问题的办法将随机变量$X$进行分解后在计算期望。

## 详细过程

为了解决这个问题，我们先给出如下定理。

**定理一：设有三个随机变量$X、Y、Z$，其中**

$$
X\in\{x_1,x_2,...x_i\},P(X=x_i)=p_i.\\\\
Y\in\{y_1,y_2,...y_j\},P(Y=y_i)=q_i.\\\\
Z \in\{x_1,x_2,...x_i\}\cup\{ y_1,y_2,...y_j\},P(Z=x_i)=wp_i,P(Z=y_j)=r q_j,w+r=1.
$$

**那么$E(Z)=wE(X)+rE(Y)$**.

证明：

因为
$$
\begin{align}
\sum_1^iw p_i+\sum_i^jrp_j&=w\sum_1^i p_i+r\sum_i^jp_j\\\\
&=w+r\\\\
&=1
\end{align}
$$
所以
$$
\begin{align}
E(Z)&=\sum_1^ix_iP(Z=x_i)+\sum_1^jy_iP(Z=y_i)\\\\
&=\sum_1^iwx_ip_i+\sum_1^jry_jq_j\\\\
&=w\sum_1^ix_ip_i+r\sum_1^jy_jp_j\\\\
&=wE(X)+rE(Y)
\end{align}
$$

证毕.

这个定理给我们提供一个期望的方法，如果直接求随机变量$Z$的不太容易时，可将$Z$按照取值范围分成$X$和$Y$两个部分，而$Z$按照一定的概率取$X$的值或$Y$的值，那求$Z$的期望可归结为求$X$和$Y$的期望。以上结论还可推广到分成$N$个部分的情况,接下来就用这个定理来解决“背单词问题”。

**定义：设有$n-i$个新词，$m+i$个旧词，每天从中随机选择$k$个单词来背$(0\le i\le\min(n,k))$，经过$X_i$天之后把所有的单词背完。**

根据这个定义，$X_0$就是在有$n$个新词，$m$个旧词的情况下背完单词的天数；$X_1$就是在有$n-1$个新词，$m+1$个旧词的情况下背完单词的天数...，我们目的就是要求$E(X_0)$,直接求$E(X_0)$比较麻烦，但是我们可以利用$E(X_i)(0\lt i\le\min(n,k))$的结果来求得$E(X_0)$。

在有$n$个新词，$m$个旧词,每天抽取$k$单词的情况下，第一天所抽取的新词数为T,则$\max(k-m,0)\le T\le \min(n,k)$,则$P(T=t)=\frac{C_n^tC_m^{k-t}}{C_{m+n}^{k}}$。那么第一天过后有$n-t$个新词，$m+t$个旧词,根据定义需要$X_t$天背完。设$X_t\in\{x_{t1},x_{t2},...x_{ti}\}$，则$X_t+1\in\{x_{t1}+1,x_{t2}+1,...x_{ti}+1\}$那么$P(X_0=x_{ti}+1)=P(T=t)P(X_t+1=x_{ti}+1)$(这里没有考虑各个$X_t$之间取值有可能重复，实际上是否重复并不影响我们计算期望),为了公式看起来比较简便令$a=max(k-m,0),b=min(n,k)$，利用定理一的结论可得

$$
\begin{align}
E(X_0)&=\sum_{t=a}^bP(T=t)E(X_t+1)\\\\
&=\sum_{t=a}^bP(T=t)E(X_t)+P(T=t)\\\\
&=\sum_{t=a}^bP(T=t)+\sum_{t=a}^bP(T=t)E(X_t)\\\\
&=1+\sum_{t=a}^bP(T=t)E(X_t)
\end{align}
$$

在给定$k$的情况下，注意到$E(X_t)$的值只与$n-t,m+t$有关，设$E(X_t)=f(n-t,m+t)$,所以$E(X_0)=f(n,m)$。可得如下递推公式

$$
f(n,m)=1+\sum_{t=max(k-m,0)}^{min(n,k)}\frac{C_n^tC_m^{k-t}}{C_{m+n}^{k}}
f(n-t,m+t);(n\ne0)
$$

很遗憾我无法给出一个封闭式，这就是最终的结论。接下来我们利用这个公式计算f(3000,0),注意到等式的右边也可能含有$f(n,m)$项，为了方便递归计算，将等式改写成如下形式

$$
f(n,m) =
\begin{cases}
1+\sum_{t=k-m}^{min(n,k)}\frac{C_n^tC_m^{k-t}}{C_{m+n}^{k}}
f(n-t,m+t);k\gt m,n\ne0\\\\
\frac{1+\sum_{t=1}^{min(n,k)}\frac{C_n^tC_m^{k-t}}{C_{m+n}^{k}}
f(n-t,m+t)}{1-\frac{C_m^k}{C_{m+n}^{k}}};& k\le m,n\ne0  \\\\
0;(n=0)\\\\
\end{cases}
$$

## 实验

为了解决我们在最开始提出的问题，我们需要利用递推式来计算$f(3000,0)$，接下来我们用python语言来实现这个计算过程，计算代码如下。

``` python
# coding=utf-8


# 计算连乘
def facc(n, m):
    prod = 1
    for i in range(n, m + 1):
        prod *= i
    return prod


# 计算组合数
def C(n, m):
    if n == 0 or m == 0:
        return 1
    return facc(n - m + 1, n) / facc(1, m)


s = 3000  # 总的单词数
k = 50  # 每天抽取的单词数
record = [None] * (s + 1)  # 存储计算结果，record[n]存储f(n,m)的值


# 运用自底而上的方法计算f(n,m)
def f(n, m):
    record[0] = 0
    for i in range(1, n + m + 1):
        j = n + m - i
        # 下面的过程相当于计算f(i,j)
        if k > j:
            record[i] = 1
            for t in range(k - j, min(i, k) + 1):
                record[i] += 1.0 * C(i, t) * C(j, k - t) / C(i + j, k) * record[i - t]
        else:
            record[i] = 1
            for t in range(1, min(i, k) + 1):
                record[i] += 1.0 * C(i, t) * C(j, k - t) / C(i + j, k) * record[i - t]
            record[i] /= 1 - 1.0 * C(j, k) / C(i + j, k)


f(s, 0)  # 计算结果

print record[s]

```
由于在给定总单词数$s$的情况下,$m$的值可以由$n$的值确定，所以只需要一个一维数组存储中间结果。运行上述代码的输出为511.29694026,也就是说在总单词为3000个，每天随机50个背的情况下，大概需要511.29694026天背完。

为了进一步判断我们的推理是否正确，我们来做一下仿真实验，也就是利用计算机来模拟一下背单词的过程，看看与我们的计算结果相差多远，仿真的代码如下。

``` python
# coding=utf-8


import random

s = 3000  # 总单词数
k = 50  # 每天抽取的单词数
random_num = range(0, s)  # 从中随机抽取单词
times = 100000  # 实验模拟次数
acc_X = 0  # 总共背单词的天数
count_times = times  # 还未完成的实验次数

while count_times > 0:
    word = [False] * s  # s个没有背过的单词
    memory_word_count = 0  # 背过的单词个数
    X = 0  # 天数
    while memory_word_count < s:
        memory_word_num = random.sample(random_num, k)
        for i in memory_word_num:
            if not word[i]:
                memory_word_count += 1
                word[i] = True
        X += 1
    print X
    acc_X += X
    count_times -= 1

print acc_X * 1.0 / times

```
上述代码在我的计算机上，某次的输出结果为511.74665，可以看出实验的结果和我们计算的结果是很接近的。

## 总结

这个问题是我在写“[抓球问题](/2016/10/13/problem-about-grip-ball/)”的一个扩展问题，最开始我也以为可以很简单的用抓球模型来解决，在进行深入思考之后，发现这个问题却并非如此容易，关键在于每天所抽取50个单词是不会重复的，这就大大增加了复杂性。我们在解决问题的时候是将原问题分解为多个小的子问题，然后将子问题的解合并得到原问题的解，类似于算法里面的“分治算法”的思想。上面的过程只得到了一个递推式，并没有得到一个封闭的解，如果你有一个封闭解或者对上面的推理过程有其他任何的想法，欢迎发送邮件到我的邮箱<shenyuhello@163.com>讨论。
