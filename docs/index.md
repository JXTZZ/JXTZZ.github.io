---
title: 首页
hide:
  - navigation
  - toc
  - footer
---

<div class="home-hero" markdown>
<div class="hero-copy" markdown>
<div class="eyebrow">LOTUSY / FIELD NOTES</div>

# 保持好奇，<br>让知识慢慢生长。

从深度学习的第一行代码，到让机器人理解并行动。<br>这里收藏我的论文阅读、工程实践，和那些值得记下的发现。

<div class="hero-actions" markdown>
[开始阅读 →](papers/index.md){ .md-button .md-button--primary }
[关于我](about.md){ .quiet-link }
</div>
<div class="hero-caption">LEARN. BUILD. UNDERSTAND.</div>
</div>
<div class="hero-art" aria-hidden="true">
<svg viewBox="0 0 480 400" xmlns="http://www.w3.org/2000/svg">
<defs><pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="currentColor" opacity=".09"/></pattern></defs>
<rect x="20" y="20" width="440" height="360" rx="20" fill="url(#grid)"/>
<circle cx="250" cy="190" r="130" fill="none" stroke="currentColor" stroke-dasharray="4 8" opacity=".25"/>
<ellipse cx="244" cy="330" rx="158" ry="23" fill="currentColor" opacity=".06"/>
<path d="M96 327h254M149 314v-22h86v22" fill="none" stroke="currentColor" stroke-width="3"/>
<path d="M191 290l-31-98 103-74 68 64" fill="none" stroke="currentColor" stroke-width="22" stroke-linejoin="round"/>
<path d="M191 290l-31-98 103-74 68 64" fill="none" stroke="#f0f1df" stroke-width="12" stroke-linejoin="round"/>
<g fill="#dce8aa" stroke="currentColor" stroke-width="3"><circle cx="191" cy="286" r="18"/><circle cx="160" cy="192" r="18"/><circle cx="263" cy="118" r="16"/></g>
<path d="M320 190l13-13 25 19-9 22m-29-28 2 27 17 8" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M328 275l28-16 28 16v32l-28 16-28-16zM328 275l28 16 28-16m-28 16v32" fill="#dce8aa" stroke="currentColor" stroke-width="2"/>
<path d="M344 233v15M311 261l-11-7M388 253l12-7" stroke="currentColor" stroke-width="2"/>
<g fill="currentColor" font-family="monospace" font-size="10" letter-spacing="2"><text x="42" y="58">OBSERVE → LEARN → ACT</text><text x="48" y="354">FIG. 01 / EMBODIED INTELLIGENCE</text><text x="315" y="100">π(a | o)</text></g>
<circle cx="410" cy="57" r="5" fill="#176c58"/>
</svg>
</div>
</div>

<div class="home-meta"><span>南京大学 · 软件工程</span><span>具身智能 / VLA / 深度学习</span><span>持续更新的个人知识库</span></div>

<div class="section-heading"><span class="eyebrow">01 / EXPLORE</span><h2>沿着兴趣，找到一条线索</h2></div>

<div class="topic-grid">
<a class="topic-card" href="papers/"><span class="card-number">01 — RESEARCH</span><h3>论文阅读 <span>↗</span></h3><p>从 ALOHA、DP3 到 OpenVLA，拆解机器人策略的任务、方法与边界。</p><span class="card-foot">具身智能 · 论文精读</span></a>
<a class="topic-card" href="deep-learning/"><span class="card-number">02 — FOUNDATIONS</span><h3>理论基础 <span>↗</span></h3><p>用数学推导和代码，串起回归、神经网络、正则化与注意力机制。</p><span class="card-foot">深度学习 · 数学直觉</span></a>
<a class="topic-card" href="engineering/"><span class="card-number">03 — PRACTICE</span><h3>工程实践 <span>↗</span></h3><p>记录 Tensor 的细节、开发中的问题，以及从想法到实现的过程。</p><span class="card-foot">PyTorch · 开发手记</span></a>
</div>

<div class="home-bottom" markdown>
<div markdown>
<div class="section-heading"><span class="eyebrow">02 / RECENT NOTES</span><h2>最近在学什么</h2></div>

<a class="note-row" href="journal/2026-09-10/"><time datetime="2026-09-10">09.10<small>2026</small></time><div><span class="note-tag">PYTHON / 数据结构</span><h3>重新理解 Python 中的堆</h3><p>从完全二叉树到 heapq，把基础补扎实。</p></div><span class="row-arrow">↗</span></a>
<a class="note-row" href="journal/2026-09-08/"><time datetime="2026-09-08">09.08<small>2026</small></time><div><span class="note-tag">DAILY / 学习日志</span><h3>RoboSPA 与一次工具排障</h3><p>空间推理的阅读速记，和 Snipaste 自启问题复盘。</p></div><span class="row-arrow">↗</span></a>
<a class="note-row" href="journal/2026-09-07/"><time datetime="2026-09-07">09.07<small>2026</small></time><div><span class="note-tag">DAILY / 学习日志</span><h3>从动作推理到评估泛化</h3><p>DroneCATS、MINERVA 阅读记录与 WSL 网络配置。</p></div><span class="row-arrow">↗</span></a>

[全部学习日志 →](journal/index.md){ .text-link }
</div>
<aside class="reading-note" markdown>
<span class="eyebrow">ON MY DESK</span>

### 让阅读之间发生连接

一篇论文的价值，也在于它与已有知识产生的联系。最近的阅读围绕一个问题展开：机器人怎样从观察走向行动？

<div class="reading-path" markdown>
[01　ALOHA · 双臂灵巧操作](papers/aloha.md)

[02　DP3 · 三维表示与动作生成](papers/dp3.md)

[03　OpenVLA · 视觉语言动作模型](papers/openvla.md)

[04　SPOT · 从人类演示学习物体轨迹](papers/spot.md)
</div>
</aside>
</div>
