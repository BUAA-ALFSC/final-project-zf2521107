# Meta-Learning PINN Loss Functions

基于 MindSpore 深度学习框架复现 Psaros 等人 (2022) 的论文 *"Meta-learning PINN loss functions"*。

[![MindSpore](https://img.shields.io/badge/MindSpore-2.x-blue)](https://www.mindspore.cn/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 团队成员

|学号|姓名|
| --- | --- | 
|ZF2521107|丁佩东|
|ZF2521122|李广志|


## 项目简介

物理信息神经网络（Physics-Informed Neural Networks, PINN）求解偏微分方程时，其性能高度依赖于损失函数中各项权重的设计。传统方法采用均匀的均方误差（MSE）损失，无法适应不同 PDE 任务的特异性。

本项目使用 **First-Order MAML**（模型无关元学习）算法，从参数化 PDE 任务分布中自动学习最优损失函数，使得 PINN 在求解新的、未见过的 PDE 参数时表现更优。

### 验证问题：一维 Burgers 方程

$$u_t + u u_x = \nu u_{xx}, \quad x \in [-1, 1], t \in [0, 1]$$

$$u(x,0) = -\sin(\pi x), \quad u(-1,t) = u(1,t) = 0$$

### 核心方法

| 组件 | 说明 |
|------|------|
| **内循环** | 在每个任务上，使用当前损失函数对 PINN 参数做 K=5 步 SGD 适配 |
| **外循环** | 在采样任务上评估适配后模型，使用 Adam 更新损失函数参数 |
| **First-Order 近似** | 通过 `stop_gradient` 断开二阶梯度路径，降低计算复杂度 |
| **LAL 参数化** | $\ell_{\alpha,c}(r) = \frac{\|\alpha-2\|}{\alpha}\left(\left(\frac{r^2/c^2}{\|\alpha-2\|}+1\right)^{\alpha/2}-1\right)$，4 个可学习参数 |
| **FFN 参数化** | [2, 40, 40, 1] 无偏置 ReLU+Softplus 网络，1720 参数，Eq.28 正则化 |

## 环境依赖

- **Python** >= 3.8（Notebook 记录环境为 Python 3.10.11）
- **MindSpore** >= 2.0（Notebook 记录环境为 MindSpore 2.9.0）
- **NumPy** >= 1.20
- **SciPy** >= 1.8
- **Matplotlib** >= 3.5
- **Jupyter Notebook**（运行 `MetaPINN_Reproduction.ipynb` 时需要）

### 安装

```bash
pip install mindspore numpy scipy matplotlib jupyter
```

## 项目结构

```
├── MetaPINN_Reproduction.ipynb   # 主代码文件（Jupyter Notebook）
├── 研究报告.md                    # 完整学术研究报告
├── 实验报告.md                    # 详细实验报告（消融实验、超参数分析）
├── README.md                     # 本文件
├── imgs/                         # 结果图片
│   ├── improved_result.png       # 综合分析面板图
│   ├── result.png                # 中间结果
│   └── ...
└── Psaros 等 - 2022 - Meta-learning PINN loss functions.pdf  # 原始论文
```

## 复现实验说明

本项目提供两条复现路径：

1. `MetaPINN_Reproduction.ipynb`：主复现实验，包含标量权重/LAL 路线、元训练、元测试和综合可视化。
2. `meta_pinn_ffn_script.py`：FFN 损失函数参数化实验脚本，复现 FFN 损失网络与 MSE/L1/Cauchy 基线对比。

建议优先运行 Notebook；若只需要复现 FFN 对比结果，再运行 Python 脚本。

### 运行 Notebook 主实验

1. 安装依赖后，打开 Jupyter Notebook：

```bash
jupyter notebook MetaPINN_Reproduction.ipynb
```

2. 按顺序运行所有 Cell（约需 5-10 分钟，取决于硬件）：

| 部分 | Cell 范围 | 内容 | 预计耗时 |
|------|-----------|------|----------|
| 环境准备 | 0-3 | 导入库、定义参考解 | < 1 min |
| 数据生成 | 4-9 | 多任务数据生成、参考解验证 | ~1 min |
| 模型定义 | 10-15 | PINN 模型、功能式损失、MAML 框架 | < 1 min |
| 元训练 | 16-18 | First-Order MAML 训练（200 轮 × 5 任务） | ~3-5 min |
| 元测试 | 19-21 | 在新任务上对比元学习 vs 均匀权重 | ~1-2 min |
| 可视化 | 22-23 | 6 子图综合分析面板 | < 1 min |

3. 运行完成后检查以下输出：

| 输出 | 说明 |
|------|------|
| Notebook 单元输出 | 参考解验证、元训练损失、元测试 L2 误差、数据损失和 PDE 残差 |
| `imgs/improved_result.png` | 主实验综合结果图 |
| `imgs/Figure_1.png`、`imgs/Figure_2.png` | 中间实验图或报告引用图 |

### 运行 FFN 损失实验

如果需要复现 FFN 损失网络结果，运行：

```bash
python meta_pinn_ffn_script.py
```

该脚本会执行以下流程：

1. 生成 5 个训练任务和 1 个未见测试任务。
2. 初始化 `[2, 40, 40, 1]` 无偏置 FFN 损失网络。
3. 先用合成数据将 FFN 预训练为近似 MSE。
4. 执行 500 轮 First-Order MAML 元训练。
5. 在测试任务上分别训练 FFN、MSE、L1、Cauchy 四种 PINN。
6. 输出对比表，并保存 `imgs/ffn_result.png`。

预计耗时取决于 CPU/GPU 和 MindSpore 后端配置，CPU 环境通常需要数分钟到十余分钟。脚本使用 `ms.PYNATIVE_MODE`，便于调试，但速度会慢于图模式。

### 复现检查标准

一次完整复现应至少得到以下结果：

| 检查项 | 预期现象 |
|------|----------|
| 参考解验证 | 初始条件误差约为 `1e-16`，边界条件误差接近 0 |
| 元训练损失 | 随迭代整体下降，允许小幅随机波动 |
| 元测试任务 | 测试黏度为 `nu = 0.012 / pi ≈ 0.003820` |
| 主图输出 | 成功生成 `imgs/improved_result.png` |
| FFN 图输出 | 成功生成 `imgs/ffn_result.png` |

由于神经网络初始化、随机采样点和 MindSpore 后端存在随机性，最终数值可能与报告表格略有差异。判断复现是否成功时，应优先看趋势是否一致：元学习损失通常会降低 PDE 残差，但整体 L2 误差在当前设置下不一定优于 MSE 基线。

### 常见问题

| 问题 | 处理方式 |
|------|----------|
| `ModuleNotFoundError: mindspore` | 确认当前 Python 环境已安装 MindSpore，并且 Notebook kernel 指向同一环境 |
| `ModuleNotFoundError: scipy` | 运行 `pip install scipy` |
| CUDA 相关 warning | 本项目可在 CPU 上运行；若使用 CPU，可忽略 CUDA 路径 warning |
| 运行时间过长 | 减小 `meta_iters`、`epochs`、`n_f` 或先只运行 Notebook 主实验 |
| 输出数值和 README 不完全一致 | 检查随机种子、MindSpore 版本和是否按顺序运行全部 Cell |

## 实验设置

### 任务分布

| 类型 | 粘度系数 $\nu$ 取值 | 数量 |
|------|---------------------|------|
| **元训练** | $\{0.005/\pi, 0.008/\pi, 0.01/\pi, 0.015/\pi, 0.02/\pi\}$ | 5 |
| **元测试** | $0.012/\pi$（未见过的中间值） | 1 |

### 模型架构

- 网络结构：$[2, 50, 50, 50, 50, 1]$（4 隐藏层，Tanh 激活）
- 优化器：内循环 SGD (lr=$10^{-3}$, K=5)，外循环 Adam (lr=$10^{-3}$)
- 损失参数化：**LAL** (Learned Adaptive Loss)，4 参数 $(\alpha_u, c_u, \alpha_f, c_f)$；**FFN** (Feed-Forward Network)，1720 参数
- LAL 初始化：$\alpha=2.01, c=1/\sqrt{2}$（近似 MSE）；FFN 预训练 200 步 Adam 逼近 MSE

### 参考解

使用 **Method of Lines + 迎风格式 + RK45 自适应步长**生成 Burgers 方程精确参考解，自动满足稳定性约束。

## 实验结果

### LAL 元训练

在 5 个训练任务上元训练 1000 轮（T=1, Adam 外循环），LAL 参数演化：
- $\alpha_u$: 2.01 → 1.77（趋向 L1-like），$c_u$: 0.71 → 1.40
- $\alpha_f$: 2.01 → 1.98（保持 MSE-like），$c_f$: 0.71 → 1.43
- 元损失从 0.287 下降至 0.072

### FFN 元训练

在 5 个训练任务上元训练 500 轮（T=1, Adam 外循环 lr=1e-4），FFN 权重演化：
- |W1|: 0.0477 → 0.0437, |W2|: 0.0301 → 0.0285, |W3|: 0.0675 → 0.0706
- 元损失从 1.7855 下降至 1.4265（含 Eq.28 正则化，s=0.01, coef=0.01）
- FFN 预训练 200 步 Adam 逼近 MSE（MSE 从 13.09 降至 1.21）

### 元测试（ν=0.012/π, 未见任务）

| 方法 | L2 相对误差 | 数据损失 (MSE) | PDE 残差损失 |
|------|-----------|---------------|-------------|
| **FFN (ours)** | 0.999 | 2.56×10⁻¹ | **2.17×10⁻⁵** |
| **LAL (ours)** | 0.710 | 9.64×10⁻² | 3.67×10⁻² |
| MSE | **0.532** | **4.23×10⁻²** | 8.57×10⁻² |
| L1 | 0.794 | 1.49×10⁻¹ | 4.14×10⁻³ |
| Cauchy | 0.647 | 8.13×10⁻² | 3.37×10⁻² |

### 标量权重基线（前版本）

| 指标 | 元学习权重 (λ_u=0.772, λ_f=1.0) | 均匀权重 (λ_u=λ_f=1.0) | 变化 |
|------|-----------|---------|------|
| L2 相对误差 | 0.691 | 0.544 | +27% |
| 数据损失 (BC/IC) | 9.35×10⁻² | 4.80×10⁻² | +95% |
| PDE 残差损失 | 3.22×10⁻² | 7.24×10⁻² | **-55%** |

**分析**：三种参数化方法（标量权重、LAL、FFN）均一致地学习到降低 PDE 残差的策略：
- FFN 表现最极端：PDE 残差仅 2.17×10⁻⁵（vs MSE 的 8.57×10⁻²），降低了 99.97%，但数据拟合完全退化（L2=0.999）
- LAL 降低 PDE 49%，标量权重降低 55%
- 三者均未在整体 L2 精度上超越 MSE 基线

FFN 的 1720 个参数给予了损失函数极大的灵活性，导致其"过度学习"了偏重 PDE 约束的策略——说明 Eq.28 正则化（s=0.01）在当前设置下约束力不足。L1 损失也表现出类似的极端偏重 PDE 倾向（L2=0.794）。Cauchy 损失（L2=0.647）是最接近 MSE（0.532）的替代方案。

这表明在 Burgers 方程上，标准 MSE 损失已经接近最优——元学习发现的损失函数倾向于过度强调 PDE 约束，需要更强的正则化或在验证损失中更好地平衡数据与物理项。

> 数值因随机种子略有波动，详见 Notebook 运行输出和 `imgs/improved_result.png`。

## 主要改进（相对原始简化版）

1. **正确的 PDE 实现**：加入非线性对流项 $u \cdot u_x$，求解完整 Burgers 方程
2. **LAL + FFN 损失参数化**：实现论文 Section 3.4.1.1 自适应损失（4 参数）和 Section 3.4.1.2 FFN 损失网络（1720 参数，Eq.28 正则化）
3. **多基线对比**：MSE、L1、Cauchy 三种基线损失函数
4. **论文一致的元训练**：Adam 外循环，T=1 任务采样
5. **数值稳定的参考解**：Method of Lines + 迎风格式 + RK45 自适应步长
6. **梯度函数预定义**：避免重复创建 `ms.grad`，提升性能

## 参考文献

1. Psaros, A. F., et al. (2022). *Meta-learning PINN loss functions.* Journal of Computational Physics, 458, 111121.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.
3. Finn, C., Abbeel, P., & Levine, S. (2017). *Model-agnostic meta-learning for fast adaptation of deep networks.* ICML 2017.
