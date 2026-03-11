# GLP 训练评估指标详解

> 所有指标均在**激活空间**（ESM2-650M Layer 17, dim=1280）直接计算，不经过下游任务。
> Eval buffer: 从训练数据最后一个 part 文件取 8192 个样本作为"真实分布"参考。

---

## 1. NLL (Negative Log-Likelihood)

### 意义

NLL 是衡量生成模型质量的**金标准**。它直接回答：GLP 学到的概率分布 $p_\theta(x)$ 对真实激活数据 $x$ 赋予了多高的概率密度？

- **NLL 越低** → 模型认为真实数据越"自然"，分布学得越好
- 对比基线：如果 NLL ≈ 标准高斯的 NLL（$\frac{d}{2}\log(2\pi) \approx 918$），说明模型没学到任何东西

### 数学公式

Flow matching 学习的是一个从数据 $x_0$ 到噪声 $x_1 \sim \mathcal{N}(0,I)$ 的速度场 $v_\theta(z_t, t)$。通过 **continuous normalizing flow (CNF)** 的变量替换公式：

$$
\log p_\theta(x_0) = \log p_1(z_1) - \int_0^1 \text{Tr}\left(\frac{\partial v_\theta}{\partial z_t}\right) dt
$$

其中：
- $z_1$ 是从 $x_0$ 沿 ODE $\frac{dz}{dt} = v_\theta(z_t, t)$ 积分到 $t=1$ 的结果
- $\log p_1(z_1) = -\frac{1}{2}\|z_1\|^2 - \frac{d}{2}\log(2\pi)$（标准高斯对数概率）
- $\text{Tr}\left(\frac{\partial v_\theta}{\partial z_t}\right)$ 是 Jacobian 的迹

**Jacobian 迹**无法直接计算（需要 $d=1280$ 次反向传播），所以用 **Hutchinson trace estimator** 近似：

$$
\text{Tr}(J) \approx \frac{1}{K}\sum_{k=1}^K \epsilon_k^T J \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I)
$$

其中 $\epsilon^T J \epsilon$ 可以通过一次 vector-Jacobian product (VJP) 高效计算。

最终 NLL：

$$
\text{NLL} = -\log p_\theta(x_0) = \frac{1}{2}\|z_1\|^2 + \frac{d}{2}\log(2\pi) + \int_0^1 \text{Tr}(J) \, dt
$$

### 代码

```python
def compute_nll_hutchinson(model, latents, num_timesteps=100, n_hutchinson=5):
    dt = 1.0 / num_timesteps
    z_t = latents.clone()                          # (N, 1, dim)
    log_det_sum = torch.zeros(batch_size, device=device)

    for step in range(num_timesteps):              # 数值积分 ∫₀¹ dt
        t = step * dt

        # 构造 timestep
        t_tensor = torch.full((batch_size,), t, device=device)
        indices = (t_tensor * len(model.scheduler.timesteps)).long()
        timesteps = model.scheduler.timesteps.to(device)[indices][:, None, None]

        # Hutchinson trace estimator
        trace_est = torch.zeros(batch_size, device=device)
        for _ in range(n_hutchinson):              # K 次随机探测
            eps = torch.randn_like(z_t)            # ε ~ N(0, I)
            z_t_input = z_t.detach().requires_grad_(True)

            with torch.enable_grad():
                v = model.denoiser(latents=z_t_input, timesteps=timesteps)
                # VJP: 计算 εᵀ (∂v/∂z)
                vjp = torch.autograd.grad(v, z_t_input, grad_outputs=eps)[0]

            # εᵀ J ε = (vjp * eps).sum()
            trace_est += (vjp * eps).sum(dim=-1).sum(dim=-2)

        log_det_sum += (trace_est / n_hutchinson) * dt

        # Euler 积分: z_{t+dt} = z_t + v(z_t, t) * dt
        with torch.no_grad():
            v = model.denoiser(latents=z_t, timesteps=timesteps)
            z_t = z_t + v * dt

    # z_1 处的标准高斯对数概率
    log_pz1 = -0.5 * z_t.pow(2).sum(dim=-1).sum(dim=-2) - 0.5 * dim * log(2π)

    # log p(x) = log p(z_1) - ∫ Tr(J) dt
    nll = -(log_pz1 - log_det_sum)                 # (N,)
    return nll.mean(), nll.std()
```

### 计算参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_timesteps` | 50 | ODE 积分步数，越大越精确，越慢 |
| `n_hutchinson` | 3 | Hutchinson 随机向量数，越大方差越小 |
| 评估样本数 | 512 | 从 eval buffer 取前 512 个 |

---

## 2. FID (Fréchet Inception Distance)

### 意义

FID 衡量**生成分布与真实分布在统计特征（均值 + 协方差）上的距离**。

- 原始 FID 用于图像，通过 Inception 网络提取特征后计算
- 我们直接在 1280 维激活空间计算（先 PCA 降到 128 维避免高维协方差矩阵的数值问题）
- **FID 越低** → 生成分布越接近真实分布（0 = 完全匹配）
- FID 同时捕捉 **mode collapse**（均值偏移）和 **diversity 不足**（协方差不匹配）

### 数学公式

给定真实分布 $(\mu_r, \Sigma_r)$ 和生成分布 $(\mu_f, \Sigma_f)$：

$$
\text{FID} = \|\mu_r - \mu_f\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_f - 2(\Sigma_r \Sigma_f)^{1/2}\right)
$$

其中：
- $\|\mu_r - \mu_f\|^2$：均值偏移项，衡量生成样本的"中心"是否正确
- $\text{Tr}(\Sigma_r + \Sigma_f - 2\sqrt{\Sigma_r\Sigma_f})$：协方差匹配项，衡量生成样本的"分散程度"是否正确
- $(\Sigma_r \Sigma_f)^{1/2}$ 通过特征值分解计算：$\text{Tr}(\sqrt{\Sigma_r\Sigma_f}) = \sum_i \sqrt{\lambda_i}$，其中 $\lambda_i$ 是 $\Sigma_r\Sigma_f$ 的特征值

### 代码

```python
def compute_fid(real, fake):
    """real, fake: (N, dim) tensors"""
    mu_r, mu_f = real.mean(dim=0), fake.mean(dim=0)

    # 协方差矩阵
    cov_r = ((real - mu_r).T @ (real - mu_r)) / (N - 1)
    cov_f = ((fake - mu_f).T @ (fake - mu_f)) / (N - 1)

    # 均值项
    mean_term = (mu_r - mu_f).dot(mu_r - mu_f)

    # 协方差项: Tr(Σ_r + Σ_f - 2√(Σ_r·Σ_f))
    product = cov_r @ cov_f
    eigvals = torch.linalg.eigvalsh(product)
    eigvals = torch.clamp(eigvals, min=0)          # 数值修正负特征值
    sqrt_trace = eigvals.sqrt().sum()

    trace_term = cov_r.trace() + cov_f.trace() - 2 * sqrt_trace
    return mean_term + trace_term
```

降维预处理（在 `run_evaluation` 中）：

```python
# PCA 降维到 128 维，避免 1280×1280 协方差矩阵的数值问题
combined = torch.cat([real_flat, fake_flat], dim=0)
U, S, Vh = torch.linalg.svd(combined - combined.mean(0), full_matrices=False)
proj = Vh[:128]                                     # 取前 128 个主成分
real_proj = (real_flat - mean) @ proj.T             # (N, 128)
fake_proj = (fake_flat - mean) @ proj.T
fid = compute_fid(real_proj, fake_proj)
```

---

## 3. MMD (Maximum Mean Discrepancy)

### 意义

MMD 是一种**非参数化**的分布距离度量。它不假设分布是高斯的（FID 假设高斯），而是通过 kernel 函数在再生核希尔伯特空间（RKHS）中比较两个分布的嵌入。

- **MMD 越低** → 两个分布越接近（0 = 完全匹配）
- MMD 比 FID 更灵活：能捕捉高阶矩的差异，不只是 mean + covariance
- 使用 RBF (Gaussian) kernel，sigma 自适应设为真实数据距离的中位数（median heuristic）

### 数学公式

给定核函数 $k(x, y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$（RBF kernel）：

$$
\text{MMD}^2 = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]
$$

其中 $x, x'$ 独立采样自真实分布，$y, y'$ 独立采样自生成分布。

展开为样本估计：

$$
\text{MMD}^2 \approx \frac{1}{N^2}\sum_{i,j} k(x_i, x_j) + \frac{1}{M^2}\sum_{i,j} k(y_i, y_j) - \frac{2}{NM}\sum_{i,j} k(x_i, y_j)
$$

**Sigma 选择**（median heuristic）：

$$
\sigma = \text{median}\left(\{\|x_i - x_j\| : i \neq j\}\right)
$$

这确保 kernel 不会太窄（只看局部）或太宽（忽略所有差异）。

### 代码

```python
def compute_mmd(x, y, sigma=1.0):
    """x, y: (N, dim) tensors"""
    def rbf_kernel(a, b, sigma):
        dist = torch.cdist(a, b, p=2).pow(2)       # (N, M) 距离矩阵
        return torch.exp(-dist / (2 * sigma ** 2))  # RBF kernel

    xx = rbf_kernel(x, x, sigma).mean()             # E[k(x, x')]
    yy = rbf_kernel(y, y, sigma).mean()             # E[k(y, y')]
    xy = rbf_kernel(x, y, sigma).mean()             # E[k(x, y)]
    return (xx + yy - 2 * xy).item()
```

自适应 sigma（在 `run_evaluation` 中）：

```python
# Median heuristic: sigma = 真实数据间距离的中位数
dists = torch.cdist(real_proj[:500], real_proj[:500], p=2)
sigma = dists.median().item()
mmd = compute_mmd(real_proj[:2048], fake_proj[:2048], sigma=sigma)
```

---

## 4. 辅助指标

### gen_mean_err — 均值偏差

$$
\text{gen\_mean\_err} = \frac{1}{d}\sum_{j=1}^d \left| \bar{x}_j^{\text{gen}} - \bar{x}_j^{\text{real}} \right|
$$

生成样本与真实样本在每个维度上均值的平均绝对偏差。反映模型是否学到了正确的数据"中心"。

### gen_std_err — 标准差偏差

$$
\text{gen\_std\_err} = \frac{1}{d}\sum_{j=1}^d \left| \text{std}_j^{\text{gen}} - \text{std}_j^{\text{real}} \right|
$$

生成样本与真实样本在每个维度上标准差的平均绝对偏差。反映模型是否学到了正确的数据"分散程度"。

---

## 指标对比总结

| 指标 | 类型 | 假设 | 捕捉什么 | 计算代价 | 理想值 |
|------|------|------|----------|----------|--------|
| NLL | 似然 | 连续归一化流 | 模型对真实数据赋予的概率密度 | 高（需 autograd） | 越低越好 |
| FID | 参数化距离 | 高斯分布 | 均值 + 协方差匹配 | 低 | 0 |
| MMD | 非参数距离 | 无（依赖 kernel） | 所有阶矩的匹配 | 中 | 0 |
| gen_mean_err | 统计量 | 无 | 一阶矩偏差 | 极低 | 0 |
| gen_std_err | 统计量 | 无 | 二阶矩偏差 | 极低 | 0 |

### 训练过程中期望看到的趋势

1. **train/loss**（MSE）：快速下降后趋于平稳
2. **NLL**：随训练下降，反映模型对数据的拟合程度
3. **FID**：从高值快速下降，最终趋近 0
4. **MMD**：与 FID 趋势类似，但对高阶差异更敏感
5. **gen_mean_err / gen_std_err**：应快速趋近 0

如果 FID/MMD 下降但 NLL 上升，可能意味着模型出现了 mode collapse（只学了部分模式但学得很像）。
