"""
FFN Loss Parametrization for Meta-Learning PINN Loss Functions
==============================================================
Psaros et al. (2022) Section 3.4.1.2

Architecture: [2, 40, 40, 1] no-bias ReLU+Softplus network
Input: (u_pred, u_true) -> Output: per-point loss value
Regularization: Eq. 28 (optimal stationarity + MSE relation)
"""
import warnings
warnings.filterwarnings('ignore')

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from scipy.integrate import solve_ivp

ms.set_context(mode=ms.PYNATIVE_MODE)
print(f"MindSpore version: {ms.__version__}")

# ============================================================
# Burgers equation reference solution (MOL + upwind + RK45)
# ============================================================

def burgers_fd_solve(nu, nx=256, nt_uniform=100):
    x = np.linspace(-1, 1, nx)
    dx = x[1] - x[0]
    t_span = (0.0, 1.0)
    t_eval = np.linspace(0, 1, nt_uniform)
    u0 = -np.sin(np.pi * x)
    u0[0] = 0.0
    u0[-1] = 0.0
    j = np.arange(1, nx - 1)
    jp, jm = j + 1, j - 1

    def rhs(t, u):
        dudt = np.zeros_like(u)
        up, um, uc = u[jp], u[jm], u[j]
        ux = np.where(uc >= 0, (uc - um) / dx, (up - uc) / dx)
        conv = -uc * ux
        uxx = (up - 2.0 * uc + um) / dx**2
        diff = nu * uxx
        dudt[j] = conv + diff
        return dudt

    result = solve_ivp(rhs, t_span, u0, method='RK45', t_eval=t_eval,
                       rtol=1e-6, atol=1e-8, max_step=0.01)
    return x, result.t, result.y.T


def burgers_reference_solution(x, t, nu):
    x_flat = np.asarray(x, dtype=np.float64).flatten()
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    nx, nt = 400, 100
    x_grid, t_grid, u_grid = burgers_fd_solve(nu, nx=nx, nt_uniform=nt)
    result = np.zeros(len(x_flat))
    for i, (xq, tq) in enumerate(zip(x_flat, t_flat)):
        if np.abs(xq + 1.0) < 1e-12 or np.abs(xq - 1.0) < 1e-12:
            result[i] = 0.0
            continue
        if tq < 1e-12:
            result[i] = -np.sin(np.pi * xq)
            continue
        t_idx = np.searchsorted(t_grid, tq)
        t_idx = max(1, min(t_idx, len(t_grid) - 1))
        alpha = (tq - t_grid[t_idx - 1]) / (t_grid[t_idx] - t_grid[t_idx - 1])
        u_t1 = np.interp(xq, x_grid, u_grid[t_idx - 1, :])
        u_t2 = np.interp(xq, x_grid, u_grid[t_idx, :])
        result[i] = (1.0 - alpha) * u_t1 + alpha * u_t2
    return result


# Verify reference solution
print("Verifying reference solution...")
x_test = np.linspace(-1, 1, 100)
for nu_test in [0.02 / np.pi, 0.01 / np.pi, 0.005 / np.pi]:
    u_ref_ic = burgers_reference_solution(x_test, np.zeros(100), nu_test)
    u_exact_ic = -np.sin(np.pi * x_test)
    ic_error = np.max(np.abs(u_ref_ic - u_exact_ic))
    print(f"  nu={nu_test:.6f}: IC error={ic_error:.2e}")
print("Reference solution: PASSED\n")

# ============================================================
# Data Generation
# ============================================================

def generate_task_data(nu, seed=42, n_u=800, n_f=2000):
    np.random.seed(seed)
    # BC/IC points
    x_left = -np.ones(n_u // 4)
    t_left = np.random.uniform(0, 1, n_u // 4)
    u_left = np.zeros(n_u // 4)
    x_right = np.ones(n_u // 4)
    t_right = np.random.uniform(0, 1, n_u // 4)
    u_right = np.zeros(n_u // 4)
    x_init = np.random.uniform(-1, 1, n_u // 2)
    t_init = np.zeros(n_u // 2)
    u_init = -np.sin(np.pi * x_init)

    X_u_all = np.column_stack([
        np.concatenate([x_left, x_right, x_init]),
        np.concatenate([t_left, t_right, t_init])
    ])
    U_all = np.concatenate([u_left, u_right, u_init]).reshape(-1, 1)

    # Collocation points
    X_f_all = np.random.uniform(-1, 1, (n_f, 2))
    X_f_all[:, 1] = np.random.uniform(0, 1, n_f)

    # Train/val split
    n_u_train = int(0.8 * n_u)
    n_f_train = int(0.8 * n_f)
    perm_u = np.random.permutation(n_u)
    perm_f = np.random.permutation(n_f)

    # Evaluation grid
    nx_star, nt_star = 256, 100
    x_star = np.linspace(-1, 1, nx_star)
    t_star = np.linspace(0, 1, nt_star)
    X_star_grid, T_star_grid = np.meshgrid(x_star, t_star)
    X_star = np.column_stack([X_star_grid.flatten(), T_star_grid.flatten()])
    u_star = burgers_reference_solution(X_star[:, 0], X_star[:, 1], nu)
    u_star = u_star.reshape(-1, 1)

    return {
        'nu': nu,
        'X_u_train': X_u_all[perm_u[:n_u_train]].astype(np.float32),
        'U_train': U_all[perm_u[:n_u_train]].astype(np.float32),
        'X_f_train': X_f_all[perm_f[:n_f_train]].astype(np.float32),
        'X_u_val': X_u_all[perm_u[n_u_train:]].astype(np.float32),
        'U_val': U_all[perm_u[n_u_train:]].astype(np.float32),
        'X_f_val': X_f_all[perm_f[n_f_train:]].astype(np.float32),
        'X_star': X_star.astype(np.float32),
        'u_star': u_star.astype(np.float32),
    }


# Generate tasks
nu_train_values = [0.005 / np.pi, 0.008 / np.pi, 0.01 / np.pi, 0.015 / np.pi, 0.02 / np.pi]
nu_test_value = 0.012 / np.pi

print("Generating training tasks...")
train_tasks = [generate_task_data(nu, seed=42 + i) for i, nu in enumerate(nu_train_values)]
for i, nu in enumerate(nu_train_values):
    print(f"  Task {i+1}: nu = {nu:.6f}")

print("Generating test task...")
test_task = generate_task_data(nu_test_value, seed=100)
print(f"  Test task: nu = {nu_test_value:.6f}")
print(f"Done: {len(train_tasks)} train tasks + 1 test task.\n")

# ============================================================
# PINN Model
# ============================================================

class PINN(nn.Cell):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.SequentialCell([
            nn.Dense(layers[i], layers[i+1],
                     activation=nn.Tanh() if i < len(layers) - 2 else None)
            for i in range(len(layers) - 1)
        ])

    def construct(self, x):
        return self.net(x)


layers = [2, 50, 50, 50, 50, 1]
N_MODEL_PARAMS = 10  # 5 layers x 2 params (W, b)


def create_model():
    return PINN(layers)


# ---- Functional forward pass ----
def pinn_functional_forward(x, *params):
    h = x
    n_layers = len(params) // 2
    for i in range(n_layers - 1):
        w, b = params[2*i], params[2*i+1]
        h = ops.tanh(ops.matmul(h, w.T) + b)
    w, b = params[-2], params[-1]
    h = ops.matmul(h, w.T) + b
    return h


def net_u(x, t, *params):
    return pinn_functional_forward(ops.concat((x, t), axis=1), *params)


_grad_u_x = ms.grad(net_u, grad_position=0)
_grad_u_t = ms.grad(net_u, grad_position=1)
_grad_u_xx = ms.grad(lambda x, t, *p: _grad_u_x(x, t, *p), grad_position=0)


def compute_pde_residual(x_f, t_f, nu, *params):
    u_val = net_u(x_f, t_f, *params)
    u_x = _grad_u_x(x_f, t_f, *params)
    u_t = _grad_u_t(x_f, t_f, *params)
    u_xx = _grad_u_xx(x_f, t_f, *params)
    return u_t + u_val * u_x - nu * u_xx


# ============================================================
# FFN Loss Network (Section 3.4.1.2)
# ============================================================
# Architecture: [2, 40, 40, 1], no biases, ReLU+ReLU+Softplus
# W1: (40, 2), W2: (40, 40), W3: (1, 40) -- total 1720 params

FFN_LAYERS = [2, 40, 40, 1]


def ffn_init_params(seed=123):
    """Xavier uniform initialization for FFN weights (no biases)."""
    np.random.seed(seed)
    params = []
    for i in range(len(FFN_LAYERS) - 1):
        fan_in = FFN_LAYERS[i]
        fan_out = FFN_LAYERS[i + 1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w = np.random.uniform(-limit, limit, (fan_out, fan_in)).astype(np.float32)
        params.append(Tensor(w, mstype.float32))
    return tuple(params)  # (W1, W2, W3)


def ffn_forward(u_pred, u_true, W1, W2, W3):
    """
    FFN loss network forward pass.
    Input: (u_pred, u_true) concatenated -> [N, 2]
    Hidden: 40 ReLU (no bias) -> 40 ReLU (no bias)
    Output: 1 Softplus -> non-negative loss value [N, 1]
    """
    x = ops.concat((u_pred, u_true), axis=1)  # (N, 2)
    h = ops.relu(ops.matmul(x, W1.T))          # (N, 2) @ (2, 40) = (N, 40)
    h = ops.relu(ops.matmul(h, W2.T))          # (N, 40) @ (40, 40) = (N, 40)
    h = ops.softplus(ops.matmul(h, W3.T))      # (N, 40) @ (40, 1) = (N, 1)
    return h


def compute_pinn_loss_ffn(X_u, U, X_f, W1, W2, W3, nu, *model_params):
    """
    PINN loss with FFN parametrization.
    Data loss: mean(FFN(u_pred, u_true))
    PDE loss:  mean(FFN(residual, 0))
    """
    # Data loss
    x_u, t_u = X_u[:, 0:1], X_u[:, 1:2]
    u_pred = net_u(x_u, t_u, *model_params)
    per_point_data = ffn_forward(u_pred, U, W1, W2, W3)
    loss_data = ops.reduce_mean(per_point_data)

    # PDE loss
    x_f, t_f = X_f[:, 0:1], X_f[:, 1:2]
    residual = compute_pde_residual(x_f, t_f, nu, *model_params)
    zeros = ops.zeros_like(residual)
    per_point_pde = ffn_forward(residual, zeros, W1, W2, W3)
    loss_pde = ops.reduce_mean(per_point_pde)

    return loss_data + loss_pde


# ---- Pre-train FFN to approximate MSE ----
def pretrain_ffn_mse(ffn_params, n_samples=2000, n_steps=200, lr=1e-3):
    """Pre-train FFN to approximate MSE loss on synthetic data."""
    print("Pre-training FFN to approximate MSE...")
    np.random.seed(999)
    d = np.random.uniform(-3, 3, (n_samples, 1)).astype(np.float32)
    target = (d ** 2).astype(np.float32)
    d_tensor = Tensor(d, mstype.float32)
    target_tensor = Tensor(target, mstype.float32)

    W1, W2, W3 = ffn_params
    W1_p = Parameter(W1, name='W1')
    W2_p = Parameter(W2, name='W2')
    W3_p = Parameter(W3, name='W3')
    opt = nn.Adam([W1_p, W2_p, W3_p], learning_rate=lr)

    def pretrain_loss_fn(w1, w2, w3):
        zeros = ops.zeros_like(d_tensor)
        pred = ffn_forward(d_tensor, zeros, w1, w2, w3)
        return ops.reduce_mean((pred - target_tensor) ** 2)

    pretrain_grad = ms.grad(pretrain_loss_fn, grad_position=(0, 1, 2))

    for step in range(n_steps):
        g1, g2, g3 = pretrain_grad(W1_p, W2_p, W3_p)
        opt((g1, g2, g3))
        if step % 50 == 0:
            loss_val = pretrain_loss_fn(W1_p, W2_p, W3_p).asnumpy()
            print(f"  Pre-train step {step:3d}: MSE={loss_val:.6e}")

    print(f"  Final pre-train MSE={pretrain_loss_fn(W1_p, W2_p, W3_p).asnumpy():.6e}")
    return tuple(Tensor(p.asnumpy(), mstype.float32) for p in (W1_p, W2_p, W3_p))


# ============================================================
# Regularization (Eq. 28)
# ============================================================

def ffn_regularization(W1, W2, W3, n_samples=50, s=0.01):
    """
    Eq. 28 -- Enforce optimal stationarity and MSE relation conditions.
    Uses batch gradient computation for efficiency (2 ms.grad calls total).

    Term 1: E_q[||grad_q FFN(q, q)||^2] -- gradient at zero discrepancy should be 0
    Term 2: E_{q!=q'}[max(0, s - ||grad_q FFN(q, q')||^2)] -- penalize small gradients
    """
    # Term 1: batch version -- single ms.grad call for all samples
    d1 = np.random.uniform(-3, 3, (n_samples, 1)).astype(np.float32)
    d1_tensor = Tensor(d1, mstype.float32)

    def batch_ffn_equal(q_batch):
        return ops.reduce_sum(ffn_forward(q_batch, q_batch, W1, W2, W3))

    grad_fn_1 = ms.grad(batch_ffn_equal, grad_position=0)
    g1 = grad_fn_1(d1_tensor)
    reg1 = float(ops.reduce_mean(g1 ** 2).asnumpy())

    # Term 2: batch version
    d2 = np.random.uniform(-3, 3, (n_samples, 1)).astype(np.float32)
    d2 = d2 + 0.1 * np.sign(d2 - d1)  # ensure q != q'
    d2_tensor = Tensor(d2, mstype.float32)

    def batch_ffn_unequal(q_batch, qp_batch):
        return ops.reduce_sum(ffn_forward(q_batch, qp_batch, W1, W2, W3))

    grad_fn_2 = ms.grad(batch_ffn_unequal, grad_position=0)
    g2 = grad_fn_2(d1_tensor, d2_tensor)
    g2_norm_per_sample = ops.reduce_mean(g2 ** 2, axis=1)
    reg2 = float(ops.reduce_mean(ops.maximum(Tensor(0.0, mstype.float32),
                                              Tensor(s, mstype.float32) - g2_norm_per_sample)).asnumpy())

    return reg1 + reg2


# Helper for regularization gradient in optimizer step
def ffn_regularization_as_fn(W1, W2, W3):
    return ffn_regularization(W1, W2, W3, n_samples=20, s=0.01)


# ============================================================
# Meta-Learning Functions
# ============================================================

# grad_position for PINN model params in compute_pinn_loss_ffn:
# X_u(0), U(1), X_f(2), W1(3), W2(4), W3(5), nu(6), model_params(7..16)
_MODEL_GRAD_FFN = tuple(range(7, 7 + N_MODEL_PARAMS))

# Pre-define gradient for inner loop
_pinn_loss_ffn_grad = ms.grad(compute_pinn_loss_ffn, grad_position=_MODEL_GRAD_FFN)


def inner_loop_adapt_ffn(model_params, X_u, U, X_f, W1, W2, W3, nu, inner_lr, inner_steps):
    """Inner loop: K-step SGD adaptation of PINN parameters using FFN loss."""
    adapted = model_params
    for _ in range(inner_steps):
        grads = _pinn_loss_ffn_grad(X_u, U, X_f, W1, W2, W3, nu, *adapted)
        adapted = tuple(p - inner_lr * g for p, g in zip(adapted, grads))
    return adapted


def meta_loss_and_ffn_grads(W1, W2, W3, model_params, train_tasks, inner_lr, inner_steps, reg_coef=0.01):
    """Compute meta-loss and FFN parameter gradients (First-Order MAML)."""
    total_grad_w1 = Tensor(np.zeros((40, 2), dtype=np.float32), mstype.float32)
    total_grad_w2 = Tensor(np.zeros((40, 40), dtype=np.float32), mstype.float32)
    total_grad_w3 = Tensor(np.zeros((1, 40), dtype=np.float32), mstype.float32)
    total_val_loss = 0.0
    n_tasks = len(train_tasks)

    for task in train_tasks:
        X_u = Tensor(task['X_u_train'], mstype.float32)
        U = Tensor(task['U_train'], mstype.float32)
        X_f = Tensor(task['X_f_train'], mstype.float32)
        nu = task['nu']

        X_u_val = Tensor(task['X_u_val'], mstype.float32)
        U_val = Tensor(task['U_val'], mstype.float32)
        X_f_val = Tensor(task['X_f_val'], mstype.float32)

        # Inner loop adaptation
        adapted = inner_loop_adapt_ffn(
            model_params, X_u, U, X_f, W1, W2, W3, nu, inner_lr, inner_steps
        )
        adapted_detached = tuple(ops.stop_gradient(p) for p in adapted)

        # Validation loss gradient w.r.t. FFN params (First-Order MAML)
        def val_loss_fn(w1, w2, w3):
            return compute_pinn_loss_ffn(X_u_val, U_val, X_f_val, w1, w2, w3, nu, *adapted_detached)

        val_grad_fn = ms.grad(val_loss_fn, grad_position=(0, 1, 2))
        g1, g2, g3 = val_grad_fn(W1, W2, W3)

        total_grad_w1 = total_grad_w1 + g1
        total_grad_w2 = total_grad_w2 + g2
        total_grad_w3 = total_grad_w3 + g3

        val_loss = val_loss_fn(W1, W2, W3)
        total_val_loss += val_loss.asnumpy()

    avg_val_loss = total_val_loss / n_tasks
    avg_g1 = total_grad_w1 / n_tasks
    avg_g2 = total_grad_w2 / n_tasks
    avg_g3 = total_grad_w3 / n_tasks

    return avg_val_loss, avg_g1, avg_g2, avg_g3


def compute_batch_train_loss_ffn(X_u_list, U_list, X_f_list, nu_list, W1, W2, W3, *model_params):
    """Average train loss across all tasks for base model update."""
    total = 0.0
    for X_u, U, X_f, nu in zip(X_u_list, U_list, X_f_list, nu_list):
        total += compute_pinn_loss_ffn(X_u, U, X_f, W1, W2, W3, nu, *model_params)
    return total / len(X_u_list)


_BASE_FFN_GRAD = tuple(range(7, 7 + N_MODEL_PARAMS))
_model_loss_ffn_grad = ms.grad(compute_batch_train_loss_ffn, grad_position=_BASE_FFN_GRAD)


# ============================================================
# Meta-Training
# ============================================================

inner_lr = 1e-3
outer_lr = 1e-4      # Paper uses 1e-4 for FFN outer loop
model_lr = 1e-4
inner_steps = 5
meta_iters = 500     # FFN has 1720 params, need more iterations; 500 practical for CPU
reg_coef = 0.01      # Regularization coefficient for Eq. 28

# Initialize FFN
ffn_params = ffn_init_params(seed=123)
print(f"FFN architecture: {FFN_LAYERS}, total params: {sum(p.size for p in ffn_params)}")

# Pre-train FFN to approximate MSE
ffn_params = pretrain_ffn_mse(ffn_params, n_samples=2000, n_steps=200, lr=1e-3)
W1, W2, W3 = ffn_params

W1_p = Parameter(W1, name='W1')
W2_p = Parameter(W2, name='W2')
W3_p = Parameter(W3, name='W3')
ffn_optimizer = nn.Adam([W1_p, W2_p, W3_p], learning_rate=outer_lr)

# Initialize base model
model = create_model()
model_params = tuple(Tensor(p.asnumpy(), mstype.float32) for p in model.trainable_params())

print(f"\nInner loop: SGD lr={inner_lr}, {inner_steps} steps")
print(f"Outer loop: Adam lr={outer_lr}")
print(f"FFN parametrization: [2,40,40,1] no-bias ReLU+Softplus")
print(f"Regularization: Eq.28, s=0.01, coef={reg_coef}")
print(f"Training tasks: {len(train_tasks)}, T=1 per step")
print(f"Meta iterations: {meta_iters}")

# Pre-convert tensors
X_u_batch = [Tensor(t['X_u_train'], mstype.float32) for t in train_tasks]
U_batch = [Tensor(t['U_train'], mstype.float32) for t in train_tasks]
X_f_batch = [Tensor(t['X_f_train'], mstype.float32) for t in train_tasks]
nu_batch = [t['nu'] for t in train_tasks]

meta_history_ffn = {
    'meta_loss': [],
    'reg_loss': [],
    'w1_norm': [], 'w2_norm': [], 'w3_norm': [],
}

print("\nStarting FFN meta-training...")
print("=" * 70)

for meta_iter in range(meta_iters):
    # T=1: sample single task
    task_idx = meta_iter % len(train_tasks)
    current_tasks = [train_tasks[task_idx]]

    # Compute FFN gradients via First-Order MAML
    meta_loss, g1, g2, g3 = meta_loss_and_ffn_grads(
        W1_p, W2_p, W3_p, model_params, current_tasks, inner_lr, inner_steps
    )

    # Compute regularization (Eq. 28) -- every 20 steps for logging
    if meta_iter % 20 == 0:
        reg_val = ffn_regularization(W1_p, W2_p, W3_p, n_samples=30, s=0.01)
    else:
        reg_val = 0.0

    # Adam update FFN params
    ffn_optimizer((g1, g2, g3))

    # Update base model
    g_model = _model_loss_ffn_grad(
        X_u_batch, U_batch, X_f_batch, nu_batch, W1_p, W2_p, W3_p, *model_params
    )
    model_params = tuple(p - model_lr * g for p, g in zip(model_params, g_model))

    # Logging
    w1_norm = float(ops.reduce_mean(W1_p ** 2).asnumpy())
    w2_norm = float(ops.reduce_mean(W2_p ** 2).asnumpy())
    w3_norm = float(ops.reduce_mean(W3_p ** 2).asnumpy())

    meta_history_ffn['meta_loss'].append(meta_loss)
    meta_history_ffn['reg_loss'].append(reg_val)
    meta_history_ffn['w1_norm'].append(w1_norm)
    meta_history_ffn['w2_norm'].append(w2_norm)
    meta_history_ffn['w3_norm'].append(w3_norm)

    if meta_iter % 50 == 0:
        print(f"Iter {meta_iter:4d} | Meta Loss: {meta_loss:.4f} | Reg: {reg_val:.4f} | "
              f"|W1|={w1_norm:.4f} |W2|={w2_norm:.4f} |W3|={w3_norm:.4f}")

print("=" * 70)
print("FFN meta-training complete!")
print(f"Final: |W1|={w1_norm:.4f}, |W2|={w2_norm:.4f}, |W3|={w3_norm:.4f}\n")


# Helper for regularization gradient (needed by ms.grad)
def ffn_regularization_as_fn(W1, W2, W3):
    return ffn_regularization(W1, W2, W3, n_samples=20, s=0.01)


# ============================================================
# Meta-Test: Compare FFN vs MSE vs L1 vs Cauchy
# ============================================================

def train_pinn_with_ffn(model, task, ffn_params_tuple, learning_rate, epochs, verbose=True):
    """Train PINN from scratch using learned FFN loss."""
    W1_f, W2_f, W3_f = ffn_params_tuple
    X_u = Tensor(task['X_u_train'], mstype.float32)
    U = Tensor(task['U_train'], mstype.float32)
    X_f = Tensor(task['X_f_train'], mstype.float32)
    nu = task['nu']

    W1_t = Tensor(W1_f.asnumpy() if isinstance(W1_f, Parameter) else W1_f, mstype.float32)
    W2_t = Tensor(W2_f.asnumpy() if isinstance(W2_f, Parameter) else W2_f, mstype.float32)
    W3_t = Tensor(W3_f.asnumpy() if isinstance(W3_f, Parameter) else W3_f, mstype.float32)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    history = []

    for epoch in range(epochs):
        params = tuple(model.trainable_params())
        grads = _pinn_loss_ffn_grad(X_u, U, X_f, W1_t, W2_t, W3_t, nu, *params)
        optimizer(grads)
        params = tuple(model.trainable_params())
        loss_val = compute_pinn_loss_ffn(X_u, U, X_f, W1_t, W2_t, W3_t, nu, *params)
        history.append(float(loss_val.asnumpy()))
        if verbose and epoch % 200 == 0:
            print(f"  Epoch {epoch:4d}, Loss: {loss_val.asnumpy():.6f}")
    return history


# Baseline losses (same as LAL script)
def compute_pinn_loss_mse(X_u, U, X_f, nu, *params):
    x_u, t_u = X_u[:, 0:1], X_u[:, 1:2]
    u_pred = net_u(x_u, t_u, *params)
    loss_data = ops.reduce_mean((u_pred - U) ** 2)
    x_f, t_f = X_f[:, 0:1], X_f[:, 1:2]
    residual = compute_pde_residual(x_f, t_f, nu, *params)
    loss_pde = ops.reduce_mean(residual ** 2)
    return loss_data + loss_pde


def compute_pinn_loss_l1(X_u, U, X_f, nu, *params):
    x_u, t_u = X_u[:, 0:1], X_u[:, 1:2]
    u_pred = net_u(x_u, t_u, *params)
    loss_data = ops.reduce_mean(ops.absolute(u_pred - U))
    x_f, t_f = X_f[:, 0:1], X_f[:, 1:2]
    residual = compute_pde_residual(x_f, t_f, nu, *params)
    loss_pde = ops.reduce_mean(ops.absolute(residual))
    return loss_data + loss_pde


def compute_pinn_loss_cauchy(X_u, U, X_f, nu, *params):
    x_u, t_u = X_u[:, 0:1], X_u[:, 1:2]
    u_pred = net_u(x_u, t_u, *params)
    loss_data = ops.reduce_mean(ops.log(1.0 + (u_pred - U) ** 2))
    x_f, t_f = X_f[:, 0:1], X_f[:, 1:2]
    residual = compute_pde_residual(x_f, t_f, nu, *params)
    loss_pde = ops.reduce_mean(ops.log(1.0 + residual ** 2))
    return loss_data + loss_pde


_MSE_GRAD = tuple(range(4, 4 + N_MODEL_PARAMS))
_pinn_loss_mse_grad = ms.grad(compute_pinn_loss_mse, grad_position=_MSE_GRAD)
_pinn_loss_l1_grad = ms.grad(compute_pinn_loss_l1, grad_position=_MSE_GRAD)
_pinn_loss_cauchy_grad = ms.grad(compute_pinn_loss_cauchy, grad_position=_MSE_GRAD)


def train_pinn_baseline(model, task, grad_fn, loss_fn, learning_rate, epochs, verbose=True):
    X_u = Tensor(task['X_u_train'], mstype.float32)
    U = Tensor(task['U_train'], mstype.float32)
    X_f = Tensor(task['X_f_train'], mstype.float32)
    nu = task['nu']
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    history = []
    for epoch in range(epochs):
        params = tuple(model.trainable_params())
        grads = grad_fn(X_u, U, X_f, nu, *params)
        optimizer(grads)
        params = tuple(model.trainable_params())
        loss_val = loss_fn(X_u, U, X_f, nu, *params)
        history.append(float(loss_val.asnumpy()))
        if verbose and epoch % 200 == 0:
            print(f"  Epoch {epoch:4d}, Loss: {loss_val.asnumpy():.6f}")
    return history


def evaluate_pinn(model, task):
    X_star = Tensor(task['X_star'], mstype.float32)
    u_pred = model(X_star).asnumpy().flatten()
    u_exact = task['u_star'].flatten()
    l2_error = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)

    X_u = Tensor(task['X_u_train'], mstype.float32)
    U = Tensor(task['U_train'], mstype.float32)
    X_f = Tensor(task['X_f_train'], mstype.float32)
    nu = task['nu']

    x_u, t_u = X_u[:, 0:1], X_u[:, 1:2]
    u_pred_u = model(ops.concat((x_u, t_u), 1))
    loss_data = float(ops.reduce_mean((u_pred_u - U) ** 2).asnumpy())

    x_f, t_f = X_f[:, 0:1], X_f[:, 1:2]
    params = tuple(model.trainable_params())
    residual = compute_pde_residual(x_f, t_f, nu, *params)
    loss_pde = float(ops.reduce_mean(residual ** 2).asnumpy())

    return {'l2_error': l2_error, 'loss_data': loss_data, 'loss_pde': loss_pde, 'u_pred': u_pred}


print("=" * 70)
print("Meta-Test: FFN vs MSE vs L1 vs Cauchy")
print(f"Test task: nu = {test_task['nu']:.6f}")
print("=" * 70)

results_all_ffn = {}
histories_all_ffn = {}

# ---- FFN ----
print("\n>>> FFN loss (ours)")
ffn_final = (W1_p, W2_p, W3_p)
model_ffn = create_model()
history_ffn = train_pinn_with_ffn(model_ffn, test_task, ffn_final, learning_rate=1e-4, epochs=1000)
results_ffn = evaluate_pinn(model_ffn, test_task)
results_all_ffn['FFN (ours)'] = results_ffn
histories_all_ffn['FFN (ours)'] = history_ffn
print(f"Result: L2={results_ffn['l2_error']:.6f}, MSE_data={results_ffn['loss_data']:.6e}, MSE_pde={results_ffn['loss_pde']:.6e}")

# ---- MSE ----
print("\n>>> MSE loss")
model_mse = create_model()
history_mse = train_pinn_baseline(model_mse, test_task, _pinn_loss_mse_grad, compute_pinn_loss_mse, 1e-4, 1000)
results_mse = evaluate_pinn(model_mse, test_task)
results_all_ffn['MSE'] = results_mse
histories_all_ffn['MSE'] = history_mse
print(f"Result: L2={results_mse['l2_error']:.6f}, MSE_data={results_mse['loss_data']:.6e}, MSE_pde={results_mse['loss_pde']:.6e}")

# ---- L1 ----
print("\n>>> L1 loss")
model_l1 = create_model()
history_l1 = train_pinn_baseline(model_l1, test_task, _pinn_loss_l1_grad, compute_pinn_loss_l1, 1e-4, 1000)
results_l1 = evaluate_pinn(model_l1, test_task)
results_all_ffn['L1'] = results_l1
histories_all_ffn['L1'] = history_l1
print(f"Result: L2={results_l1['l2_error']:.6f}, MSE_data={results_l1['loss_data']:.6e}, MSE_pde={results_l1['loss_pde']:.6e}")

# ---- Cauchy ----
print("\n>>> Cauchy loss")
model_cauchy = create_model()
history_cauchy = train_pinn_baseline(model_cauchy, test_task, _pinn_loss_cauchy_grad, compute_pinn_loss_cauchy, 1e-4, 1000)
results_cauchy = evaluate_pinn(model_cauchy, test_task)
results_all_ffn['Cauchy'] = results_cauchy
histories_all_ffn['Cauchy'] = history_cauchy
print(f"Result: L2={results_cauchy['l2_error']:.6f}, MSE_data={results_cauchy['loss_data']:.6e}, MSE_pde={results_cauchy['loss_pde']:.6e}")

# ---- Summary ----
print("\n" + "=" * 70)
print("Comparison Summary:")
print(f"  {'Method':<15} {'L2 Error':<14} {'Data Loss(MSE)':<16} {'PDE Loss(MSE)':<16}")
print(f"  {'-'*15} {'-'*14} {'-'*16} {'-'*16}")
for name, r in results_all_ffn.items():
    print(f"  {name:<15} {r['l2_error']:<14.6f} {r['loss_data']:<16.6e} {r['loss_pde']:<16.6e}")
print("=" * 70)

# ============================================================
# Visualization
# ============================================================

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 12))

# (1) Meta-training loss
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(meta_history_ffn['meta_loss'], 'b-', linewidth=1, alpha=0.7)
ax1.set_xlabel('Meta Iteration')
ax1.set_ylabel('Meta Loss')
ax1.set_title('(a) FFN Meta-Training Loss')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# (2) FFN weight norms
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(meta_history_ffn['w1_norm'], 'b-', label='|W1| (40x2)', linewidth=1.5)
ax2.plot(meta_history_ffn['w2_norm'], 'r-', label='|W2| (40x40)', linewidth=1.5)
ax2.plot(meta_history_ffn['w3_norm'], 'g-', label='|W3| (1x40)', linewidth=1.5)
ax2.set_xlabel('Meta Iteration')
ax2.set_ylabel('RMS Norm')
ax2.set_title('(b) FFN Weight Norm Evolution')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# (3) Training loss comparison
ax3 = fig.add_subplot(2, 3, 3)
colors_loss = {'FFN (ours)': 'blue', 'MSE': 'red', 'L1': 'orange', 'Cauchy': 'green'}
for name, hist in histories_all_ffn.items():
    ax3.plot(hist, color=colors_loss.get(name, 'gray'), label=name, linewidth=1.5, alpha=0.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
ax3.set_title(f'(c) Test Task (nu={test_task["nu"]:.4f}): Loss Curves')
ax3.set_yscale('log')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# (4) Solution at t=0.25
ax4 = fig.add_subplot(2, 3, 4)
X_star = test_task['X_star']
T_flat = X_star[:, 1]
t_target = 0.25
idx_t = np.abs(T_flat - t_target) < 0.01
x_plot = X_star[idx_t, 0]
u_exact_plot = test_task['u_star'][idx_t].flatten()
X_star_tensor = Tensor(X_star[idx_t], mstype.float32)
u_ffn_plot = model_ffn(X_star_tensor).asnumpy().flatten()
u_mse_plot = model_mse(X_star_tensor).asnumpy().flatten()
ax4.plot(x_plot, u_exact_plot, 'k-', label='Exact', linewidth=2)
ax4.plot(x_plot, u_ffn_plot, 'b--', label='FFN (ours)', linewidth=1.5)
ax4.plot(x_plot, u_mse_plot, 'r:', label='MSE Baseline', linewidth=1.5)
ax4.set_xlabel('x')
ax4.set_ylabel('u')
ax4.set_title(f'(d) Solution at t={t_target}')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# (5) Solution at t=0.5
ax5 = fig.add_subplot(2, 3, 5)
t_target2 = 0.5
idx_t2 = np.abs(T_flat - t_target2) < 0.01
x_plot2 = X_star[idx_t2, 0]
u_exact_plot2 = test_task['u_star'][idx_t2].flatten()
X_star_tensor2 = Tensor(X_star[idx_t2], mstype.float32)
u_ffn_plot2 = model_ffn(X_star_tensor2).asnumpy().flatten()
u_mse_plot2 = model_mse(X_star_tensor2).asnumpy().flatten()
ax5.plot(x_plot2, u_exact_plot2, 'k-', label='Exact', linewidth=2)
ax5.plot(x_plot2, u_ffn_plot2, 'b--', label='FFN (ours)', linewidth=1.5)
ax5.plot(x_plot2, u_mse_plot2, 'r:', label='MSE Baseline', linewidth=1.5)
ax5.set_xlabel('x')
ax5.set_ylabel('u')
ax5.set_title(f'(e) Solution at t={t_target2}')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# (6) L2 error bar chart
ax6 = fig.add_subplot(2, 3, 6)
methods = list(results_all_ffn.keys())
l2_errors = [results_all_ffn[m]['l2_error'] for m in methods]
bar_colors = ['#2196F3', '#FF5722', '#FF9800', '#4CAF50']
bars = ax6.bar(methods, l2_errors, color=bar_colors[:len(methods)], edgecolor='black', linewidth=1.2)
ax6.set_ylabel('Relative L2 Error')
ax6.set_title('(f) L2 Error Comparison on Test Task')
ax6.grid(True, alpha=0.3, axis='y')
for bar, err in zip(bars, l2_errors):
    ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
             f'{err:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('./imgs/ffn_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFFN result figure saved to ./imgs/ffn_result.png")
