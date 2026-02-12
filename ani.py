import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # <-- CRITICAL FIX
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -------------------------------
# Loss function
# -------------------------------
def loss(x, y):
    return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

# -------------------------------
# Gradient
# -------------------------------
def gradient(x, y):
    dx = np.cos(x) * np.cos(y) + 0.2 * x
    dy = -np.sin(x) * np.sin(y) + 0.2 * y
    return dx, dy

# -------------------------------
# Surface
# -------------------------------
x = np.linspace(-5, 5, 150)
y = np.linspace(-5, 5, 150)
X, Y = np.meshgrid(x, y)
Z = loss(X, Y)

# -------------------------------
# Plot setup
# -------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)
ax.set_title("Dynamic 3D: GD vs Momentum")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Loss")

# -------------------------------
# Hyperparameters
# -------------------------------
lr = 0.08
beta = 0.9
steps = 60

# -------------------------------
# Initial state
# -------------------------------
x_gd, y_gd = 4.0, 4.0
x_m, y_m = 4.0, 4.0
vx, vy = 0.0, 0.0

gd_x, gd_y, gd_z = [], [], []
m_x, m_y, m_z = [], [], []

gd_line, = ax.plot([], [], [], 'r-o', label="Gradient Descent")
m_line, = ax.plot([], [], [], 'c-o', label="Momentum")

ax.legend()

# -------------------------------
# Animation update
# -------------------------------
def update(frame):
    global x_gd, y_gd, x_m, y_m, vx, vy

    # Gradient Descent
    dx, dy = gradient(x_gd, y_gd)
    x_gd -= lr * dx
    y_gd -= lr * dy
    gd_x.append(x_gd)
    gd_y.append(y_gd)
    gd_z.append(loss(x_gd, y_gd))
    gd_line.set_data(gd_x, gd_y)
    gd_line.set_3d_properties(gd_z)

    # Momentum GD
    dx, dy = gradient(x_m, y_m)
    vx = beta * vx + lr * dx
    vy = beta * vy + lr * dy
    x_m -= vx
    y_m -= vy
    m_x.append(x_m)
    m_y.append(y_m)
    m_z.append(loss(x_m, y_m))
    m_line.set_data(m_x, m_y)
    m_line.set_3d_properties(m_z)

    return gd_line, m_line

# -------------------------------
# KEEP REFERENCE (CRITICAL)
# -------------------------------
ani = FuncAnimation(fig, update, frames=steps, interval=300)

plt.show()
