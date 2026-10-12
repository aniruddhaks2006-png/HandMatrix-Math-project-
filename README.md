# HandMatrix: Real-Time Spatial Convolution Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Landmark%20Tracking-orange.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License">
</p>

An interactive, real-time spatial image processing framework that dynamically binds 2D spatial convolution operators to two-hand geometric landmarks.

Using MediaPipe's hand tracking pipeline and OpenCV, **HandMatrix** turns the thumb and index finger tips of both hands into an arbitrary 4-point quadrilateral control cage. The script builds analytical discrete kernels (Sobel gradients, Laplacian second-order derivatives, high-pass sharpening, and Gaussian smoothing), scales their footprint based on hand spread, and applies them strictly within the bounded region with feathered alpha-mask blending. A built-in layer stacking architecture allows freezing regions and compounding filters sequentially across the video feed.

---

## Key Features

- **Dynamic Quadrilateral Spatial Hull:** Maps control points across dual hands to define a dynamic convex region of interest.
- **Anti-Self-Intersection Geometry:** Applies centroid angular sorting ($\text{atan2}$) to ensure the control polygon stays convex and never twists when hands cross paths[cite: 1].
- **Analytical Kernel Synthesis:** Synthesizes Sobel and derivative operators via outer products of 1D binomial coefficients and difference vectors with continuous $L_1$ normalization[cite: 1].
- **Scale-Adaptive Footprints:** Dynamically scales kernel dimensions ($3\times3$ up to $31\times31$) relative to the Euclidean diagonal of the hand polygon[cite: 1].
- **Layer Stacking System:** Freeze regions to the background stack (`L`) to composite multiple spatial filters on top of each other.
- **Single-Hand Gesture Classification:** Real-time state machine switches image filtering kernels using finger extensions on the leftmost hand[cite: 1].
- **Boundary Feathering:** Eliminates harsh edge artifacts using Gaussian-blurred alpha polygon masks[cite: 1].
- **Threaded Buffer Management:** Background capture thread prevents frame-buffer latency and maintains fluid full-screen rendering[cite: 1].

---

## Mathematical Architecture

### 1. Centroid-Relative Vertex Ordering
To prevent self-intersection as hands translate and rotate, the 4 landmark control points $P_i = (x_i, y_i)$ are sorted relative to their centroid $C = (\bar{x}, \bar{y})$[cite: 1]:

$$\bar{x} = \frac{1}{4}\sum_{i=1}^{4} x_i, \quad \bar{y} = \frac{1}{4}\sum_{i=1}^{4} y_i$$

$$\theta_i = \text{atan2}(y_i - \bar{y},\; x_i - \bar{x})$$

The vertices are sorted in monotonic counter-clockwise order by $\theta_i$, ensuring a valid non-self-intersecting hull for polygon rasterization[cite: 1].

### 2. Discrete Separable Kernel Construction
Rather than relying on fixed static masks, horizontal and vertical Sobel filters are synthesized as outer products of a binomial smoothing vector $B$ and a discrete derivative vector $D$ of odd dimension $n \in \{3, 5, \dots, 31\}$[cite: 1]:

$$B_k = \binom{n - 1}{k}, \quad k \in [0, n - 1]$$

$$D_k = k - \left\lfloor \frac{n}{2} \right\rfloor, \quad k \in [0, n - 1]$$

$$K_x = B^T \otimes D, \quad K_y = D^T \otimes B$$

Kernels are normalized by absolute sum to prevent numerical overflow and maintain energy stability across arbitrary kernel scales[cite: 1]:

$$\hat{K} = \frac{K}{\sum_{i,j} \vert{}K_{i,j}\vert{}}$$

Gradient vector magnitude is computed across channels as[cite: 1]:

$$\vert{}\nabla I\vert{} = \sqrt{(I * K_x)^2 + (I * K_y)^2}$$

---

## Gestures & Controls

### Gesture Control Reference
Operated using the **leftmost hand** detected in the frame[cite: 1]:

| Gesture | Fingers Extended | Convolution Operator[cite: 1] |
| :--- | :--- | :--- |
| **Closed Fist** | 0 fingers | **Laplacian Derivative** ($\nabla^2 I$)[cite: 1] |
| **Index Extended** | 1 finger | **Sobel X** (Horizontal Gradient $\frac{\partial I}{\partial x}$)[cite: 1] |
| **Victory / Peace** | 2 fingers | **Sobel Y** (Vertical Gradient $\frac{\partial I}{\partial y}$)[cite: 1] |
| **Three Fingers** | 3 fingers | **Sobel Magnitude** ($\sqrt{G_x^2 + G_y^2}$)[cite: 1] |
| **Four Fingers** (Thumb Tucked) | 4 fingers | **High-Pass Sharpen** ($\delta - \nabla^2$)[cite: 1] |
| **Open Palm** (Thumb Out) | 5 fingers | **Gaussian Smoothing** ($G_\sigma$)[cite: 1] |

### Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `L` | **Lock / Push Layer:** Commits current quadrilateral region to the persistent stack and clears hands for a new region. |
| `U` / `Z` | **Undo:** Pops the top-most locked filter layer off the stack. |
| `C` | **Clear:** Flushes all stacked layers and resets hand slot tracking. |
| `Q` / `ESC` | **Exit:** Releases camera capture and closes full-screen window. |
| **Palms Together** | **Cancel Gesture:** Hold wrists together for $\approx 0.7\text{s}$ to wipe all layers without touching the keyboard[cite: 1]. |

---

## Project Structure

```text
├── main.py              # Application entry point, pipeline loop, and OpenCV renderer
├── requirements.txt     # Python library dependencies
├── .gitignore           # Ignores compiled artifacts and local model checkpoints
└── README.md            # Technical documentation
