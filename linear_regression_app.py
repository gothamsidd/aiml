"""
====================================================================
  INTERACTIVE LINEAR REGRESSION VISUALIZER
  Class Project — Built with Streamlit, Plotly, NumPy, Matplotlib
====================================================================

HOW TO RUN:
    pip install streamlit plotly numpy matplotlib
    streamlit run linear_regression_app.py
====================================================================
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Linear Regression Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (polished dark-academic theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0f1117;
    color: #e2e8f0;
}
h1 { font-weight: 800; letter-spacing: -1px; }
h2, h3 { font-weight: 600; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
}
.insight-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.metric-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: #818cf8;
    font-weight: 600;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════

def generate_dataset(dataset_type: str, n: int, noise: float, outlier: bool, seed: int = 42):
    """
    Generate synthetic X, y data.

    Parameters
    ----------
    dataset_type : 'Clean Linear' | 'Noisy'  | 'Custom Upload'
    n            : number of points
    noise        : standard deviation of Gaussian noise added to y
    outlier      : whether to inject 3 extreme outlier points
    seed         : random seed for reproducibility

    Returns
    -------
    X, y : numpy arrays of shape (n,)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, n)

    # True underlying relationship: y = 2x + 5  (+  noise)
    y = 2.0 * X + 5.0 + rng.normal(0, noise, n)

    if outlier:
        # Inject 3 obvious outliers far from the true line
        X = np.append(X, [2.0, 5.0, 8.0])
        y = np.append(y, [30.0, 0.5, 35.0])

    return X, y


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error = (1/n) * sum( (y_true - y_pred)^2 )

    This is our LOSS FUNCTION — a single number that tells
    how far our line is from the real data points.
    Lower MSE = better fit.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def predict(X: np.ndarray, m: float, b: float) -> np.ndarray:
    """
    Linear model prediction:  ŷ = m*X + b
    m = slope   (how steep the line is)
    b = intercept (where the line crosses y-axis)
    """
    return m * X + b


def gradient_descent_path(X, y, m_init, b_init, lr, n_steps):
    """
    Run Gradient Descent for n_steps iterations.

    At each step:
        ∂J/∂m = (-2/n) * sum( X * (y - ŷ) )
        ∂J/∂b = (-2/n) * sum( y - ŷ )

        m  ←  m - lr * ∂J/∂m
        b  ←  b - lr * ∂J/∂b

    Returns lists of (m, b, loss) at every step.
    """
    m, b = m_init, b_init
    history_m, history_b, history_loss = [m], [b], [mse(y, predict(X, m, b))]
    n = len(X)

    for _ in range(n_steps):
        y_pred = predict(X, m, b)
        residuals = y - y_pred          # actual minus predicted
        grad_m = (-2 / n) * np.dot(X, residuals)
        grad_b = (-2 / n) * np.sum(residuals)

        m = m - lr * grad_m
        b = b - lr * grad_b

        history_m.append(m)
        history_b.append(b)
        history_loss.append(mse(y, predict(X, m, b)))

    return np.array(history_m), np.array(history_b), np.array(history_loss)


def compute_loss_surface(X, y, m_range, b_range, grid_size=60):
    """
    Evaluate MSE on a 2-D grid of (m, b) values.
    This creates the 'loss landscape' — the bowl-shaped surface
    that Gradient Descent tries to roll down to reach the minimum.
    """
    ms = np.linspace(*m_range, grid_size)
    bs = np.linspace(*b_range, grid_size)
    MM, BB = np.meshgrid(ms, bs)
    ZZ = np.array([
        [mse(y, predict(X, mm, bb)) for mm in ms]
        for bb in bs
    ])
    return ms, bs, MM, BB, ZZ


def ols_solution(X, y):
    """
    Ordinary Least Squares (OLS) — the EXACT analytic solution.
    No iteration needed; directly gives the best m and b.

    m* = Cov(X,y) / Var(X)
    b* = mean(y) - m* * mean(X)

    Used to compare with Gradient Descent.
    """
    m_opt = np.cov(X, y, ddof=1)[0, 1] / np.var(X, ddof=1)
    b_opt = np.mean(y) - m_opt * np.mean(X)
    return m_opt, b_opt


# ════════════════════════════════════════════
#  SIDEBAR  — all user controls live here
# ════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # ── Dataset ──────────────────────────────
    st.markdown("### 📦 Dataset")
    dataset_type = st.selectbox("Type", ["Clean Linear", "Noisy"])
    n_points = st.slider("# of Data Points", 20, 200, 80, step=10)
    noise_level = st.slider("Noise Level (σ)", 0.1, 10.0, 2.5, step=0.1,
                            help="Standard deviation of Gaussian noise on y")
    add_outliers = st.checkbox("Inject Outliers", value=False,
                               help="Adds 3 extreme outlier points to show their effect")
    data_seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    # ── Manual Line ──────────────────────────
    st.markdown("### ✏️ Manual Line (Tab 1 & 2)")
    manual_m = st.slider("Slope  (m)", -5.0, 10.0, 2.0, step=0.1,
                         help="Steepness of the line: y changes by m for every 1-unit change in x")
    manual_b = st.slider("Intercept  (b)", -10.0, 20.0, 5.0, step=0.1,
                         help="Where the line crosses the y-axis (when x = 0)")

    st.markdown("---")
    # ── Gradient Descent ─────────────────────
    st.markdown("### 🎯 Gradient Descent")
    lr = st.select_slider("Learning Rate (α)",
                          options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                          value=0.01,
                          help="Step size for each parameter update. Too big → diverges. Too small → slow.")
    n_steps = st.slider("Iterations", 10, 500, 100, step=10)
    gd_m_init = st.slider("Initial Slope m₀", -5.0, 8.0, 0.0, step=0.5)
    gd_b_init = st.slider("Initial Intercept b₀", -10.0, 15.0, 0.0, step=0.5)

    st.markdown("---")
    st.caption("📘 Adjust any control and every chart updates instantly.")


# ════════════════════════════════════════════
#  GENERATE DATA
# ════════════════════════════════════════════

X, y = generate_dataset(dataset_type, n_points, noise_level, add_outliers, int(data_seed))
y_pred_manual = predict(X, manual_m, manual_b)
current_mse = mse(y, y_pred_manual)
m_opt, b_opt = ols_solution(X, y)
optimal_mse = mse(y, predict(X, m_opt, b_opt))


# ════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════

st.markdown("# 📈 Linear Regression — Interactive Visualizer")
st.markdown(
    "**Drag the sidebar sliders** to intuitively understand how a model *learns*, "
    "*measures error*, and *converges* to the best fit line."
)
st.markdown("---")

# Quick metric summary row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{current_mse:.2f}</div>'
                f'<div class="metric-label">Your Line MSE</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{optimal_mse:.2f}</div>'
                f'<div class="metric-label">Best Possible MSE</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{m_opt:.3f}</div>'
                f'<div class="metric-label">Optimal Slope m*</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{b_opt:.3f}</div>'
                f'<div class="metric-label">Optimal Intercept b*</div></div>', unsafe_allow_html=True)

st.markdown("")

# ════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════

tabs = st.tabs([
    "📌 Tab 1 — Data & Line Fit",
    "📐 Tab 2 — Error & Residuals",
    "🗺️ Tab 3 — Loss Landscape",
    "🚀 Tab 4 — Gradient Descent",
    "⚡ Tab 5 — Learning Rate",
    "🌩️ Tab 6 — Noise & Outliers",
])


# ─────────────────────────────────────────────
#  TAB 1: Data & Line Fit
# ─────────────────────────────────────────────
with tabs[0]:
    st.markdown("## 📌 Data Distribution & Line Fitting")

    st.markdown("""
    <div class="insight-box">
    <b>What is Linear Regression?</b><br>
    It finds the <em>best straight line</em> through data points. The line is defined by:
    <code>ŷ = m·x + b</code>  where <b>m</b> is the slope and <b>b</b> is the y-intercept.<br><br>
    👉 <b>What to observe:</b> Move the slope and intercept sliders (sidebar) and watch how the line changes.
    Try to match the data as closely as possible!
    </div>
    """, unsafe_allow_html=True)

    # Sort for clean line drawing
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]

    fig = go.Figure()

    # Scatter: actual data
    fig.add_trace(go.Scatter(
        x=X, y=y, mode='markers',
        marker=dict(color='#38bdf8', size=7, opacity=0.75, line=dict(color='#0ea5e9', width=0.5)),
        name='Data Points'
    ))

    # Manual line (user-controlled)
    y_line_manual = predict(X_sorted, manual_m, manual_b)
    fig.add_trace(go.Scatter(
        x=X_sorted, y=y_line_manual, mode='lines',
        line=dict(color='#f472b6', width=3),
        name=f'Your Line  (m={manual_m}, b={manual_b})'
    ))

    # Optimal OLS line
    y_line_opt = predict(X_sorted, m_opt, b_opt)
    fig.add_trace(go.Scatter(
        x=X_sorted, y=y_line_opt, mode='lines',
        line=dict(color='#4ade80', width=2, dash='dot'),
        name=f'Best Fit  (m={m_opt:.2f}, b={b_opt:.2f})'
    ))

    fig.update_layout(
        template='plotly_dark',
        title=dict(text="Scatter Plot — Your Line vs Best Fit", font=dict(size=16)),
        xaxis_title="X (Input Feature)",
        yaxis_title="y (Target)",
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=500,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"🎯 **Your MSE = {current_mse:.3f}**  vs  ✅ Optimal MSE = {optimal_mse:.3f}  "
            f"({'🎉 Great fit!' if current_mse < optimal_mse * 1.5 else '⬆️ Try adjusting the sliders'})")


# ─────────────────────────────────────────────
#  TAB 2: Error / Residuals
# ─────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📐 Error / Loss Function (MSE)")

    st.markdown("""
    <div class="insight-box">
    <b>What is MSE?</b><br>
    For every point, the <em>residual</em> = actual y − predicted ŷ.<br>
    MSE = average of all squared residuals = <code>(1/n) Σ (yᵢ − ŷᵢ)²</code><br><br>
    We square the errors so: (a) negatives don't cancel positives, (b) large errors are punished more.<br><br>
    👉 <b>What to observe:</b> Longer red lines = bigger errors. The MSE drops as your line moves closer to the data.
    </div>
    """, unsafe_allow_html=True)

    fig2 = go.Figure()

    # Data points
    fig2.add_trace(go.Scatter(
        x=X, y=y, mode='markers',
        marker=dict(color='#38bdf8', size=7, opacity=0.7),
        name='Actual y'
    ))

    # Regression line
    y_pred_sorted = predict(np.sort(X), manual_m, manual_b)
    fig2.add_trace(go.Scatter(
        x=np.sort(X), y=y_pred_sorted, mode='lines',
        line=dict(color='#f472b6', width=2.5),
        name='Predicted ŷ'
    ))

    # Residual lines (vertical segments from point to line)
    for xi, yi, ypi in zip(X, y, y_pred_manual):
        fig2.add_shape(type='line',
                       x0=xi, x1=xi, y0=yi, y1=ypi,
                       line=dict(color='#fb7185', width=1.5, dash='dot'))

    # Add invisible scatter just for legend entry
    fig2.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#fb7185', dash='dot', width=2),
        name='Residual (error)'
    ))

    fig2.update_layout(
        template='plotly_dark',
        title="Residuals — Distance from Line to Each Point",
        xaxis_title="X", yaxis_title="y",
        height=500, margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Residual distribution histogram
    residuals = y - y_pred_manual
    fig_hist = px.histogram(
        x=residuals, nbins=30,
        labels={'x': 'Residual (y − ŷ)'},
        title="Residual Distribution  (ideally centered at 0)",
        template='plotly_dark',
        color_discrete_sequence=['#818cf8'],
    )
    fig_hist.add_vline(x=0, line_color='#4ade80', line_dash='dash', annotation_text='Zero Error')
    fig_hist.update_layout(height=280, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.metric("Current MSE", f"{current_mse:.4f}", delta=f"{current_mse - optimal_mse:.4f} above optimal")


# ─────────────────────────────────────────────
#  TAB 3: Loss Landscape
# ─────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🗺️ Loss Surface (Parameter Space)")

    st.markdown("""
    <div class="insight-box">
    <b>What is the Loss Surface?</b><br>
    For every possible (m, b) pair there is one MSE value.
    Together they form a 3-D bowl-shaped surface called the <em>loss landscape</em>.<br>
    The very bottom of the bowl = optimal parameters.<br><br>
    👉 <b>What to observe:</b> Your current (m, b) appears as a red dot.
    The closer it is to the center of the contours, the better your line fits!
    </div>
    """, unsafe_allow_html=True)

    col_surf, col_cnt = st.columns(2)

    # Build grid around the optimal solution
    m_center, b_center = m_opt, b_opt
    m_range = (m_center - 5, m_center + 5)
    b_range = (b_center - 10, b_center + 10)

    ms, bs, MM, BB, ZZ = compute_loss_surface(X, y, m_range, b_range, grid_size=50)

    with col_surf:
        fig3d = go.Figure(data=[
            go.Surface(x=MM, y=BB, z=ZZ,
                       colorscale='Viridis', opacity=0.85,
                       showscale=False,
                       contours=dict(z=dict(show=True, usecolormap=True, project_z=True)))
        ])
        # Mark current manual position
        cur_loss = mse(y, predict(X, manual_m, manual_b))
        fig3d.add_trace(go.Scatter3d(
            x=[manual_m], y=[manual_b], z=[cur_loss + 1],
            mode='markers+text',
            marker=dict(size=8, color='#f43f5e'),
            text=['You'], textposition='top center',
            name='Your Position'
        ))
        # Mark optimal
        fig3d.add_trace(go.Scatter3d(
            x=[m_opt], y=[b_opt], z=[optimal_mse + 1],
            mode='markers+text',
            marker=dict(size=8, color='#4ade80', symbol='diamond'),
            text=['Optimal'], textposition='top center',
            name='Optimal'
        ))
        fig3d.update_layout(
            template='plotly_dark',
            title="3-D Loss Surface  J(m, b)",
            scene=dict(
                xaxis_title='Slope m',
                yaxis_title='Intercept b',
                zaxis_title='MSE Loss'
            ),
            height=480, margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with col_cnt:
        fig_cnt = go.Figure()
        fig_cnt.add_trace(go.Contour(
            x=ms, y=bs, z=ZZ,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            line=dict(smoothing=0.85),
            name='Loss Contour'
        ))
        # Current position
        fig_cnt.add_trace(go.Scatter(
            x=[manual_m], y=[manual_b], mode='markers+text',
            marker=dict(color='#f43f5e', size=14, symbol='x'),
            text=['You'], textposition='top right',
            name='Your (m, b)'
        ))
        # Optimal
        fig_cnt.add_trace(go.Scatter(
            x=[m_opt], y=[b_opt], mode='markers+text',
            marker=dict(color='#4ade80', size=12, symbol='star'),
            text=['Optimal'], textposition='top right',
            name='Optimal'
        ))
        fig_cnt.update_layout(
            template='plotly_dark',
            title="Contour Map (Top-Down View)",
            xaxis_title="Slope m",
            yaxis_title="Intercept b",
            height=480, margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_cnt, use_container_width=True)


# ─────────────────────────────────────────────
#  TAB 4: Gradient Descent
# ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🚀 Gradient Descent — The Learning Algorithm")

    st.markdown("""
    <div class="insight-box">
    <b>How does the model learn?</b><br>
    It starts at a random (m, b) and repeatedly nudges in the <em>downhill direction</em>
    of the loss surface until it reaches the minimum.<br>
    At each step:  <code>m ← m − α · ∂J/∂m</code>  and  <code>b ← b − α · ∂J/∂b</code><br>
    <b>α</b> is the <em>learning rate</em> — size of each step.<br><br>
    👉 <b>What to observe:</b> The red path traces the model's journey from start → optimal.
    Adjust learning rate in the sidebar to see convergence change dramatically.
    </div>
    """, unsafe_allow_html=True)

    # Run gradient descent with sidebar params
    hist_m, hist_b, hist_loss = gradient_descent_path(
        X, y, gd_m_init, gd_b_init, lr, n_steps
    )

    col_a, col_b = st.columns(2)

    with col_a:
        # GD path on contour
        ms_gd, bs_gd, _, _, ZZ_gd = compute_loss_surface(
            X, y, (m_opt - 6, m_opt + 6), (b_opt - 12, b_opt + 12), grid_size=50
        )
        fig_gd = go.Figure()
        fig_gd.add_trace(go.Contour(
            x=ms_gd, y=bs_gd, z=ZZ_gd,
            colorscale='Blues',
            contours=dict(showlabels=False),
            opacity=0.6, name='Loss Landscape'
        ))
        fig_gd.add_trace(go.Scatter(
            x=hist_m, y=hist_b, mode='lines+markers',
            line=dict(color='#fb923c', width=2),
            marker=dict(size=4, color='#fb923c'),
            name='GD Path'
        ))
        fig_gd.add_trace(go.Scatter(
            x=[gd_m_init], y=[gd_b_init], mode='markers+text',
            marker=dict(color='#f43f5e', size=12, symbol='circle'),
            text=['Start'], textposition='top right', name='Start'
        ))
        fig_gd.add_trace(go.Scatter(
            x=[hist_m[-1]], y=[hist_b[-1]], mode='markers+text',
            marker=dict(color='#4ade80', size=12, symbol='star'),
            text=['End'], textposition='top right', name='End'
        ))
        fig_gd.add_trace(go.Scatter(
            x=[m_opt], y=[b_opt], mode='markers',
            marker=dict(color='white', size=10, symbol='x'),
            name='True Optimal'
        ))
        fig_gd.update_layout(
            template='plotly_dark',
            title=f"Gradient Descent Path (α={lr}, {n_steps} steps)",
            xaxis_title="Slope m", yaxis_title="Intercept b",
            height=440, margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_gd, use_container_width=True)

    with col_b:
        # Loss vs iterations
        fig_loss_iter = go.Figure()
        fig_loss_iter.add_trace(go.Scatter(
            x=list(range(len(hist_loss))), y=hist_loss,
            mode='lines', line=dict(color='#a78bfa', width=2.5),
            name='Training Loss'
        ))
        fig_loss_iter.add_hline(y=optimal_mse,
                                line_color='#4ade80', line_dash='dash',
                                annotation_text=f'Optimal MSE = {optimal_mse:.2f}')
        fig_loss_iter.update_layout(
            template='plotly_dark',
            title="Loss vs Iterations",
            xaxis_title="Iteration",
            yaxis_title="MSE",
            height=440, margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_loss_iter, use_container_width=True)

    # Final GD result summary
    final_m, final_b = hist_m[-1], hist_b[-1]
    final_mse = hist_loss[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Final m (GD)", f"{final_m:.4f}", delta=f"{final_m - m_opt:.4f} from optimal")
    c2.metric("Final b (GD)", f"{final_b:.4f}", delta=f"{final_b - b_opt:.4f} from optimal")
    c3.metric("Final MSE (GD)", f"{final_mse:.4f}", delta=f"{final_mse - optimal_mse:.4f} from optimal")


# ─────────────────────────────────────────────
#  TAB 5: Learning Rate Experiments
# ─────────────────────────────────────────────
with tabs[4]:
    st.markdown("## ⚡ Learning Rate & Convergence Behavior")

    st.markdown("""
    <div class="insight-box">
    <b>Why does learning rate matter?</b><br>
    • <b>Too small (e.g. 0.0001):</b> Takes thousands of steps — very slow convergence.<br>
    • <b>Just right (e.g. 0.01):</b> Smooth, steady descent to the minimum.<br>
    • <b>Too large (e.g. 0.5+):</b> Overshoots! Bounces back and forth — may diverge.<br><br>
    👉 <b>What to observe:</b> All three curves start at the same point.
    Watch how their loss trajectories differ dramatically.
    </div>
    """, unsafe_allow_html=True)

    # Compare 4 fixed learning rates
    compare_lrs = [0.0005, 0.01, 0.05, 0.3]
    colors = ['#38bdf8', '#4ade80', '#f97316', '#f43f5e']
    labels = ['Very Small (0.0005)', 'Optimal (~0.01)', 'Large (0.05)', 'Too Large (0.3)']

    fig_lr = go.Figure()
    for lr_c, col, lab in zip(compare_lrs, colors, labels):
        _, _, h_loss = gradient_descent_path(X, y, 0.0, 0.0, lr_c, 200)
        # Clip for readability — divergence shows as very large loss
        h_loss_clipped = np.clip(h_loss, 0, max(h_loss[0] * 2, optimal_mse * 50))
        fig_lr.add_trace(go.Scatter(
            x=list(range(len(h_loss_clipped))), y=h_loss_clipped,
            mode='lines', line=dict(color=col, width=2.5), name=lab
        ))

    fig_lr.add_hline(y=optimal_mse,
                     line_color='white', line_dash='dot',
                     annotation_text='Optimal MSE')
    fig_lr.update_layout(
        template='plotly_dark',
        title="Loss vs Iterations — 4 Learning Rates Compared (200 steps, same start)",
        xaxis_title="Iteration",
        yaxis_title="MSE Loss",
        height=500, margin=dict(l=40, r=20, t=70, b=40),
    )
    st.plotly_chart(fig_lr, use_container_width=True)

    # Parameter evolution for current lr
    hist_m2, hist_b2, _ = gradient_descent_path(X, y, gd_m_init, gd_b_init, lr, n_steps)
    fig_params = go.Figure()
    fig_params.add_trace(go.Scatter(
        x=list(range(len(hist_m2))), y=hist_m2,
        mode='lines', line=dict(color='#f472b6', width=2), name='Slope m'
    ))
    fig_params.add_trace(go.Scatter(
        x=list(range(len(hist_b2))), y=hist_b2,
        mode='lines', line=dict(color='#38bdf8', width=2), name='Intercept b'
    ))
    fig_params.add_hline(y=m_opt, line_color='#f472b6', line_dash='dot',
                         annotation_text=f'm* = {m_opt:.2f}')
    fig_params.add_hline(y=b_opt, line_color='#38bdf8', line_dash='dot',
                         annotation_text=f'b* = {b_opt:.2f}')
    fig_params.update_layout(
        template='plotly_dark',
        title=f"Parameter Evolution  (α={lr})",
        xaxis_title="Iteration", yaxis_title="Parameter Value",
        height=320, margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_params, use_container_width=True)


# ─────────────────────────────────────────────
#  TAB 6: Noise & Outliers
# ─────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 🌩️ Effect of Noise & Outliers")

    st.markdown("""
    <div class="insight-box">
    <b>Why do outliers hurt Linear Regression?</b><br>
    MSE <em>squares</em> the error — so a point that is 10 units away contributes 100 to the loss,
    while a point 1 unit away contributes only 1.
    This makes the best-fit line get <em>pulled</em> toward outliers.<br><br>
    👉 <b>What to observe:</b> Enable "Inject Outliers" in the sidebar and watch how the
    optimal line shifts. Compare the two fits side-by-side.
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # Left: clean data
    X_clean, y_clean = generate_dataset(dataset_type, n_points, noise_level, False, int(data_seed))
    m_clean, b_clean = ols_solution(X_clean, y_clean)

    # Right: with outliers
    X_out, y_out = generate_dataset(dataset_type, n_points, noise_level, True, int(data_seed))
    m_out_opt, b_out_opt = ols_solution(X_out, y_out)

    sort_c = np.argsort(X_clean)
    sort_o = np.argsort(X_out)

    with col_l:
        fig_clean = go.Figure()
        fig_clean.add_trace(go.Scatter(
            x=X_clean, y=y_clean, mode='markers',
            marker=dict(color='#38bdf8', size=7, opacity=0.7), name='Data'
        ))
        fig_clean.add_trace(go.Scatter(
            x=X_clean[sort_c], y=predict(X_clean[sort_c], m_clean, b_clean),
            mode='lines', line=dict(color='#4ade80', width=2.5), name='Best Fit'
        ))
        fig_clean.update_layout(
            template='plotly_dark',
            title=f"✅ Without Outliers  (m={m_clean:.2f}, b={b_clean:.2f})",
            xaxis_title="X", yaxis_title="y",
            height=380, margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_clean, use_container_width=True)
        st.metric("MSE (no outliers)", f"{mse(y_clean, predict(X_clean, m_clean, b_clean)):.3f}")

    with col_r:
        fig_out = go.Figure()
        # Normal points
        fig_out.add_trace(go.Scatter(
            x=X_out[:-3], y=y_out[:-3], mode='markers',
            marker=dict(color='#38bdf8', size=7, opacity=0.7), name='Normal Data'
        ))
        # Outliers highlighted in red
        fig_out.add_trace(go.Scatter(
            x=X_out[-3:], y=y_out[-3:], mode='markers',
            marker=dict(color='#f43f5e', size=14, symbol='x-open', line=dict(width=3)),
            name='⚠️ Outliers'
        ))
        # Distorted best fit
        fig_out.add_trace(go.Scatter(
            x=X_out[sort_o], y=predict(X_out[sort_o], m_out_opt, b_out_opt),
            mode='lines', line=dict(color='#f97316', width=2.5), name='Distorted Fit'
        ))
        # True optimal (without outliers) for comparison
        fig_out.add_trace(go.Scatter(
            x=X_out[sort_o], y=predict(X_out[sort_o], m_clean, b_clean),
            mode='lines', line=dict(color='#4ade80', width=1.5, dash='dot'),
            name='Original Fit (ref)'
        ))
        fig_out.update_layout(
            template='plotly_dark',
            title=f"⚠️ With Outliers  (m={m_out_opt:.2f}, b={b_out_opt:.2f})",
            xaxis_title="X", yaxis_title="y",
            height=380, margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_out, use_container_width=True)
        st.metric("MSE (with outliers)", f"{mse(y_out, predict(X_out, m_out_opt, b_out_opt)):.3f}",
                  delta=f"+{mse(y_out, predict(X_out, m_out_opt, b_out_opt)) - mse(y_clean, predict(X_clean, m_clean, b_clean)):.3f}")

    # Noise effect: MSE vs noise level
    st.markdown("### How does noise level affect model accuracy?")
    noise_vals = np.linspace(0.1, 10.0, 40)
    mse_vals = []
    for nv in noise_vals:
        Xn, yn = generate_dataset(dataset_type, n_points, nv, False, int(data_seed))
        mo, bo = ols_solution(Xn, yn)
        mse_vals.append(mse(yn, predict(Xn, mo, bo)))

    fig_noise = go.Figure()
    fig_noise.add_trace(go.Scatter(
        x=noise_vals, y=mse_vals,
        mode='lines+markers',
        line=dict(color='#c084fc', width=2),
        marker=dict(size=5),
        name='MSE vs Noise'
    ))
    fig_noise.add_vline(x=noise_level, line_color='#fbbf24', line_dash='dash',
                        annotation_text=f'Current noise = {noise_level}')
    fig_noise.update_layout(
        template='plotly_dark',
        title="Optimal MSE vs Noise Level",
        xaxis_title="Noise Level (σ)",
        yaxis_title="Optimal MSE",
        height=300, margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_noise, use_container_width=True)


# ════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.8rem;'>"
    "Built with Streamlit · NumPy · Plotly &nbsp;|&nbsp; "
    "Interactive Linear Regression Visualizer &nbsp;|&nbsp; Class Project"
    "</div>",
    unsafe_allow_html=True
)