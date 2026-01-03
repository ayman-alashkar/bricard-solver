# Bricard 6R Linkage — Tensor-Based Kinematic Analysis (Streamlit)

Interactive Streamlit application for analyzing the **line‑symmetric Bricard 6R** overconstrained mechanism using a **tensor-based** closure formulation.

Author: **Ayman Alashkar** (OIST)  
Date: **December 2025**

---

## What this app does

Given the line‑symmetric Bricard 6R parameters, the app:

- Computes **two solution branches** (Form I and Form II) over a user-defined **θ₁ sweep**.
- Evaluates **Jacobian singular values (SVD)** along each branch (useful for spotting near‑singular configurations).
- Performs **collision detection** and reports **physically valid** θ₁ regions (based on geometric clearance).
- Lets you **download CSVs** for both forms.

---

## Quick start

### 1) Create an environment (recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
```

### 2) Install dependencies

Minimum:

```bash
pip install streamlit numpy pandas
```

Optional (for nicer validity-colored angle curves):

```bash
pip install altair
```

### 3) Run the app

```bash
streamlit run bricard_6r_streamlit.py
```

---

## Using the UI

All inputs are in the **sidebar**:

### Linkage parameters (line symmetry)

- Link lengths: `a₁₂=a₄₅`, `a₂₃=a₅₆`, `a₃₄=a₆₁`
- Twist angles (degrees): `α₁₂=α₄₅`, `α₂₃=α₅₆`, `α₃₄=α₆₁`
- Offsets: `R₁=R₄`, `R₂=R₅`, `R₃=R₆`

### θ₁ sweep

- `θ₁ min / θ₁ max` (degrees)
- Number of samples

### Collision geometry

Used only for physical validity (clearance) checks:

- Joint radius
- Offset radius
- Link width / depth

### Advanced settings (if the solver struggles)

- **Grid search resolution**: larger values improve chances of finding both forms but are slower.
- **Solver tolerance**: loosen if you get “no solution found”.
- **Initial guesses (degrees)**: manually seed Form I and Form II near expected configurations.

---

## Outputs

### Metrics

- “Form I solved / Form II solved” = percentage of θ₁ samples where the nonlinear solver converged.
- “Form I valid / Form II valid” = converged *and* collision-free (clearance ≥ 0).

### Valid motion ranges

Shows contiguous θ₁ intervals where the configuration is both mathematically closed and collision-free.

### Singular values (SVD)

Plots the six singular values of the 6×6 Jacobian along the sweep for each form.

### Joint angles vs θ₁

Plots θ₂(θ₁), θ₃(θ₁).  
If Altair is installed, the curves are colored by physical validity (valid vs collision).

### Raw data + CSV export

Expand “Raw Data” to inspect tables. Use the download buttons to export:

- `form_I.csv`
- `form_II.csv`

---

## Mathematical model (high level)

The implementation follows a tensor formulation:

- Inter‑link rotation tensor `Q^{n+1,n} = Q(θ, α)` (3×3)
- Cumulative rotations `Q^{n,0}` built by multiplying the six inter‑link tensors
- Closure residual combines:
  - rotational closure (return to identity)
  - translational closure `Σ (R_n e^n_3 + a_n e^{n+1}_1) = 0`

The **line‑symmetric constraint** is enforced as:

`[θ₁, θ₂, θ₃, θ₁, θ₂, θ₃]`.

---

## How “Form I” and “Form II” are found

This is a **numerical branch finding** approach:

1. At the first θ₁ sample, the app runs a grid of initial guesses for (θ₂, θ₃) and collects distinct converged roots.
2. It chooses:
   - Form I = one root (sorted first),
   - Form II = the root **farthest** from Form I in angular distance.
3. During the sweep, each form is tracked by **continuation**: the previous (θ₂, θ₃) becomes the next initial guess.
4. If enabled, a small random “recovery” search attempts to re-lock onto a branch if a step fails.

---

## Notes / troubleshooting

- If you see **“No solution found”**:
  - Increase *Grid search resolution*.
  - Loosen *Solver tolerance* (e.g., 1e‑6).
  - Provide better *initial guesses* for Form I / II.
- Collision checks depend strongly on the geometry radii/width/depth you choose.

---

## Repository tips

Recommended files to include alongside this script:

- `README.md` (this file)
- `requirements.txt` (optional)
- `LICENSE` (optional)

Example `requirements.txt`:

```txt
streamlit
numpy
pandas
altair
```