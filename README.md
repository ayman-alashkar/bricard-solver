# Bricard 6R Linkage Kinematic Solver

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

Interactive kinematic solver for the **original general line-symmetric Bricard 6R linkage** using tensor method.


## Features

- 📊 **Kinematic Curves** — Plot θ₂ and θ₃ vs input θ₁
- 🎯 **3D Visualization** — Interactive 3D view of the linkage
- 🔄 **Phase Portrait** — Configuration space (θ₂ vs θ₃)
- 📋 **Data Export** — Download kinematic data as CSV
- ⚙️ **Preset Configurations** — Load pre-defined examples
- 🎨 **Modern UI** — Clean, responsive interface

## Mathematical Background

The solver implements the direct trigonometric solution:

$$\theta = \psi \pm \arccos\left(\frac{-R}{\sqrt{P^2+Q^2}}\right)$$

where $\psi = \text{atan2}(P, Q)$

Based on: A Tensor Method for the Kinematical Analysis of the Line-Symmetric Bricard 6R Linkage (2025).

---
## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/ayman-alashkar/bricard-solver.git
cd bricard-solver

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run bricard_solver_enhanced.py
```

### Quick Start (No Installation)

Just visit the deployed app: `https://bricard-solver.streamlit.app`

## Deployment to Streamlit Cloud (Free)

### Step 1: Create a GitHub Repository

1. Go to [github.com](https://github.com) and create a new repository
2. Name it something like `bricard-solver`
3. Upload these files:
   - `bricard_solver_enhanced.py` (rename to `app.py` or `streamlit_app.py`)
   - `requirements.txt`
   - `README.md` (optional)

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository and branch
5. Set the main file path (e.g., `app.py`)
6. Click **"Deploy!"**

Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Alternative Deployment Options

| Platform | Link | Notes |
|:---------|:-----|:------|
| Streamlit Cloud | [share.streamlit.io](https://share.streamlit.io) | Free, easiest |
| Hugging Face Spaces | [huggingface.co/spaces](https://huggingface.co/spaces) | Free, ML-focused |
| Railway | [railway.app](https://railway.app) | $5 free credit |
| Render | [render.com](https://render.com) | Free tier available |

## File Structure

```
bricard-solver/
├── app.py                      # Main Streamlit app (enhanced version)
├── bricard_solver_standalone.py # Standalone matplotlib version
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── bricard_kinematic_curves.png # Example output
```

## Usage

1. **Set Link Parameters** — Configure a₁, α₁, R₁ for links 1&4, 2&5, 3&6
2. **Adjust Input Angle** — Use the slider to change θ₁
3. **View Results** — See θ₂, θ₃ for both closure forms
4. **Explore Tabs** — Check kinematic curves, 3D view, phase portrait
5. **Export Data** — Download CSV for further analysis

## Parameters

Due to line symmetry, the Bricard 6R linkage satisfies:
- Link length: $a_i = a_{i+3}$
- Twist angle: $\alpha_i = \alpha_{i+3}$  
- Offset: $R_i = R_{i+3}$

You only need to specify parameters for links 1, 2, 3.

## References

1. Bricard, R. (1897). "Mémoire sur la théorie de l'octaèdre articulé." *J. Math. Pures Appl.*
2. Casey, J. & Lam, V.C. (1986). "A tensor method for the kinematical analysis of systems of rigid bodies." *Mech. Mach. Theory*
3. Song, C.-Y., Chen, Y., & Chen, I.-M. (2014). "Kinematic study of the original and revised general line-symmetric Bricard 6R linkages." *J. Mech. Robot.*

## Author

**Ayman Alashkar**  
Mechanics and Materials Unit  
Okinawa Institute of Science and Technology (OIST)

December 2025

## License

MIT License - Feel free to use and modify!
