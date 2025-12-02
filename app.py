import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------- Utility functions --------------------------

DISTRIBUTIONS = {
    "Normal (Gaussian)": stats.norm,
    "Log-Normal": stats.lognorm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Beta": stats.beta,
    "Exponential": stats.expon,
    "Cauchy": stats.cauchy,
    "Chi-squared": stats.chi2,
    "Laplace": stats.laplace,
    "Uniform": stats.uniform,
    "Logistic": stats.logistic,
    "Rayleigh": stats.rayleigh,
}

DISTRIBUTION_SLIDERS = {
    "Normal (Gaussian)": [("loc", -10, 10, 0), ("scale", 0.01, 10, 1)],
    "Log-Normal": [("s", 0.1, 3, 1), ("loc", -5, 5, 0), ("scale", 0.01, 5, 1)],
    "Gamma": [("a", 0.1, 10, 1), ("loc", -5, 5, 0), ("scale", 0.01, 5, 1)],
    "Weibull": [("c", 0.2, 10, 1), ("loc", -5, 5, 0), ("scale", 0.01, 10, 1)],
    "Beta": [("a",0.1,5,1),("b",0.1,5,1), ("loc", -1, 1, 0), ("scale", 0.01, 5, 1)],
    "Exponential": [("loc",-5,5,0),("scale",0.01,5,1)],
    "Cauchy": [("loc", -10, 10, 0), ("scale", 0.01, 10, 1)],
    "Chi-squared": [("df", 0.5, 10, 1), ("loc", -2, 2, 0), ("scale", 0.01, 5, 1)],
    "Laplace": [("loc", -10, 10, 0), ("scale", 0.01, 5, 1)],
    "Uniform": [("loc", -10, 10, 0), ("scale", 0.01, 20, 1)],
    "Logistic": [("loc", -10, 10, 0), ("scale", 0.01, 10, 1)],
    "Rayleigh": [("loc", -10, 10, 0), ("scale", 0.01, 10, 1)],
}

def fit_distribution(data, distribution_obj):
    # Defensive - restrict to 10000 rows
    data = np.asarray(data)
    if len(data) > 10000:
        data = data[:10000]
    try:
        params = distribution_obj.fit(data)
    except Exception as e:
        params = (np.nan,)*distribution_obj.numargs + (np.nan, np.nan)
    return params

def compute_fit_quality(data, dist_obj, fitted_params, bins):
    # Calc the PDF at the bin centers based on fit
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    try:
        pdf_fitted = dist_obj.pdf(bin_centers, *fitted_params[:-2], loc=fitted_params[-2], scale=fitted_params[-1])
        error = np.abs(hist - pdf_fitted)
        return {
            'avg abs error': np.mean(error),
            'max abs error': np.max(error)
        }
    except Exception as e:
        return {
            'avg abs error': np.nan,
            'max abs error': np.nan
        }

# ----------------------- Streamlit webapp setup --------------------------

st.set_page_config(page_title="Histogram Fitter", layout="wide")
st.title("📊 Histogram Distribution Fitter Webapp")
st.caption("by Taranprit Saini | NE 111 | December 2, 2025")

# Data Input Section
st.header("1️⃣ Enter Your Data")
input_mode = st.radio("Select Data Input Method:", ["Manual Entry", "Upload CSV File"], horizontal=True)
data_uploaded = None
df = None
if input_mode == "Manual Entry":
    sample_text = "12.0\n13.1\n15.2\n14.9\n13.2\n15.6\n13.5\n17.3"
    manual_data = st.text_area("Paste or enter your data (one value per line):", sample_text, height=200)
    try:
        data_lines = [float(x.strip()) for x in manual_data.strip().split("\n") if x.strip()]
        data_uploaded = np.array(data_lines)
        st.success(f"{len(data_uploaded)} data points loaded.")
    except Exception as e:
        st.warning("Please check your data format.")
else:
    uploaded_file = st.file_uploader("Upload CSV file (one column of numbers)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # attempt to use first column
            col = df.columns[0]
            data_uploaded = df[col].dropna().to_numpy()
            st.success(f"{len(data_uploaded)} data points loaded from '{col}'.")
        except Exception as e:
            st.warning('Upload error: Make sure your CSV has a single column of numbers.')

st.session_state['data'] = data_uploaded

# Distribution Fitting & Visualization Section
st.divider()
st.header("2️⃣ Distribution Fitting & Visualization")
data = st.session_state.get('data', None)
if data is None or len(data)==0:
    st.info("ℹ️ Please enter or upload data above to start.")
else:
        # Top row: Distribution selection and fitting mode
        col_top1, col_top2 = st.columns([2, 1])
        with col_top1:
            dist_name = st.selectbox("📈 Select a distribution to fit:", list(DISTRIBUTIONS.keys()), index=0)
        with col_top2:
            fit_mode = st.radio("⚙️ Fitting mode:", ["Automatic (Best Fit)", "Manual (Adjust Parameters)"], horizontal=True)
        
        distribution = DISTRIBUTIONS[dist_name]
        
        # Manual parameter sliders (if in manual mode)
        if fit_mode == "Manual (Adjust Parameters)":
            with st.expander("🎚️ Manual Parameter Adjustment", expanded=True):
                slider_settings = DISTRIBUTION_SLIDERS[dist_name]
                user_params = []
                slider_cols = st.columns(min(len(slider_settings), 4))
                for idx, (param, mi, ma, default) in enumerate(slider_settings):
                    with slider_cols[idx % len(slider_cols)]:
                        val = st.slider(f"{param}", float(mi), float(ma), float(default), key=f"slider_{dist_name}_{param}")
                        user_params.append(val)
                # For manual mode, need right number of args before loc/scale
                num_shapes = distribution.numargs
                shapes = user_params[:num_shapes]
                loc = user_params[num_shapes] if num_shapes < len(user_params) else 0
                scale = user_params[num_shapes+1] if num_shapes+1 < len(user_params) else 1
                params = tuple(list(shapes) + [loc, scale])
        else:
            # Automatic fitting
            params = fit_distribution(data, distribution)
        
        # Plot settings
        with st.expander("📊 Plot Settings", expanded=False):
            bins = st.slider("Number of histogram bins:", 10, 100, 30, key='bins')
            plot_min = st.number_input("Plot X-min", value=float(np.min(data))-2, step=1.0, key='pmin')
            plot_max = st.number_input("Plot X-max", value=float(np.max(data))+2, step=1.0, key='pmax')
        
        # Main visualization area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            x = np.linspace(plot_min, plot_max, 300)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data, bins=bins, alpha=0.4, color='skyblue', density=True, label="Data Histogram", edgecolor='black', linewidth=0.5)
            
            try:
                pdf_vals = distribution.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
                ax.plot(x, pdf_vals, 'r-', linewidth=2, label=f'Fit: {dist_name}')
            except Exception as e:
                pdf_vals = np.zeros_like(x)
                st.warning(f"Could not compute PDF: {str(e)}")
            
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'Distribution Fit: {dist_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("📋 Fit Results")
            
            # Fit Parameters
            with st.expander("🔢 Fitted Parameters", expanded=True):
                param_names = (distribution.shapes or "")
                param_names = [x.strip() for x in param_names.split(",") if x.strip()]
                param_names += ["loc", "scale"]
                param_dict = dict(zip(param_names, params))
                
                # Display parameters in a nicer format
                for key, value in param_dict.items():
                    if np.isnan(value):
                        st.write(f"**{key}**: `NaN`")
                    else:
                        st.write(f"**{key}**: `{value:.6f}`")
            
            # Fit Quality Metrics
            with st.expander("📊 Fit Quality Metrics", expanded=True):
                fit_quality = compute_fit_quality(data, distribution, params, bins)
                for key, value in fit_quality.items():
                    if np.isnan(value):
                        st.metric(key.replace('_', ' ').title(), "NaN")
                    else:
                        st.metric(key.replace('_', ' ').title(), f"{value:.6f}")
            
            # Data summary
            with st.expander("📈 Data Summary", expanded=False):
                st.write(f"**Number of points**: {len(data)}")
                st.write(f"**Mean**: {np.mean(data):.4f}")
                st.write(f"**Std Dev**: {np.std(data):.4f}")
                st.write(f"**Min**: {np.min(data):.4f}")
                st.write(f"**Max**: {np.max(data):.4f}")

        # Instructions
        with st.expander("ℹ️ Instructions & Information", expanded=False):
            st.markdown("""
            **How to use this app:**
            1. **Enter Data**: In the section above, either:
               - Paste your data manually (one value per line)
               - Upload a CSV file with a single column of numbers
            
            2. **Select Distribution**: Choose from 12 available distributions:
               - Normal, Log-Normal, Gamma, Weibull, Beta
               - Exponential, Cauchy, Chi-squared, Laplace
               - Uniform, Logistic, Rayleigh
            
            3. **Choose Fitting Mode**:
               - **Automatic**: Uses scipy's built-in fitting function to find optimal parameters
               - **Manual**: Adjust sliders to manually tune distribution parameters
            
            4. **Review Results**: 
               - View the visualization comparing your data histogram with the fitted curve
               - Check the fitted parameters and quality metrics
               - Compare different distributions to find the best fit
            """)