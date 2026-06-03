import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from utils import preprocess_image, load_trained_model, predict_digit
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Digit Recognition ML Demo",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Root color scheme - professional & elegant */
    :root {
        --primary: #2E86AB;
        --secondary: #A23B72;
        --accent: #F18F01;
        --success: #06A77D;
        --light-bg: #F8F9FA;
        --card-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        --card-shadow-hover: 0 4px 12px rgba(0, 0, 0, 0.12);
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }

    /* Header section */
    .header-container {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(46, 134, 171, 0.3);
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.95;
        letter-spacing: 0.3px;
    }

    /* Card component */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .custom-card:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-2px);
    }

    /* Prediction result card */
    .prediction-result-card {
        background: linear-gradient(135deg, #2E86AB 0%, #1a5a7a 100%);
        color: white;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(46, 134, 171, 0.25);
        margin: 1.5rem 0;
    }

    .prediction-digit {
        font-size: 6rem;
        font-weight: 800;
        line-height: 1;
        margin: 1rem 0;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        letter-spacing: -2px;
    }

    .prediction-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
        font-weight: 600;
    }

    /* Metrics grid */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        border-left: 4px solid #2E86AB;
        transition: all 0.3s ease;
    }

    .metric-box:hover {
        border-left-color: #F18F01;
        box-shadow: var(--card-shadow-hover);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86AB;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed #2E86AB;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        background: rgba(46, 134, 171, 0.04);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .upload-area:hover {
        border-color: #F18F01;
        background: rgba(241, 143, 1, 0.04);
        box-shadow: var(--card-shadow);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB 0%, #1a5a7a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46, 134, 171, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(46, 134, 171, 0.3);
    }

    /* Confidence bar */
    .confidence-bar {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #06A77D 0%, #00d4aa 100%);
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 0 8px rgba(6, 167, 125, 0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: white;
    }

    /* Section header */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2E86AB;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #F18F01;
        letter-spacing: -0.5px;
    }

    /* Image preview card */
    .image-preview-card {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.95rem;
        border-top: 1px solid #e0e0e0;
    }

    .footer-text {
        margin: 0.3rem 0;
    }

    /* Info box */
    .info-box {
        background: rgba(46, 134, 171, 0.08);
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Success message */
    .success-message {
        background: rgba(6, 167, 125, 0.1);
        border-left: 4px solid #06A77D;
        color: #0d4d2f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        .prediction-digit {
            font-size: 4rem;
        }
        .metrics-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model_cached():
    """Load and cache the trained model"""
    return load_trained_model()

model = load_model_cached()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="custom-card" style="margin-bottom: 1.5rem;">
        <h3 style="color: #2E86AB; margin-top: 0;">📋 Project Overview</h3>
        <p style="color: #555; line-height: 1.6; margin-bottom: 0;">
            A machine learning classification system using Logistic Regression 
            to recognize handwritten digits with high accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="margin: 1.5rem 0;">
        <div style="margin-bottom: 1.2rem;">
            <p style="font-size: 0.85rem; color: #999; text-transform: uppercase; 
                     letter-spacing: 1px; margin: 0.3rem 0; font-weight: 600;">
                📊 Dataset
            </p>
            <p style="color: #333; margin: 0.2rem 0; font-size: 1rem;">
                MNIST Digits Dataset
            </p>
            <p style="color: #999; font-size: 0.85rem; margin: 0;">1,797 training samples</p>
        </div>
        
        <div style="margin-bottom: 1.2rem;">
            <p style="font-size: 0.85rem; color: #999; text-transform: uppercase; 
                     letter-spacing: 1px; margin: 0.3rem 0; font-weight: 600;">
                🤖 Algorithm
            </p>
            <p style="color: #333; margin: 0.2rem 0; font-size: 1rem;">
                Logistic Regression
            </p>
            <p style="color: #999; font-size: 0.85rem; margin: 0;">Multinomial classifier</p>
        </div>
        
        <div style="margin-bottom: 1.2rem;">
            <p style="font-size: 0.85rem; color: #999; text-transform: uppercase; 
                     letter-spacing: 1px; margin: 0.3rem 0; font-weight: 600;">
                👤 Author
            </p>
            <p style="color: #333; margin: 0.2rem 0; font-size: 1rem;">
                [Your Name Here]
            </p>
        </div>
        
        <div>
            <p style="font-size: 0.85rem; color: #999; text-transform: uppercase; 
                     letter-spacing: 1px; margin: 0.3rem 0; font-weight: 600;">
                🎓 University
            </p>
            <p style="color: #333; margin: 0.2rem 0; font-size: 1rem;">
                [Your University]
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Handwritten Digit Recognition</h1>
    <p class="header-subtitle">Deep learning classification with Logistic Regression • Interactive Demo</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# METRICS SECTION
# ============================================================================
st.markdown('<h3 class="section-header">📊 Model Performance</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")

metrics_data = [
    ("Accuracy", "95.56%", "Testing accuracy on 360 samples"),
    ("Precision", "95.53%", "Weighted average across all classes"),
    ("Recall", "95.56%", "Weighted average across all classes"),
    ("F1-Score", "95.51%", "Harmonic mean of precision and recall"),
]

with col1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-value">95.56%</div>
        <div class="metric-label">Accuracy</div>
        <p style="font-size: 0.75rem; color: #999; margin: 0.5rem 0 0 0;">Testing accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-value">95.53%</div>
        <div class="metric-label">Precision</div>
        <p style="font-size: 0.75rem; color: #999; margin: 0.5rem 0 0 0;">Weighted avg</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-value">95.56%</div>
        <div class="metric-label">Recall</div>
        <p style="font-size: 0.75rem; color: #999; margin: 0.5rem 0 0 0;">Weighted avg</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-value">95.51%</div>
        <div class="metric-label">F1-Score</div>
        <p style="font-size: 0.75rem; color: #999; margin: 0.5rem 0 0 0;">Harmonic mean</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

# ============================================================================
# MAIN PREDICTION INTERFACE
# ============================================================================
st.markdown('<h3 class="section-header">🔍 Make a Prediction</h3>', unsafe_allow_html=True)

col_upload, col_predict = st.columns([1.2, 1], gap="large")

with col_upload:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('**Upload Image**')
    
    uploaded_file = st.file_uploader(
        label="Upload a handwritten digit image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Supported formats: PNG, JPG, JPEG. Recommended: 28x28 px or larger"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Image")
        
        with st.expander("📋 Image Details"):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(f"**Size:** {image.size}")
                st.write(f"**Format:** {image.format}")
            with col_info2:
                st.write(f"**Mode:** {image.mode}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #999;">
            <p style="font-size: 3rem; margin: 0;">📁</p>
            <p style="margin-top: 1rem;">Click or drag image here</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_predict:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('**Prediction**')
    
    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            <p style="margin: 0; font-size: 0.95rem;">
                👈 Upload an image on the left to get started
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🚀 Predict Digit", use_container_width=True):
            with st.spinner("🔄 Analyzing image..."):
                try:
                    # Preprocess image
                    preprocessed = preprocess_image(image)
                    
                    # Make prediction
                    prediction, confidence, probabilities = predict_digit(model, preprocessed)
                    
                    # Store in session state
                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.session_state.probabilities = probabilities
                    st.session_state.prediction_made = True
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.prediction_made = False
        
        # Display prediction results
        if 'prediction_made' in st.session_state and st.session_state.prediction_made:
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            
            st.markdown(f"""
            <div class="prediction-result-card">
                <p class="prediction-label">Predicted Digit</p>
                <div class="prediction-digit">{prediction}</div>
                <p style="margin: 1rem 0 0 0; font-size: 1.1rem; font-weight: 500;">
                    Confidence: {int(confidence*100)}%
                </p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {int(confidence*100)}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

# ============================================================================
# PROBABILITY DISTRIBUTION
# ============================================================================
if 'probabilities' in st.session_state:
    st.markdown('<h3 class="section-header">📈 Probability Distribution</h3>', unsafe_allow_html=True)
    
    probabilities = st.session_state.probabilities
    prediction = st.session_state.prediction
    
    # Create columns for probability table and chart
    col_table, col_chart = st.columns([0.8, 1.2], gap="large")
    
    with col_table:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('**Confidence by Digit**')
        
        # Create probability data
        prob_list = []
        for digit in range(10):
            prob_list.append({
                'Digit': digit,
                'Probability': f"{probabilities[digit]:.4f}",
                'Percentage': f"{probabilities[digit]*100:.1f}%"
            })
        
        # Display as dataframe
        st.dataframe(prob_list, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#06A77D' if i == prediction else '#B0B8C1' for i in range(10)]
        
        bars = ax.bar(
            range(10),
            probabilities,
            color=colors,
            alpha=0.85,
            edgecolor='white',
            linewidth=2
        )
        
        ax.set_xlabel('Digit', fontsize=11, fontweight='600', color='#333')
        ax.set_ylabel('Probability', fontsize=11, fontweight='600', color='#333')
        ax.set_xticks(range(10))
        ax.set_ylim(0, max(probabilities) * 1.15)
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ddd')
        ax.spines['bottom'].set_color('#ddd')
        
        # Add value labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='600', color='#333'
            )
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
st.markdown('<h3 class="section-header">🔍 Model Evaluation</h3>', unsafe_allow_html=True)

confusion_matrix_data = np.array([
    [35, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 30, 0, 1, 0, 1, 1, 0, 1, 2],
    [0, 0, 35, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 37, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 36, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 37, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 34, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 36, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 1, 31, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 1, 33]
])

col_cm1, col_cm2 = st.columns([1, 1.2], gap="large")

with col_cm1:
    st.markdown("""
    <div class="custom-card">
        <p style="margin-top: 0; color: #2E86AB; font-weight: 600;">Confusion Matrix Info</p>
        <p style="color: #555; line-height: 1.7; margin-bottom: 1rem;">
            This matrix shows how well the model classifies each digit. 
            Diagonal elements represent correct predictions, while off-diagonal elements show misclassifications.
        </p>
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
            <p style="margin: 0.3rem 0; color: #333; font-size: 0.95rem;">
                <strong>Total Test Samples:</strong> 360
            </p>
            <p style="margin: 0.3rem 0; color: #333; font-size: 0.95rem;">
                <strong>Correct Predictions:</strong> 344
            </p>
            <p style="margin: 0.3rem 0; color: #333; font-size: 0.95rem;">
                <strong>Accuracy Rate:</strong> 95.56%
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_cm2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
        cbar_kws={'label': 'Count', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white'
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='600', color='#333')
    ax.set_ylabel('True Label', fontsize=12, fontweight='600', color='#333')
    ax.set_title('Confusion Matrix - Test Set', fontsize=13, fontweight='600', color='#333', pad=15)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p class="footer-text"><strong>🎓 University Machine Learning Project</strong></p>
    <p class="footer-text">Handwritten Digit Recognition using Logistic Regression</p>
    <p class="footer-text">Built with Python • scikit-learn • Streamlit</p>
    <p class="footer-text" style="margin-top: 1rem; font-size: 0.85rem; color: #999;">
        © 2026 Machine Learning Demonstration
    </p>
</div>
""", unsafe_allow_html=True)