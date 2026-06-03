import streamlit as st
import numpy as np
from PIL import Image
from utils import preprocess_image, load_trained_model, predict_digit

# Load the trained model
model = load_trained_model()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
    }
    .prediction-digit {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: #4CAF50;
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .prob-container {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
    }
    .digit-label {
        font-weight: bold;
        margin-right: 1rem;
        min-width: 30px;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #dee2e6;
    }
    .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.title("📊 About the Project")
    st.markdown("""
    **Machine Learning Project**  
    *Handwritten Digit Recognition*

    **Algorithm:** Logistic Regression  
    **Dataset:** MNIST Digits (8x8)  
    **Accuracy:** 95.56%

    **Features:**
    - Real-time prediction
    - Confidence scoring
    - Probability distribution
    - Image preprocessing
    """)

    st.markdown("---")
    st.markdown("**Model Performance:**")
    metrics = {
        "Accuracy": "95.56%",
        "Precision": "95.53%",
        "Recall": "95.56%",
        "F1-Score": "95.51%"
    }

    for metric, value in metrics.items():
        st.metric(metric, value)

    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("🐍 Python | 📈 scikit-learn | 🎨 Streamlit | 🖼️ PIL")

    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">🔢 Handwritten Digit Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6c757d; margin-bottom: 2rem;">Upload an image of a handwritten digit and watch our AI predict it instantly!</p>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Your Digit Image")
uploaded_file = st.file_uploader(
    "Choose an image file (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image of a handwritten digit (0-9) for best results"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Create two columns for image and results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption='Your Uploaded Image', use_column_width=True)

        # Show preprocessing info
        with st.expander("🔧 Preprocessing Details"):
            st.markdown("""
            **What happens to your image:**
            1. Convert to grayscale
            2. Resize to 8×8 pixels
            3. Normalize pixel values (0-1)
            4. Flatten to 64 features
            """)

    with col2:
        # Preprocess the image
        preprocessed = preprocess_image(image)

        # Make prediction
        prediction, confidence, probabilities = predict_digit(model, preprocessed)

        # Prediction display
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Result")
        st.markdown(f'<div class="prediction-digit">{prediction}</div>', unsafe_allow_html=True)

        # Confidence bar
        st.markdown("**Confidence Level:**")
        confidence_percentage = int(confidence * 100)
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_percentage}%"></div>
        </div>
        <p style="text-align: center; margin: 0.5rem 0;">{confidence_percentage}% confident</p>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Probability distribution
        st.markdown("### 📊 Probability Distribution")
        st.markdown("How confident is the model for each digit?")

        # Sort probabilities for better visualization
        sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

        for digit, prob in sorted_probs:
            prob_percentage = prob * 100
            is_prediction = digit == prediction

            # Color coding
            if is_prediction:
                color = "#4CAF50"  # Green for prediction
                bg_color = "rgba(76, 175, 80, 0.1)"
            elif prob > 0.1:
                color = "#FF9800"  # Orange for significant probability
                bg_color = "rgba(255, 152, 0, 0.1)"
            else:
                color = "#9E9E9E"  # Gray for low probability
                bg_color = "rgba(158, 158, 158, 0.1)"

            st.markdown(f"""
            <div class="prob-container" style="background: {bg_color}; border-left: 4px solid {color};">
                <div class="digit-label" style="color: {color};">{digit}</div>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span style="font-size: 0.9rem;">{prob_percentage:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.3); border-radius: 4px; height: 8px;">
                        <div style="background: {color}; height: 100%; border-radius: 4px; width: {prob_percentage}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Welcome message when no image uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin: 2rem 0;">
        <h2>👋 Welcome!</h2>
        <p style="font-size: 1.2rem;">Upload a handwritten digit image above to get started with AI-powered recognition.</p>
        <p style="font-size: 1rem; opacity: 0.9;">Our model can recognize digits from 0 to 9 with 95.56% accuracy!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sample images section
    st.markdown("### 💡 Tips for Best Results")
    tips_col1, tips_col2 = st.columns(2)
    with tips_col1:
        st.markdown("""
        ✅ **Good practices:**
        - Use clear, handwritten digits
        - Center the digit in the image
        - Avoid multiple digits
        - Use high contrast
        """)
    with tips_col2:
        st.markdown("""
        ❌ **Avoid:**
        - Blurry or distorted images
        - Very small or large digits
        - Background noise
        - Slanted or rotated digits
        """)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
**University Machine Learning Project**  
*Logistic Regression for Digit Classification*  
Built with ❤️ using Python, scikit-learn, and Streamlit
""")
st.markdown('</div>', unsafe_allow_html=True)