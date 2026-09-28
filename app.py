import streamlit as st
import joblib
import os

# 1. Cấu hình trang UI dạng Rộng (Wide) để bố cục được thoáng đẹp
st.set_page_config(
    page_title="Naive Bayes NLP Classifier",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS tinh chỉnh giao diện
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #4F8BF9; text-align: center; margin-bottom: 0px; }
    .sub-title { font-size: 1rem; color: #888; text-align: center; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# 2. Tải mô hình từ thư mục models/
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "models")
    
    sms_model = joblib.load(os.path.join(models_dir, "sms_model.pkl"))
    sms_vec = joblib.load(os.path.join(models_dir, "sms_vectorizer.pkl"))
    
    imdb_model = joblib.load(os.path.join(models_dir, "imdb_model.pkl"))
    imdb_vec = joblib.load(os.path.join(models_dir, "imdb_vectorizer.pkl"))
    
    return sms_model, sms_vec, imdb_model, imdb_vec

try:
    sms_model, sms_vec, imdb_model, imdb_vec = load_models()
except Exception as e:
    st.error(f"⚠️ Lỗi khi tải các file mô hình từ thư mục models/: {e}")

# 3. Sidebar chuyển đổi bài toán
st.sidebar.title("📌 Chọn Bài Toán")
task = st.sidebar.radio(
    "Chuyển đổi ứng dụng:",
    ("📩 SMS Spam Classifier", "🎬 IMDb Sentiment Analysis")
)

st.sidebar.divider()
st.sidebar.info(
    "💡 **Mô hình Naive Bayes** thích hợp cho các bài toán phân loại văn bản nhờ khả năng tính toán xác suất độc lập giữa các từ nhanh chóng và chính xác."
)

# Tiêu đề ứng dụng
st.markdown("<h1 class='main-title'>🤖 Naive Bayes Text Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Mini Project: Ứng dụng Naive Bayes cho SMS Spam & IMDb Movie Review</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BÀI TOÁN 1: SMS SPAM
# -----------------------------------------------------------------------------
if task == "📩 SMS Spam Classifier":
    st.subheader("📩 Phân loại Tin nhắn SMS (Spam / Ham)")
    
    # Session state quản lý nội dung ô input
    if "sms_input" not in st.session_state:
        st.session_state.sms_input = ""

    st.markdown("**💡 Dùng thử các mẫu tin nhắn nhanh:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    if col_ex1.button("📩 Mẫu SMS Spam 1"):
        st.session_state.sms_input = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461."
    if col_ex2.button("📩 Mẫu SMS Spam 2"):
        st.session_state.sms_input = "FREE entry in a £1000 weekly competition! Text FA to 87121 to receive entry."
    if col_ex3.button("✅ Mẫu SMS Bình thường"):
        st.session_state.sms_input = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."

    user_input = st.text_area("Nhập nội dung tin nhắn SMS:", value=st.session_state.sms_input, height=120)
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        submit = st.button("🔍 Phân loại Tin Nhắn", type="primary")
    with col_btn2:
        if st.button("🗑️ Xóa nội dung"):
            st.session_state.sms_input = ""
            st.rerun()

    if submit:
        if user_input.strip():
            # Biến đổi văn bản & Dự đoán
            vec_input = sms_vec.transform([user_input])
            pred = sms_model.predict(vec_input)[0]
            probs = sms_model.predict_proba(vec_input)[0]
            
            # Xác định vị trí nhãn
            classes = list(sms_model.classes_)
            spam_idx = classes.index("spam") if "spam" in classes else 1
            spam_prob = probs[spam_idx]
            ham_prob = 1.0 - spam_prob
            
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            
            is_spam = str(pred).lower() in ['1', 'spam']
            
            with res_col1:
                if is_spam:
                    st.error("### 🚨 Kết quả: SPAM")
                    st.metric("Xác suất Spam", f"{spam_prob*100:.2f}%")
                else:
                    st.success("### ✅ Kết quả: HAM (Bình thường)")
                    st.metric("Xác suất Bình thường", f"{ham_prob*100:.2f}%")

            with res_col2:
                st.write("**Chi tiết phân bố xác suất của Naive Bayes:**")
                st.write(f"🟢 **HAM (Bình thường):** {ham_prob*100:.2f}%")
                st.progress(float(ham_prob))
                st.write(f"🔴 **SPAM (Rác):** {spam_prob*100:.2f}%")
                st.progress(float(spam_prob))
        else:
            st.warning("⚠️ Vui lòng nhập nội dung tin nhắn.")

# -----------------------------------------------------------------------------
# BÀI TOÁN 2: IMDB SENTIMENT
# -----------------------------------------------------------------------------
else:
    st.subheader("🎬 Phân tích Cảm xúc IMDb Movie Review")
    
    if "imdb_input" not in st.session_state:
        st.session_state.imdb_input = ""

    st.markdown("**💡 Dùng thử các mẫu đánh giá phim:**")
    col_ex1, col_ex2 = st.columns(2)
    
    if col_ex1.button("😊 Mẫu Review Tích cực"):
        st.session_state.imdb_input = "This movie was absolutely fantastic! The story was well-crafted and acting was brilliant."
    if col_ex2.button("😞 Mẫu Review Tiêu cực"):
        st.session_state.imdb_input = "A complete waste of time. Terrible acting, boring plot, and poor direction."

    user_input = st.text_area("Nhập đoạn đánh giá phim (Tiếng Anh):", value=st.session_state.imdb_input, height=120)
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        submit = st.button("🎬 Phân tích Cảm Xúc", type="primary")
    with col_btn2:
        if st.button("🗑️ Xóa nội dung"):
            st.session_state.imdb_input = ""
            st.rerun()

    if submit:
        if user_input.strip():
            # Biến đổi văn bản & Dự đoán
            vec_input = imdb_vec.transform([user_input])
            pred = imdb_model.predict(vec_input)[0]
            probs = imdb_model.predict_proba(vec_input)[0]
            
            classes = list(imdb_model.classes_)
            pos_idx = classes.index("positive") if "positive" in classes else (classes.index("pos") if "pos" in classes else 1)
            pos_prob = probs[pos_idx]
            neg_prob = 1.0 - pos_prob
            
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            
            is_pos = str(pred).lower() in ['1', 'pos', 'positive']
            
            with res_col1:
                if is_pos:
                    st.success("### 😊 Kết quả: POSITIVE")
                    st.metric("Độ Tích Cực", f"{pos_prob*100:.2f}%")
                else:
                    st.error("### 😞 Kết quả: NEGATIVE")
                    st.metric("Độ Tiêu Cực", f"{neg_prob*100:.2f}%")

            with res_col2:
                st.write("**Chi tiết độ phân cực cảm xúc (Sentiment):**")
                st.write(f"😊 **Positive (Tích cực):** {pos_prob*100:.2f}%")
                st.progress(float(pos_prob))
                st.write(f"😞 **Negative (Tiêu cực):** {neg_prob*100:.2f}%")
                st.progress(float(neg_prob))
        else:
            st.warning("⚠️ Vui lòng nhập đoạn đánh giá phim.")