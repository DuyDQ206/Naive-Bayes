import streamlit as st
import joblib
import os

# Set up trang UI
st.set_page_config(page_title="Naive Bayes Classifier UI", page_icon="🤖", layout="centered")

# Hàm load các file model từ folder models/
@st.cache_resource
def load_models():
    # Đường dẫn tới thư mục models
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "models")
    
    # Load model & vectorizer cho SMS và IMDb
    sms_model = joblib.load(os.path.join(models_dir, "sms_model.pkl"))
    sms_vec = joblib.load(os.path.join(models_dir, "sms_vectorizer.pkl"))
    
    imdb_model = joblib.load(os.path.join(models_dir, "imdb_model.pkl"))
    imdb_vec = joblib.load(os.path.join(models_dir, "imdb_vectorizer.pkl"))
    
    return sms_model, sms_vec, imdb_model, imdb_vec

try:
    sms_model, sms_vec, imdb_model, imdb_vec = load_models()
except Exception as e:
    st.error(f"Lỗi khi tải file mô hình từ folder models/: {e}")

# Sidebar chọn bài toán
st.sidebar.title("📌 Chọn Bài Toán")
task = st.sidebar.radio("Chuyển đổi bài toán:", ("📩 SMS Spam Classifier", "🎬 IMDb Sentiment Analysis"))

st.title("🤖 Naive Bayes Text Classification")

if task == "📩 SMS Spam Classifier":
    st.subheader("📩 Phân loại Tin nhắn SMS (Spam / Ham)")
    
    # Lấy văn bản từ người dùng
    user_input = st.text_area("Nhập nội dung tin nhắn SMS:", placeholder="Gõ tin nhắn vào đây...", height=120)
    
    if st.button("Dự đoán SMS", type="primary"):
        if user_input.strip():
            # Biến đổi văn bản & dự đoán
            vec_input = sms_vec.transform([user_input])
            pred = sms_model.predict(vec_input)[0]
            prob = sms_model.predict_proba(vec_input)[0]
            confidence = max(prob) * 100
            
            if str(pred).lower() in ['1', 'spam']:
                st.error(f"🚨 Kết quả: **SPAM (Rác)** | Độ tin cậy: {confidence:.2f}%")
            else:
                st.success(f"✅ Kết quả: **HAM (Bình thường)** | Độ tin cậy: {confidence:.2f}%")
        else:
            st.warning("Vui lòng nhập văn bản.")

else:
    st.subheader("🎬 Phân tích Cảm xúc IMDb Movie Review")
    
    user_input = st.text_area("Nhập đánh giá phim (Tiếng Anh):", placeholder="Type your review here...", height=120)
    
    if st.button("Dự đoán Sentiment", type="primary"):
        if user_input.strip():
            vec_input = imdb_vec.transform([user_input])
            pred = imdb_model.predict(vec_input)[0]
            prob = imdb_model.predict_proba(vec_input)[0]
            confidence = max(prob) * 100
            
            if str(pred).lower() in ['1', 'pos', 'positive']:
                st.success(f"😊 Kết quả: **POSITIVE (Tích cực)** | Độ tin cậy: {confidence:.2f}%")
            else:
                st.error(f"😞 Kết quả: **NEGATIVE (Tiêu cực)** | Độ tin cậy: {confidence:.2f}%")
        else:
            st.warning("Vui lòng nhập văn bản.")