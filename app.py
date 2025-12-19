import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Zytra Analytics", layout="wide")

# -----------------------------
# Global UI Styling
# -----------------------------
st.markdown("""
<style>
    .main { padding: 2rem; }
    h1, h2, h3 { color: #1f2937; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Fake database (Email + Username UNIQUE)
# -----------------------------
if "users" not in st.session_state:
    st.session_state.users = {
        "admin@zytra.com": {
            "username": "admin",
            "password": "zytra123"
        }
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "reset_stage" not in st.session_state:
    st.session_state.reset_stage = None

# -----------------------------
# Helpers
# -----------------------------
def username_exists(username):
    return any(user["username"] == username for user in st.session_state.users.values())

# -----------------------------
# Authentication
# -----------------------------
def login():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Login to Zytra Analytics")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            if email in st.session_state.users and st.session_state.users[email]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.current_user = email
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid email or password")

    with col2:
        if st.button("Forgot Password"):
            st.session_state.reset_stage = "request"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def signup():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Create an Account")

    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        if email in st.session_state.users:
            st.error("Email already registered")
        elif username_exists(username):
            st.error("Username already taken")
        elif not email or not username or not password:
            st.error("Please fill all fields")
        else:
            st.session_state.users[email] = {
                "username": username,
                "password": password
            }
            st.success("Account created successfully. Please login.")

    st.markdown('</div>', unsafe_allow_html=True)

def forgot_password():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Reset Password")

    email = st.text_input("Registered Email")

    if st.button("Verify Email"):
        if email in st.session_state.users:
            st.session_state.reset_email = email
            st.session_state.reset_stage = "reset"
            st.success("Email verified. Set new password.")
            st.rerun()
        else:
            st.error("Email not found")

    st.markdown('</div>', unsafe_allow_html=True)

def reset_password():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Set New Password")

    new_pass = st.text_input("New Password", type="password")

    if st.button("Update Password"):
        st.session_state.users[st.session_state.reset_email]["password"] = new_pass
        st.session_state.reset_stage = None
        st.success("Password updated successfully. Please login.")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Forecasting helpers
# -----------------------------
def run_forecast(df, date_col, value_col, periods):
    df = df[[date_col, value_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    sarima = SARIMAX(df[value_col], order=(1,1,1), seasonal_order=(1,1,1,7))
    sarima_fit = sarima.fit(disp=False)
    sarima_fc = sarima_fit.forecast(periods)

    prophet_df = df.rename(columns={date_col:"ds", value_col:"y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    prophet_fc = model.predict(future)

    return df, sarima_fc, prophet_fc

def business_insights(hist_df, forecast, value_col):
    avg = hist_df[value_col].mean()
    peak = hist_df[value_col].max()
    f_avg = forecast.mean()

    insights = [
        f"Average historical demand: {avg:.2f}",
        f"Peak historical demand: {peak:.2f}",
        f"Forecasted average demand: {f_avg:.2f}",
        "Demand trend is increasing" if f_avg > avg else "Demand trend is decreasing"
    ]

    if f_avg > peak:
        insights.append("⚠️ High stock-out risk if inventory is not adjusted")
    else:
        insights.append("✅ Inventory levels appear sufficient")

    return insights

# -----------------------------
# MAIN UI
# -----------------------------
st.markdown("""
<h1>📊 Zytra Analytics</h1>
<p style="font-size:18px; color:#4b5563;">
Automated Demand Forecasting & Business Intelligence Platform
</p>
<hr>
""", unsafe_allow_html=True)

if not st.session_state.logged_in:

    if st.session_state.reset_stage == "request":
        forgot_password()
    elif st.session_state.reset_stage == "reset":
        reset_password()
    else:
        choice = st.radio("Select option", ["Login", "Sign Up"])
        login() if choice == "Login" else signup()

else:
    st.sidebar.title("Zytra Analytics")
    st.sidebar.markdown("Founder-led Analytics Platform")
    st.sidebar.divider()
    st.sidebar.success(f"Logged in as {st.session_state.current_user}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📂 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        file_names = [f.name for f in uploaded_files]
        selected = st.selectbox("Select file to analyze", file_names)
        file = next(f for f in uploaded_files if f.name == selected)
        df = pd.read_csv(file)

        st.subheader("Preview")
        st.dataframe(df.head())
        st.markdown('</div>', unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        date_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if numeric_cols:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Visual Analytics")
            metric = st.selectbox("Select metric", numeric_cols)

            if st.button("Generate Chart"):
                if date_cols:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                    st.line_chart(df.set_index(date_cols[0])[metric])
                else:
                    st.bar_chart(df[metric])
            st.markdown('</div>', unsafe_allow_html=True)

        if numeric_cols and date_cols:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔮 Demand Forecasting")

            date_col = st.selectbox("Date column", date_cols)
            value_col = st.selectbox("Value column", numeric_cols)
            periods = st.slider("Forecast horizon (days)", 7, 60, 30)

            if st.button("Generate Forecast"):
                hist, sarima_fc, prophet_fc = run_forecast(df, date_col, value_col, periods)

                fig, ax = plt.subplots()
                ax.plot(hist[date_col], hist[value_col], label="Historical")
                ax.plot(pd.date_range(hist[date_col].iloc[-1], periods=periods+1, freq="D")[1:], sarima_fc, label="SARIMA")
                ax.plot(prophet_fc["ds"], prophet_fc["yhat"], label="Prophet")
                ax.legend()
                st.pyplot(fig)

                insights = business_insights(hist, sarima_fc, value_col)
                for i in insights:
                    st.write("•", i)

                report = "Zytra Analytics Report\n\n" + "\n".join(insights)
                st.download_button("📥 Download Report", report, "zytra_report.txt")

            st.markdown('</div>', unsafe_allow_html=True)
