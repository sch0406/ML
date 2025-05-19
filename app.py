
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 07:53:38 2025
@author: user
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. 資料載入與基本顯示
# ===============================
st.title("運動表現資料分析")

# 狀態訊息
st.success('分析環境載入成功 ✅')
st.info("請使用側邊欄進行篩選與互動分析", icon='ℹ️')
st.error('This is an error', icon="🚨")

# 載入資料
df = pd.read_csv("bodyPerformance.csv")
df["BMI"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

# 顯示部分資料
st.header("原始資料預覽")
st.dataframe(df.head(50))

# ===============================
# 2. 側欄條件篩選
# ===============================
st.sidebar.header("🔎 資料篩選器")
age_range = st.sidebar.slider("年齡範圍", 10, 80, (20, 50))
gender = st.sidebar.selectbox("性別", ["All", "M", "F"])
bmi_range = st.sidebar.slider("BMI 範圍", 10.0, 40.0, (18.5, 24.0))

# 篩選資料
filtered_df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) &
                 (df["BMI"] >= bmi_range[0]) & (df["BMI"] <= bmi_range[1])]
if gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender]

st.subheader("篩選後的資料")
st.dataframe(filtered_df)

# ===============================
# 3. 統計摘要與欄位最大/最小值
# ===============================
st.header("統計摘要")
st.write(filtered_df.describe())

st.subheader("欄位最大/最小值")
for label, col in {
    "年齡 (age)": "age",
    "BMI": "BMI",
    "仰臥起坐 (sit-ups counts)": "sit-ups counts",
    "立定跳遠 (broad jump_cm)": "broad jump_cm"
}.items():
    if col in filtered_df.columns:
        st.write(f"{label} ➤ 最小：{filtered_df[col].min():.2f}，最大：{filtered_df[col].max():.2f}")

# ===============================
# 4. 三種圖表：箱型圖、散佈圖、雷達圖
# ===============================
st.header("互動式圖表分析")
tab1, tab2, tab3 = st.tabs(["📦 箱型圖", "⚫ 散佈圖", "📊 直方圖"])

with tab1:
    fig1 = px.box(filtered_df, x="gender", y="sit-ups counts", title="性別與仰臥起坐")
    st.plotly_chart(fig1)

with tab2:
    fig2 = px.scatter(filtered_df, x="age", y="broad jump_cm", color="gender", title="年齡與立定跳遠")
    st.plotly_chart(fig2)

with tab3:
    if filtered_df.empty:
        st.warning("⚠️ 篩選後無資料可供圖表分析，請調整側欄條件")
    else:
        bar_df = filtered_df.dropna(subset=["BMI", "sit-ups counts", "broad jump_cm"])
        if bar_df.empty:
            st.warning("⚠️ 欄位含缺值，請放寬篩選條件或填補缺失資料")
        else:
            avg_df = bar_df.groupby("gender")[["BMI", "sit-ups counts", "broad jump_cm"]].mean().reset_index()
            avg_df_melted = avg_df.melt(id_vars="gender", var_name="指標", value_name="平均值")
            fig_bar = px.bar(avg_df_melted, x="指標", y="平均值", color="gender", barmode="group",
                             title="性別平均表現直方圖")
            st.plotly_chart(fig_bar)

# ===============================
# 5. 模型訓練與預測：坐姿體前彎
# ===============================
st.header("🎯 線性迴歸模型：預測坐姿體前彎")

model_df = df[["age", "height_cm", "weight_kg", "sit and bend forward_cm"]].dropna()
X = model_df[["age", "height_cm", "weight_kg"]]
y = model_df["sit and bend forward_cm"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)

st.write(f"模型準確度 R²：{score:.2f}")

fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': '實際值', 'y': '預測值'},
                      title="實際值 vs 預測值（坐姿體前彎）")
fig_pred.add_shape(
    type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
    line=dict(color='red', dash='dash')
)
st.plotly_chart(fig_pred)

# ===============================
# 6. 使用者輸入進行預測
# ===============================
st.subheader("🔍 輸入資料 → 預測坐姿體前彎距離")

input_age = st.number_input("年齡", min_value=10, max_value=100, value=25)
input_height = st.number_input("身高 (cm)", min_value=100.0, max_value=250.0, value=170.0)
input_weight = st.number_input("體重 (kg)", min_value=30.0, max_value=200.0, value=60.0)

if st.button("預測"):
    input_data = pd.DataFrame([[input_age, input_height, input_weight]],
                              columns=["age", "height_cm", "weight_kg"])
    pred = model.predict(input_data)[0]
    st.success(f"🌟 預測坐姿體前彎距離為：{pred:.2f} cm")
