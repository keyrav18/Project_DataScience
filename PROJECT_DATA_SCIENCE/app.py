
# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Evaluasi Kesehatan Finansial BAZNAS",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard XAI: Evaluasi Kesehatan Finansial UPZ BAZNAS")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Regresi Linear)** yang dijelaskan dengan **XAI (SHAP)** untuk mengevaluasi faktor penentu kesehatan finansial (ACR) pada Unit Pengumpul Zakat.
""")

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
@st.cache_data
def load_data():
    # Data hardcoded agar deploy stabil tanpa upload file manual
    csv_data = """Wilayah,Zakat_Maal_2022,Zakat_Maal_2023,Zakat_Fitrah_2022,Zakat_Fitrah_2023,Infak_2022,Infak_2023,DSKL_2022,DSKL_2023,Penyaluran_2022,Penyaluran_2023,Total_Pengumpulan_2022,Total_Pengumpulan_2023,Pertumbuhan_Pengumpulan_Persen,ACR_2022_Persen,ACR_2023_Persen,Status_Kesehatan_2022,Status_Kesehatan_2023,Dominasi_Dana_2023,Region
BAZNAS (Pusat),536640868868,645538065987,9656466631,11190355084,70230370082,306840514386,17340431740,14587006023,584585478559,675091571281,633868137321,978155941480,54.32,92.23,69.02,Sangat Efektif,Cukup Efektif,Maal,Pusat
Nanggroe Aceh Darussalam,132179974814,175481487955,0,54641000,63389684576,65804768917,2135000,302850225,201380315240,285407294532,195571794390,241643748097,23.56,102.97,118.11,Sangat Efektif,Sangat Efektif,Maal,Sumatera
Sumatera Utara,44500220894,46081566795,1882605609,250000,8113900479,11266200601,351248495,75131575,47609321584,69354175976,54847975477,57423148971,4.70,86.80,120.78,Efektif,Sangat Efektif,Maal,Sumatera
Sumatera Barat,141694155762,157252063149,755179053,811192696,6294407988,11320132309,422239264,585341011,144235401916,160258159055,149165982067,169968729165,13.95,96.69,94.29,Sangat Efektif,Sangat Efektif,Maal,Sumatera
Riau,128610250208,177673390086,1167317750,2435253525,5265585178,24879028110,1486082071,12645867613,124912647827,187062391710,136529235207,217633539334,59.40,91.49,85.95,Sangat Efektif,Efektif,Maal,Sumatera
Jambi,46538470318,53007775398,148520054,256275563,4420376478,11873732946,947085585,1256268736,43083133946,56848386186,52054452435,66394052643,27.55,82.77,85.62,Efektif,Efektif,Maal,Sumatera
Sumatera Selatan,34363655000,34209334592,159894500,268191815,9992886703,14917190435,33761135,854468000,39658172423,44508619914,44550197338,50249184842,12.79,89.02,88.58,Efektif,Efektif,Maal,Sumatera
Bengkulu,12621231661,21493770030,37416000,15865000,1761276164,4989307453,2405000,0,11141353108,54524349098,14422328825,26498942483,83.74,77.25,205.76,Efektif,Sangat Efektif,Maal,Sumatera
Lampung,15246897755,13796286995,2747178860,2744940142,9563945918,14301866842,104150000,860017000,22055713208,28293096462,27662172533,31703110979,14.61,79.73,89.24,Efektif,Efektif,Infak,Sumatera
Kep. Bangka Belitung,23044166386,20140238837,5509000,8590000,1990344741,2556619396,6190000,96777057,25332936688,32789724894,25046210127,22802225290,-8.96,101.14,143.80,Sangat Efektif,Sangat Efektif,Maal,Sumatera
Kep. Riau,30856933375,35892148234,1230276978,943229584,5862316692,9053121173,1306781791,2091367475,40821876781,39777621736,39256308836,47979866466,22.22,103.99,82.90,Sangat Efektif,Efektif,Maal,Sumatera
DKI Jakarta,150008330371,166143247637,1344659473,4517126509,55892493489,16279550342,8015723004,3815286223,220507248640,282865282826,215261206337,190755210711,-11.38,102.44,148.29,Sangat Efektif,Sangat Efektif,Maal,Jawa
Jawa Barat,311381496792,318261998028,40754077576,61384093769,105815315544,175418639336,36101910702,28961600328,442843169308,505399360206,494052800614,584026331461,18.21,89.63,86.54,Efektif,Efektif,Maal,Jawa
Jawa Tengah,315101601583,320794738465,15994459789,2737038874,87677100361,103829237226,11477331348,13686762076,369403190882,401241609092,430250493081,441047776641,2.51,85.86,90.97,Efektif,Sangat Efektif,Maal,Jawa
DI Yogyakarta,41672106675,48891749288,192621357,631801703,18388620820,16017030286,1102192318,2046798696,59501035452,64134467762,61355541170,67587379973,10.16,96.98,94.89,Sangat Efektif,Sangat Efektif,Maal,Jawa
Jawa Timur,152562410376,156462488894,13158474937,9999263948,127782330488,175682290155,56050343617,63316254008,250148038803,376012418317,349553559418,405460297005,15.99,71.56,92.74,Efektif,Sangat Efektif,Infak,Jawa
Banten,76047215726,79485012670,7372063529,12839809896,29195229616,34755279871,11316464916,8991354613,111029316246,138053705478,123930973787,136071457050,9.80,89.59,101.46,Efektif,Sangat Efektif,Maal,Jawa
Bali,2464271252,2244275793,299928053,561757961,7781687987,12579650712,1443507906,1074104634,10509929519,13207709758,11989395198,16459789100,37.29,87.66,80.24,Efektif,Efektif,Infak,Bali_Nusa
Nusa Tenggara Barat,156670677717,89989009736,6677741957,10755796388,22681127707,33969391754,5812445004,3157615283,97138204917,127900397180,191841992385,137871813161,-28.13,50.63,92.77,Cukup Efektif,Sangat Efektif,Maal,Bali_Nusa
Nusa Tenggara Timur,0,225930081,0,180332630,0,315167500,0,700705000,0,2020398733,0,1422135211,0,0,142.07,Tidak Efektif,Sangat Efektif,DSKL,Bali_Nusa
Kalimantan Barat,8568441580,7855216050,1278226175,958577640,3775529696,5527357191,304084000,1882750831,8891156909,12757907324,13926281451,16223901712,16.50,63.84,78.64,Cukup Efektif,Efektif,Maal,Kalimantan
Kalimantan Tengah,2479359545,2374195211,1147414600,202289270,976458969,1565002890,225009700,3013503000,4153283258,5303078331,4828242814,7154990371,48.19,86.02,74.12,Efektif,Efektif,DSKL,Kalimantan
Kalimantan Selatan,17020359781,21281835457,842456210,2487246358,18326421591,39813451178,3505040500,5081561505,35581370660,68572652695,39694278082,68664094498,72.98,89.64,99.87,Efektif,Sangat Efektif,Infak,Kalimantan
Kalimantan Timur,33467743193,45393227168,2992432205,882854460,14712074347,20336221766,1905068910,2021711066,43119366426,63676196749,53077318655,68634014460,29.31,81.24,92.78,Efektif,Sangat Efektif,Maal,Kalimantan
Kalimantan Utara,4737192335,10691117944,4574412336,3276727000,836365205,3914992539,643196000,3605107500,9378033877,18390642228,10791165876,21487944983,99.13,86.90,85.59,Efektif,Efektif,Maal,Kalimantan
Sulawesi Utara,1695145556,26616704932,62043500,214115000,988834454,1759191775,234980500,846000000,1316627326,1461223705,2981004010,29436011707,887.45,44.17,4.96,Kurang Efektif,Tidak Efektif,Maal,Sulawesi
Sulawesi Tengah,4473421727,4507317187,68405500,89900500,5092972835,5533193891,80367500,164210700,8465456376,9574338817,9715167562,10294622278,5.96,87.14,93.00,Efektif,Sangat Efektif,Infak,Sulawesi
Sulawesi Selatan,45777961831,52827586721,36972312580,8394272965,45926226699,67201933961,5826887960,1693728500,104472991668,112222082903,134503389070,130117522147,-3.26,77.67,86.25,Efektif,Efektif,Infak,Sulawesi
Sulawesi Tenggara,14055789770,6725310424,1742869000,10527329100,5454835601,15999536670,3750000,11422487000,13872852542,44876060541,21257244371,44674663194,110.16,65.26,100.45,Cukup Efektif,Sangat Efektif,Infak,Sulawesi
Gorontalo,20319974071,18866052296,592159000,594198500,2855316811,7443140275,1050000,0,22101184573,28425278627,23768499882,26903391071,13.19,92.99,105.66,Sangat Efektif,Sangat Efektif,Maal,Sulawesi
Sulawesi Barat,3383360366,4464391782,4854353000,3222093650,5755106805,6496288985,67050000,1999419000,12156019955,11825461096,14059870171,16182193417,15.09,86.46,73.08,Efektif,Efektif,Infak,Sulawesi
Maluku,34123200,605948914,0,2428801000,4023352677,7387057267,0,125512100,2247268000,8372807359,4057475877,10547319281,159.95,55.39,79.38,Cukup Efektif,Efektif,Infak,Maluku_Papua
Maluku Utara,3318322619,6304128163,20240000,28577000,2849711194,2938416669,104600000,83575000,2856461042,7861704791,6292873813,9354696832,48.66,45.39,84.04,Kurang Efektif,Efektif,Maal,Maluku_Papua
Papua,2496549694,3509720961,174879912,1884069984,4370937247,5782716306,393920700,1385405556,6497374061,10470953309,7436287553,12561912807,68.93,87.37,83.35,Efektif,Efektif,Infak,Maluku_Papua
Papua Barat,266837626,226054915,51895000,166646200,89586467,170775830,52348000,35870000,658192107,462367200,460667093,599346945,30.10,142.88,77.15,Sangat Efektif,Efektif,Maal,Maluku_Papua"""
    return pd.read_csv(io.StringIO(csv_data))

df = load_data()

# ==============================================================================
# 3. PREPROCESSING & MODELING
# ==============================================================================
features_to_use = ['Region', 'Dominasi_Dana_2023', 'Pertumbuhan_Pengumpulan_Persen']
target_variable = 'ACR_2023_Persen'

X_raw = df[features_to_use]
y = df[target_variable]

# One-Hot Encoding
X_processed = pd.get_dummies(X_raw, columns=['Region', 'Dominasi_Dana_2023'], drop_first=False)
X_processed = X_processed.astype(int)

# Train Model
model = LinearRegression()
model.fit(X_processed, y)
y_pred = model.predict(X_processed)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Init Explainer di awal (Global)
explainer = shap.LinearExplainer(model, X_processed)
shap_values = explainer.shap_values(X_processed)

# ==============================================================================
# 4. TAMPILAN DASHBOARD
# ==============================================================================

# Membuat Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Overview", "📈 Performa Model", "🧠 XAI Explanation (SHAP)", "💡 Simulasi Prediksi"])

with tab1:
    st.subheader("📋 Data Laporan Keuangan BAZNAS")
    st.markdown("""
    **Penjelasan:** Tabel ini menampilkan data keuangan dari seluruh UPZ (Unit Pengumpul Zakat) BAZNAS di Indonesia. 
    Data mencakup pengumpulan dana Zakat Maal, Fitrah, Infak, DSKL, serta penyaluran dan ACR (Allocation to Collection Ratio) 
    yang menjadi indikator kesehatan finansial.
    """)
    st.dataframe(df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("📊 **Statistik Deskriptif ACR 2023:**")
        st.markdown("""
        Statistik ini menunjukkan distribusi nilai ACR 2023 di seluruh UPZ. 
        - **Mean (rata-rata)**: ACR rata-rata nasional
        - **Std (standar deviasi)**: Variasi ACR antar wilayah
        - **Min/Max**: Rentang kinerja terendah hingga tertinggi
        """)
        st.write(df['ACR_2023_Persen'].describe())
    
    with col2:
        st.write("🗺️ **Distribusi Jumlah UPZ per Region:**")
        st.markdown("""
        Grafik ini menunjukkan sebaran UPZ berdasarkan region geografis di Indonesia.
        Region dengan lebih banyak UPZ umumnya memiliki pengaruh lebih besar dalam analisis model.
        """)
        st.bar_chart(df['Region'].value_counts())
    
    st.write("🎯 **Distribusi Status Kesehatan 2023:**")
    st.markdown("""
    Kategorisasi kesehatan finansial UPZ berdasarkan nilai ACR:
    - **Sangat Efektif**: ACR ≥ 100% (penyaluran melebihi atau sama dengan pengumpulan)
    - **Efektif**: ACR 80-99%
    - **Cukup Efektif**: ACR 60-79%
    - **Kurang Efektif**: ACR < 60%
    """)
    status_counts = df['Status_Kesehatan_2023'].value_counts()
    st.bar_chart(status_counts)

with tab2:
    st.subheader("📈 Evaluasi Performa Model Regresi Linear")
    st.markdown("""
    Model regresi linear digunakan untuk memprediksi ACR berdasarkan 3 fitur utama:
    1. **Region** (lokasi geografis UPZ)
    2. **Dominasi Dana 2023** (jenis dana yang paling banyak dikumpulkan)
    3. **Pertumbuhan Pengumpulan (%)** (persentase peningkatan pengumpulan dana)
    """)
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric("R-Squared Score", f"{r2:.4f}", 
                 help="Proporsi variasi ACR yang dapat dijelaskan model. Nilai 1.0 = sempurna, >0.7 = baik")
    with col_metric2:
        st.metric("MAE", f"{mae:.2f}%", 
                 help="Mean Absolute Error: rata-rata selisih absolut prediksi dengan nilai aktual")
    with col_metric3:
        st.metric("RMSE", f"{rmse:.2f}%", 
                 help="Root Mean Squared Error: mengukur akurasi prediksi dengan memberi penalti lebih pada error besar")
    
    # Scatter Plot
    st.write("### 📊 Scatter Plot: Prediksi vs Nilai Aktual")
    st.markdown("""
    **Cara Membaca:**
    - **Sumbu X**: Nilai ACR yang diprediksi oleh model
    - **Sumbu Y**: Nilai ACR aktual dari data
    - **Garis merah putus-putus**: Garis ideal (prediksi = aktual)
    - Semakin dekat titik ke garis merah, semakin akurat prediksi
    - Warna berbeda menunjukkan region yang berbeda
    """)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_pred, y=y, hue=df['Region'], s=100, ax=ax, alpha=0.7)
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Garis Ideal (y=x)')
    ax.set_xlabel("Prediksi ACR (%)", fontsize=12)
    ax.set_ylabel("Nilai Aktual ACR (%)", fontsize=12)
    ax.set_title("Prediksi Model vs Nilai Aktual", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Koefisien
    st.write("### ⚖️ Bobot Koefisien Model")
    st.markdown("""
    **Penjelasan:** Tabel ini menunjukkan kontribusi setiap fitur terhadap prediksi ACR.
    - **Bobot positif (+)**: Fitur ini meningkatkan nilai ACR
    - **Bobot negatif (-)**: Fitur ini menurunkan nilai ACR
    - **Bobot mendekati 0**: Fitur ini memiliki pengaruh minimal
    
    Fitur dengan bobot absolut tertinggi memiliki pengaruh paling kuat terhadap prediksi.
    """)
    coef_df = pd.DataFrame({
        'Fitur': X_processed.columns, 
        'Bobot': model.coef_
    }).sort_values(by='Bobot', ascending=False)
    
    # Highlight positif/negatif
    def highlight_coef(val):
        if isinstance(val, (int, float)):
            color = 'background-color: #90EE90' if val > 0 else 'background-color: #FFB6C6'
            return color
        return ''
    
    st.dataframe(coef_df.style.applymap(highlight_coef, subset=['Bobot']), use_container_width=True)

with tab3:
    st.subheader("🧠 Interpretasi Model dengan SHAP (XAI)")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** adalah teknik XAI yang menjelaskan bagaimana setiap fitur 
    mempengaruhi prediksi model dengan cara yang adil dan konsisten berdasarkan teori permainan.
    """)
    
    st.write("### 1️⃣ Summary Plot (Beeswarm)")
    st.markdown("""
    **Cara Membaca Grafik Ini:**
    - **Sumbu Y**: Daftar fitur yang diurutkan berdasarkan kepentingan (paling penting di atas)
    - **Sumbu X**: Nilai SHAP (pengaruh terhadap prediksi)
        - Nilai positif (kanan): Fitur ini meningkatkan ACR
        - Nilai negatif (kiri): Fitur ini menurunkan ACR
    - **Warna titik**:
        - 🔴 **Merah**: Nilai fitur tinggi
        - 🔵 **Biru**: Nilai fitur rendah
    - **Sebaran titik**: Menunjukkan variasi dampak fitur di berbagai data
    
    **Contoh Interpretasi:**  
    Jika "Region_Jawa" memiliki banyak titik merah di sebelah kanan, artinya UPZ di Jawa 
    cenderung memiliki ACR lebih tinggi.
    """)
    
    fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_processed, show=False, plot_size=None)
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close()

    st.write("### 2️⃣ Bar Plot: Ranking Kepentingan Fitur")
    st.markdown("""
    **Cara Membaca:**
    - **Sumbu X**: Mean absolute SHAP value (rata-rata dampak mutlak)
    - **Sumbu Y**: Fitur-fitur yang diurutkan dari paling penting ke kurang penting
    
    Grafik ini menunjukkan fitur mana yang **paling sering mempengaruhi** prediksi model, 
    tanpa mempedulikan arah pengaruh (positif/negatif). Semakin panjang bar, semakin penting fitur tersebut.
    """)
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_processed, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()
    
    st.write("### 3️⃣ Feature Importance Table")
    st.markdown("""
    **Tabel ranking numerik** dari kepentingan fitur berdasarkan nilai SHAP rata-rata.
    """)
    
    feature_importance = pd.DataFrame({
        'Fitur': X_processed.columns,
        'Importance (mean |SHAP|)': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='Importance (mean |SHAP|)', ascending=False).reset_index(drop=True)
    
    st.dataframe(feature_importance, use_container_width=True)

with tab4:
    st.subheader("💡 Simulasi Prediksi Kesehatan Finansial")
    st.markdown("""
    Gunakan form di bawah ini untuk memprediksi ACR dari UPZ baru atau skenario hipotesis.
    Model akan memberikan prediksi ACR beserta **penjelasan visual** mengapa model memberikan nilai tersebut.
    """)
    
    # Input User
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        input_region = st.selectbox("🗺️ Pilih Region:", sorted(df['Region'].unique()))
        input_dominasi = st.selectbox("💰 Dominasi Dana:", sorted(df['Dominasi_Dana_2023'].unique()))
    with col_input2:
        input_pertumbuhan = st.number_input(
            "📈 Pertumbuhan Pengumpulan (%)", 
            min_value=-100.0, 
            max_value=1000.0, 
            value=10.0,
            step=1.0,
            help="Persentase pertumbuhan pengumpulan dana dibanding tahun lalu"
        )

    if st.button("🚀 Prediksi ACR", type="primary"):
        # 1. Siapkan data input
        input_data = pd.DataFrame({
            'Region': [input_region],
            'Dominasi_Dana_2023': [input_dominasi],
            'Pertumbuhan_Pengumpulan_Persen': [input_pertumbuhan]
        })
        
        # 2. Preprocessing input agar sesuai dengan format training
        input_processed = pd.get_dummies(input_data, columns=['Region', 'Dominasi_Dana_2023'], drop_first=False)
        
        # Menambahkan kolom yang hilang (jika ada) dengan nilai 0
        for col in X_processed.columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        # Urutkan kolom sesuai data training
        input_processed = input_processed[X_processed.columns]
        
        # 3. Prediksi
        pred_acr = model.predict(input_processed)[0]
        
        # Tentukan status kesehatan
        if pred_acr >= 100:
            status = "Sangat Efektif ⭐⭐⭐"
            color = "green"
        elif pred_acr >= 80:
            status = "Efektif ⭐⭐"
            color = "blue"
        elif pred_acr >= 60:
            status = "Cukup Efektif ⭐"
            color = "orange"
        else:
            status = "Kurang Efektif ⚠️"
            color = "red"
        
        st.success(f"### Prediksi Allocation to Collection Ratio (ACR): **{pred_acr:.2f}%**")
        st.markdown(f"**Status Kesehatan Finansial:** :{color}[{status}]")
        
        # 4. Penjelasan Lokal dengan Waterfall Plot
        st.write("---")
        st.subheader("🔍 Mengapa model memprediksi angka tersebut?")
        st.markdown("""
        **Grafik Waterfall di bawah ini** menunjukkan bagaimana model sampai pada prediksi tersebut:
        
        - **E[f(x)] (Base Value)**: Nilai ACR rata-rata dari semua data training (nilai awal sebelum melihat fitur spesifik)
        - **Panah merah (→)**: Fitur yang **menurunkan** prediksi dari base value
        - **Panah biru (→)**: Fitur yang **meningkatkan** prediksi dari base value
        - **f(x) (Final Prediction)**: Nilai prediksi akhir setelah semua fitur diperhitungkan
        
        Grafik ini menunjukkan kontribusi setiap fitur secara transparan, dari nilai dasar hingga prediksi final.
        """)

        # Hitung SHAP value khusus untuk satu data input ini
        shap_values_single = explainer.shap_values(input_processed)
        
        # Membuat objek Explanation agar kompatibel dengan waterfall plot
        explanation = shap.Explanation(
            values=shap_values_single[0], 
            base_values=explainer.expected_value, 
            data=input_processed.iloc[0].values, 
            feature_names=list(input_processed.columns)
        )
        
        # Tampilkan Grafik Waterfall dengan ukuran lebih besar
        fig_waterfall = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.tight_layout()
        st.pyplot(fig_waterfall)
        plt.close()
        
        # Insight tambahan
        st.write("---")
        st.write("### 📝 Insight & Rekomendasi")
        
        # Ambil 3 fitur paling berpengaruh
        shap_contribution = pd.DataFrame({
            'Fitur': input_processed.columns,
            'Kontribusi SHAP': shap_values_single[0]
        }).sort_values(by='Kontribusi SHAP', key=abs, ascending=False).head(3)
        
        st.markdown("**3 Fitur Paling Berpengaruh pada Prediksi Ini:**")
        for idx, row in shap_contribution.iterrows():
            impact = "meningkatkan" if row['Kontribusi SHAP'] > 0 else "menurunkan"
            st.write(f"- **{row['Fitur']}**: {impact} ACR sebesar {abs(row['Kontribusi SHAP']):.2f} poin")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Dashboard XAI BAZNAS | Powered by Streamlit, Scikit-learn & SHAP</p>
</div>
""", unsafe_allow_html=True)
