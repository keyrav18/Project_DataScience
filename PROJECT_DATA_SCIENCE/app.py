import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')  # Set backend sebelum import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

**ACR (Allocation to Collection Ratio)** = (Penyaluran / Pengumpulan) × 100%  
Semakin tinggi ACR, semakin efektif UPZ dalam menyalurkan dana yang terkumpul.
""")

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
@st.cache_data
def load_data():
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

X_processed = pd.get_dummies(X_raw, columns=['Region', 'Dominasi_Dana_2023'], drop_first=False)
X_processed = X_processed.astype(int)

model = LinearRegression()
model.fit(X_processed, y)
y_pred = model.predict(X_processed)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

explainer = shap.LinearExplainer(model, X_processed)
shap_values = explainer.shap_values(X_processed)

# ==============================================================================
# 4. TAMPILAN DASHBOARD
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Overview", "📈 Performa Model", "🧠 XAI Explanation (SHAP)", "💡 Simulasi Prediksi"])

# ==============================================================================
# TAB 1: DATA OVERVIEW
# ==============================================================================
with tab1:
    st.subheader("📋 Data Laporan Keuangan BAZNAS")
    st.markdown("""
    ### Apa itu data ini?
    Tabel ini menampilkan **data keuangan dari 36 UPZ (Unit Pengumpul Zakat)** BAZNAS di seluruh Indonesia untuk tahun 2022-2023.
    
    **Kolom penting:**
    - **Zakat Maal**: Zakat harta (emas, uang, bisnis, dll)
    - **Zakat Fitrah**: Zakat yang dibayar saat Ramadan
    - **Infak**: Sumbangan sukarela dari umat
    - **DSKL**: Dana Sosial Kemanusiaan Lainnya
    - **ACR (Allocation to Collection Ratio)**: Persentase dana yang disalurkan dibanding dana yang dikumpulkan
        - **ACR ≥ 100%**: Penyaluran ≥ Pengumpulan (Sangat Efektif)
        - **ACR 80-99%**: Efektif
        - **ACR 60-79%**: Cukup Efektif  
        - **ACR < 60%**: Kurang/Tidak Efektif
    
    **Insight:** Data ini digunakan untuk melatih model prediksi kesehatan finansial UPZ.
    """)
    
    with st.expander("🔍 Lihat Data Lengkap"):
        st.dataframe(df, use_container_width=True, height=400)
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 Statistik Deskriptif ACR 2023")
        st.markdown("""
        **Cara membaca:**
        - **Count**: Jumlah UPZ yang dianalisis (36 wilayah)
        - **Mean**: Rata-rata ACR nasional
        - **Std**: Variasi ACR antar wilayah (semakin besar = semakin beragam)
        - **Min**: ACR terendah (wilayah dengan kinerja penyaluran terendah)
        - **25%, 50%, 75%**: Kuartil distribusi ACR
        - **Max**: ACR tertinggi
        
        **Interpretasi:** Jika mean mendekati 100%, berarti rata-rata UPZ efektif menyalurkan dana yang terkumpul.
        """)
        stats_df = df['ACR_2023_Persen'].describe().to_frame()
        stats_df.columns = ['Nilai']
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    with col2:
        st.write("### 🗺️ Distribusi UPZ per Region")
        st.markdown("""
        **Penjelasan:**  
        Grafik batang ini menunjukkan jumlah UPZ di setiap region geografis Indonesia.
        
        **Insight:**
        - Region dengan lebih banyak UPZ (seperti Jawa, Sumatera) memiliki pengaruh lebih besar dalam model
        - Wilayah dengan sedikit UPZ (Pusat, Bali_Nusa, Maluku_Papua) mungkin kurang representatif
        """)
        region_counts = df['Region'].value_counts()
        st.bar_chart(region_counts)
        st.caption(f"Total: {len(df)} UPZ di {df['Region'].nunique()} region")
    
    st.write("---")
    
    st.write("### 🎯 Distribusi Status Kesehatan 2023")
    st.markdown("""
    **Penjelasan:**  
    Grafik ini menunjukkan berapa banyak UPZ di setiap kategori kesehatan finansial.
    
    **Kategorisasi:**
    - ⭐⭐⭐ **Sangat Efektif**: ACR ≥ 100% (penyaluran maksimal)
    - ⭐⭐ **Efektif**: ACR 80-99%
    - ⭐ **Cukup Efektif**: ACR 60-79%
    - ⚠️ **Kurang/Tidak Efektif**: ACR < 60%
    """)
    status_counts = df['Status_Kesehatan_2023'].value_counts().sort_index()
    fig_status, ax_status = plt.subplots(figsize=(10, 5))
    colors = {'Sangat Efektif': '#2ecc71', 'Efektif': '#3498db', 
              'Cukup Efektif': '#f39c12', 'Kurang Efektif': '#e74c3c', 'Tidak Efektif': '#c0392b'}
    bars = ax_status.bar(status_counts.index, status_counts.values, 
                         color=[colors.get(x, '#95a5a6') for x in status_counts.index])
    ax_status.set_ylabel('Jumlah UPZ', fontsize=12)
    ax_status.set_xlabel('Status Kesehatan', fontsize=12)
    ax_status.set_title('Distribusi Status Kesehatan Finansial UPZ 2023', fontsize=14, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax_status.text(bar.get_x() + bar.get_width()/2., height,
                      f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_status, clear_figure=True)
    plt.close()
    
    st.write("---")
    
    st.write("### 💰 Dominasi Jenis Dana 2023")
    st.markdown("""
    **Penjelasan:**  
    Grafik ini menunjukkan sumber dana utama yang dikumpulkan UPZ.
    
    **Insight:** Jenis dana yang dominan dapat mempengaruhi ACR karena karakteristik penyaluran yang berbeda.
    """)
    dominasi_counts = df['Dominasi_Dana_2023'].value_counts()
    fig_dom, ax_dom = plt.subplots(figsize=(8, 5))
    ax_dom.pie(dominasi_counts.values, labels=dominasi_counts.index, autopct='%1.1f%%', 
               colors=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'], startangle=90)
    ax_dom.set_title('Proporsi Dominasi Dana UPZ 2023', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_dom, clear_figure=True)
    plt.close()

# ==============================================================================
# TAB 2: PERFORMA MODEL
# ==============================================================================
with tab2:
    st.subheader("📈 Evaluasi Performa Model Regresi Linear")
    st.markdown("""
    ### Tentang Model
    Model **Regresi Linear** digunakan untuk memprediksi **ACR 2023** berdasarkan 3 fitur:
    
    1. 🗺️ **Region** (lokasi geografis UPZ)
    2. 💰 **Dominasi Dana 2023** (jenis dana utama: Maal, Infak, DSKL, atau Fitrah)
    3. 📈 **Pertumbuhan Pengumpulan (%)** (persentase peningkatan pengumpulan dana tahun 2023 vs 2022)
    
    **Rumus Model:**  
    `ACR = β₀ + β₁(Region) + β₂(Dominasi_Dana) + β₃(Pertumbuhan_Pengumpulan)`
    """)
    
    st.write("---")
    
    st.write("### 🎯 Metrik Evaluasi Model")
    st.markdown("""
    Metrik ini mengukur seberapa baik model memprediksi ACR:
    """)
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.metric("R² Score", f"{r2:.4f}")
        st.markdown("""
        **R-Squared (Koefisien Determinasi)**  
        Mengukur proporsi variasi ACR yang dijelaskan model.
        
        - **1.0** = Model sempurna (100% akurat)
        - **0.7-1.0** = Model baik
        - **0.5-0.7** = Model cukup
        - **< 0.5** = Model lemah
        
        **Nilai Anda:** Model menjelaskan **{:.1f}%** variasi ACR.
        """.format(r2 * 100))
    
    with col_metric2:
        st.metric("MAE", f"{mae:.2f}%")
        st.markdown("""
        **Mean Absolute Error**  
        Rata-rata selisih absolut antara prediksi dan nilai aktual.
        
        - Semakin kecil semakin baik
        - Dalam satuan % ACR
        
        **Interpretasi:** Rata-rata error prediksi adalah **±{:.2f}%** ACR.
        """.format(mae))
    
    with col_metric3:
        st.metric("RMSE", f"{rmse:.2f}%")
        st.markdown("""
        **Root Mean Squared Error**  
        Mengukur akurasi dengan memberi penalti lebih pada error besar.
        
        - Lebih sensitif terhadap outlier
        - Dalam satuan % ACR
        
        **Interpretasi:** Error prediksi terbobot adalah **{:.2f}%** ACR.
        """.format(rmse))
    
    st.write("---")
    
    st.write("### 📊 Scatter Plot: Prediksi vs Nilai Aktual")
    st.markdown("""
    **Cara Membaca Grafik:**
    - **Sumbu X (horizontal)**: Nilai ACR yang diprediksi model
    - **Sumbu Y (vertikal)**: Nilai ACR aktual dari data
    - **Garis merah putus-putus**: Garis ideal (prediksi = aktual)
    - **Jarak titik dari garis merah**: Menunjukkan error prediksi
        - Titik di atas garis = model underprediction (prediksi < aktual)
        - Titik di bawah garis = model overprediction (prediksi > aktual)
    - **Warna titik**: Menunjukkan region geografis
    
    **Interpretasi yang Baik:**  
    Jika semua titik dekat dengan garis merah, model akurat.
    """)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plot dengan warna per region
    for region in df['Region'].unique():
        mask = df['Region'] == region
        ax.scatter(y_pred[mask], y[mask], label=region, s=120, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Garis ideal
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2.5, label='Garis Ideal (y=x)', zorder=5)
    
    ax.set_xlabel("Prediksi ACR (%)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Nilai Aktual ACR (%)", fontsize=13, fontweight='bold')
    ax.set_title("Perbandingan Prediksi Model vs Nilai Aktual ACR", fontsize=15, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close()
    
    st.caption("💡 **Tip:** Hover pada legenda untuk melihat region. Titik yang mengelompok dekat garis merah menunjukkan prediksi akurat.")
    
    st.write("---")
    
    st.write("### ⚖️ Bobot Koefisien Model (Feature Weights)")
    st.markdown("""
    **Penjelasan:**  
    Tabel ini menunjukkan kontribusi setiap fitur terhadap prediksi ACR.
    
    **Cara Membaca:**
    - **Bobot Positif (+)**: Fitur ini **meningkatkan** nilai ACR
    - **Bobot Negatif (-)**: Fitur ini **menurunkan** nilai ACR
    - **Bobot Besar (absolut)**: Fitur ini memiliki pengaruh kuat
    - **Bobot Kecil (~0)**: Fitur ini hampir tidak berpengaruh
    
    **Contoh Interpretasi:**  
    Jika `Region_Jawa = +15.2`, artinya UPZ di Jawa cenderung memiliki ACR **15.2% lebih tinggi** dari baseline (region lain).
    
    **Warna:**
    - 🟢 **Hijau** = Koefisien positif (meningkatkan ACR)
    - 🔴 **Merah** = Koefisien negatif (menurunkan ACR)
    """)
    
    coef_df = pd.DataFrame({
        'Fitur': X_processed.columns, 
        'Bobot (Koefisien)': model.coef_
    }).sort_values(by='Bobot (Koefisien)', key=lambda x: abs(x), ascending=False)
    
    coef_df['Dampak'] = coef_df['Bobot (Koefisien)'].apply(
        lambda x: '↑ Meningkatkan ACR' if x > 0 else '↓ Menurunkan ACR'
    )
    
    def highlight_coef(row):
        if row['Bobot (Koefisien)'] > 0:
            return ['background-color: #d4edda']*len(row)
        elif row['Bobot (Koefisien)'] < 0:
            return ['background-color: #f8d7da']*len(row)
        else:
            return ['']*len(row)
    
    st.dataframe(
        coef_df.style.apply(highlight_coef, axis=1).format({'Bobot (Koefisien)': "{:.4f}"}),
        use_container_width=True,
        height=400
    )
    
    st.write("---")
    
    st.write("### 📊 Visualisasi Koefisien (Top 10)")
    fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
    
    top_coef = coef_df.head(10)
    colors_coef = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_coef['Bobot (Koefisien)']]
    
    ax_coef.barh(top_coef['Fitur'], top_coef['Bobot (Koefisien)'], color=colors_coef, edgecolor='black')
    ax_coef.set_xlabel('Bobot Koefisien', fontsize=12, fontweight='bold')
    ax_coef.set_ylabel('Fitur', fontsize=12, fontweight='bold')
    ax_coef.set_title('Top 10 Fitur dengan Pengaruh Terbesar', fontsize=14, fontweight='bold')
    ax_coef.axvline(0, color='black', linestyle='-', linewidth=1)
    ax_coef.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_coef, clear_figure=True)
    plt.close()
    
    st.caption("💡 **Insight:** Fitur di sebelah kanan garis 0 meningkatkan ACR, sebelah kiri menurunkan ACR.")

# ==============================================================================
# TAB 3: XAI SHAP
# ==============================================================================
with tab3:
    st.subheader("🧠 Interpretasi Model dengan SHAP (XAI)")
    st.markdown("""
    ### Apa itu SHAP?
    **SHAP (SHapley Additive exPlanations)** adalah teknik **Explainable AI (XAI)** yang menjelaskan 
    bagaimana setiap fitur mempengaruhi prediksi model dengan cara yang **adil dan konsisten**, 
    berdasarkan **teori permainan (Game Theory)**.
    
    **Keunggulan SHAP:**
    - ✅ Menunjukkan kontribusi setiap fitur secara transparan
    - ✅ Konsisten secara matematis (berbasis Shapley Values)
    - ✅ Dapat mendeteksi interaksi antar fitur
    - ✅ Berlaku untuk model apa pun (model-agnostic)
    
    **Konsep Kunci:**
    - **SHAP Value > 0**: Fitur ini **meningkatkan** prediksi
    - **SHAP Value < 0**: Fitur ini **menurunkan** prediksi
    - **SHAP Value = 0**: Fitur ini tidak berpengaruh
    """)
    
    st.write("---")
    
    st.write("### 1️⃣ Summary Plot (Beeswarm)")
    st.markdown("""
    **Grafik ini adalah "peta pengaruh" dari semua fitur terhadap prediksi ACR.**
    
    **Cara Membaca:**
    - **Sumbu Y**: Fitur-fitur diurutkan dari **paling penting (atas)** hingga kurang penting (bawah)
    - **Sumbu X**: Nilai SHAP (dampak terhadap prediksi)
        - **Kanan (positif)**: Fitur ini **meningkatkan** ACR
        - **Kiri (negatif)**: Fitur ini **menurunkan** ACR
        - **Sekitar 0**: Fitur ini tidak berdampak signifikan
    - **Warna Titik**:
        - 🔴 **Merah/Pink**: Nilai fitur **tinggi** (feature value = high)
        - 🔵 **Biru**: Nilai fitur **rendah** (feature value = low)
    - **Sebaran Titik**: Menunjukkan variasi dampak fitur di berbagai data
    
    **Contoh Interpretasi:**  
    Jika fitur "Region_Jawa" memiliki banyak titik merah di sebelah **kanan** (positif):
    - Artinya: UPZ di Jawa (nilai fitur = 1 = merah) cenderung **meningkatkan ACR**
    
    Jika "Pertumbuhan_Pengumpulan_Persen" memiliki titik merah di kanan dan biru di kiri:
    - Artinya: Pertumbuhan tinggi → ACR naik, Pertumbuhan rendah → ACR turun
    """)
    
    fig_shap = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_processed, show=False)
    plt.title("SHAP Summary Plot: Pengaruh Fitur terhadap Prediksi ACR", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("SHAP Value (Dampak terhadap Prediksi ACR)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_shap, clear_figure=True)
    plt.close()
    
    st.caption("💡 **Tip:** Fitur di posisi atas adalah yang paling berpengaruh dalam prediksi model.")
    
    st.write("---")
    
    st.write("### 2️⃣ Bar Plot: Ranking Kepentingan Fitur")
    st.markdown("""
    **Grafik ini merangking fitur berdasarkan "seberapa sering mereka mempengaruhi prediksi".**
    
    **Cara Membaca:**
    - **Sumbu X**: Mean |SHAP Value| (rata-rata dampak mutlak)
        - Semakin besar nilai → Semakin penting fitur tersebut
    - **Sumbu Y**: Fitur-fitur diurutkan dari **paling penting (atas)** ke kurang penting (bawah)
    
    **Perbedaan dengan Beeswarm:**
    - **Beeswarm**: Menunjukkan **arah** pengaruh (positif/negatif)
    - **Bar Plot**: Menunjukkan **besarnya** pengaruh (tanpa mempedulikan arah)
    
    **Interpretasi:**  
    Grafik ini menjawab: **"Fitur mana yang paling sering digunakan model untuk membuat keputusan?"**
    """)
    
    fig_bar = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_processed, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance: Ranking Kepentingan Fitur", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Mean |SHAP Value| (Rata-rata Dampak Mutlak)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_bar, clear_figure=True)
    plt.close()
    
    st.caption("💡 **Tip:** Fitur dengan bar terpanjang adalah yang paling kritikal untuk akurasi model.")
    
    st.write("---")
    
    st.write("### 3️⃣ Tabel Feature Importance")
    st.markdown("""
    **Tabel ranking numerik dari kepentingan fitur** berdasarkan nilai SHAP rata-rata.
    
    Kolom:
    - **Fitur**: Nama fitur
    - **Importance (mean |SHAP|)**: Rata-rata dampak mutlak fitur tersebut
    - **Ranking**: Urutan kepentingan (1 = paling penting)
    """)
    
    feature_importance = pd.DataFrame({
        'Fitur': X_processed.columns,
        'Importance (mean |SHAP|)': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='Importance (mean |SHAP|)', ascending=False).reset_index(drop=True)
    
    feature_importance['Ranking'] = range(1, len(feature_importance) + 1)
    feature_importance = feature_importance[['Ranking', 'Fitur', 'Importance (mean |SHAP|)']]
    
    st.dataframe(
        feature_importance.style.format({'Importance (mean |SHAP|)': "{:.4f}"}),
        use_container_width=True,
        height=400
    )
    
    st.write("---")
    
    st.write("### 🔍 Insight dari Analisis SHAP")
    st.markdown(f"""
    **Berdasarkan analisis SHAP di atas:**
    
    1. **Fitur paling penting:** `{feature_importance.iloc[0]['Fitur']}`  
       → Fitur ini memiliki dampak terbesar dalam memprediksi ACR
    
    2. **Top 5 fitur berpengaruh:**
    """)
    
    for idx, row in feature_importance.head(5).iterrows():
        st.markdown(f"   {idx+1}. **{row['Fitur']}** (Importance: {row['Importance (mean |SHAP|)']:.4f})")
    
    st.markdown("""
    3. **Rekomendasi untuk stakeholder:**
       - Fokuskan perhatian pada fitur-fitur di ranking teratas
       - Monitor wilayah/kategori dengan SHAP value negatif (menurunkan ACR)
       - Perbaiki strategi pengumpulan di region dengan dampak negatif
    """)

# ==============================================================================
# TAB 4: SIMULASI PREDIKSI
# ==============================================================================
with tab4:
    st.subheader("💡 Simulasi Prediksi Kesehatan Finansial")
    st.markdown("""
    ### Tentang Simulasi Ini
    Gunakan form di bawah untuk **memprediksi ACR dari UPZ baru** atau **skenario hipotesis**.
    
    **Manfaat:**
    - 🎯 Estimasi kesehatan finansial UPZ baru sebelum operasional
    - 📊 Analisis skenario "what-if" (misalnya: "Bagaimana jika pertumbuhan 50%?")
    - 🔍 Pemahaman mendalam tentang faktor-faktor yang mempengaruhi ACR
    
    **Output:**
    1. Prediksi nilai ACR (dalam %)
    2. Status kesehatan finansial (Sangat Efektif / Efektif / Cukup / Kurang Efektif)
    3. **Grafik Waterfall SHAP**: Penjelasan visual mengapa model memberikan prediksi tersebut
    """)
    
    st.write("---")
    
    st.write("### 📋 Input Parameter")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        input_region = st.selectbox(
            "🗺️ Pilih Region:", 
            sorted(df['Region'].unique()),
            help="Pilih wilayah geografis UPZ. Region yang berbeda memiliki karakteristik ekonomi dan demografis yang berbeda."
        )
        
        input_dominasi = st.selectbox(
            "💰 Dominasi Dana:", 
            sorted(df['Dominasi_Dana_2023'].unique()),
            help="Jenis dana yang paling banyak dikumpulkan UPZ.\n- Maal: Zakat harta\n- Infak: Sumbangan sukarela\n- Fitrah: Zakat fitrah\n- DSKL: Dana sosial lainnya"
        )
    
    with col_input2:
        input_pertumbuhan = st.number_input(
            "📈 Pertumbuhan Pengumpulan (%)", 
            min_value=-100.0, 
            max_value=1000.0, 
            value=10.0,
            step=1.0,
            help="Persentase pertumbuhan pengumpulan dana dibanding tahun lalu.\n- Positif: Pertumbuhan\n- Negatif: Penurunan\n- 0: Stabil"
        )
        
        st.info(f"""
        **Parameter yang Anda pilih:**
        - Region: **{input_region}**
        - Dominasi Dana: **{input_dominasi}**
        - Pertumbuhan: **{input_pertumbuhan}%**
        """)
    
    st.write("---")
    
    if st.button("🚀 Prediksi ACR", type="primary", use_container_width=True):
        # Preprocessing input
        input_data = pd.DataFrame({
            'Region': [input_region],
            'Dominasi_Dana_2023': [input_dominasi],
            'Pertumbuhan_Pengumpulan_Persen': [input_pertumbuhan]
        })
        
        input_processed = pd.get_dummies(input_data, columns=['Region', 'Dominasi_Dana_2023'], drop_first=False)
        
        for col in X_processed.columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        input_processed = input_processed[X_processed.columns]
        
        # Prediksi
        pred_acr = model.predict(input_processed)[0]
        
        # Tentukan status kesehatan
        if pred_acr >= 100:
            status = "Sangat Efektif"
            emoji = "⭐⭐⭐"
            color = "green"
            desc = "UPZ ini menyalurkan dana sama dengan atau melebihi jumlah yang dikumpulkan. Kinerja optimal!"
        elif pred_acr >= 80:
            status = "Efektif"
            emoji = "⭐⭐"
            color = "blue"
            desc = "UPZ ini menyalurkan sebagian besar dana yang dikumpulkan. Kinerja baik."
        elif pred_acr >= 60:
            status = "Cukup Efektif"
            emoji = "⭐"
            color = "orange"
            desc = "UPZ ini menyalurkan cukup banyak dana, namun masih bisa ditingkatkan."
        else:
            status = "Kurang/Tidak Efektif"
            emoji = "⚠️"
            color = "red"
            desc = "UPZ ini menyalurkan kurang dari 60% dana yang dikumpulkan. Perlu perbaikan strategis."
        
        # Display hasil
        st.success("### ✅ Prediksi Berhasil!")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric(
                label="Prediksi ACR",
                value=f"{pred_acr:.2f}%",
                delta=f"{pred_acr - df['ACR_2023_Persen'].mean():.2f}% vs rata-rata nasional"
            )
        
        with col_result2:
            st.markdown(f"""
            **Status Kesehatan Finansial:**  
            :{color}[**{status} {emoji}**]
            
            {desc}
            """)
        
        st.write("---")
        
        # Waterfall Plot
        st.subheader("🔍 Mengapa model memprediksi angka tersebut?")
        st.markdown("""
        **Grafik Waterfall di bawah ini** menunjukkan bagaimana model sampai pada prediksi tersebut **langkah demi langkah**.
        
        ### Cara Membaca Grafik Waterfall:
        
        1. **E[f(x)] (Base Value)**: 
           - Nilai ACR **rata-rata dari semua data training**
           - Ini adalah "titik awal" sebelum model melihat fitur spesifik input Anda
           - Posisi: Paling bawah
        
        2. **Panah/Bar**:
           - 🔴 **Merah (ke kiri)**: Fitur ini **menurunkan** prediksi dari base value
           - 🔵 **Biru (ke kanan)**: Fitur ini **meningkatkan** prediksi dari base value
           - **Panjang panah**: Menunjukkan seberapa besar dampak fitur tersebut
        
        3. **f(x) (Final Prediction)**:
           - Nilai prediksi **akhir** setelah semua fitur diperhitungkan
           - Posisi: Paling atas
        
        **Alur Pembacaan:**  
        Mulai dari **base value** → Ikuti panah naik/turun → Sampai di **prediksi akhir**
        
        **Contoh:**  
        Jika base value = 90%, lalu "Region_Jawa" menambah +10%, dan "Pertumbuhan_Pengumpulan" menambah +5%, 
        maka prediksi akhir = 90% + 10% + 5% = 105%
        """)
        
        # Hitung SHAP value untuk input ini
        shap_values_single = explainer.shap_values(input_processed)
        
        # Membuat Explanation object
        explanation = shap.Explanation(
            values=shap_values_single[0], 
            base_values=explainer.expected_value, 
            data=input_processed.iloc[0].values, 
            feature_names=list(input_processed.columns)
        )
        
        # Render Waterfall Plot
        fig_waterfall = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(f"SHAP Waterfall Plot: Penjelasan Prediksi ACR = {pred_acr:.2f}%", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig_waterfall, clear_figure=True)
        plt.close()
        
        st.caption("💡 **Tip:** Grafik ini menunjukkan transparansi penuh tentang bagaimana model mengambil keputusan.")
        
        st.write("---")
        
        # Insight & Rekomendasi
        st.write("### 📝 Insight & Rekomendasi")
        
        # Analisis kontribusi fitur
        shap_contribution = pd.DataFrame({
            'Fitur': input_processed.columns,
            'Kontribusi SHAP': shap_values_single[0],
            'Nilai Fitur': input_processed.iloc[0].values
        }).sort_values(by='Kontribusi SHAP', key=abs, ascending=False)
        
        # Filter hanya fitur yang aktif (nilai = 1 untuk categorical, atau non-zero untuk numerical)
        active_features = shap_contribution[shap_contribution['Nilai Fitur'] != 0].head(5)
        
        st.markdown("#### 🔝 Fitur yang Paling Berpengaruh pada Prediksi Ini:")
        
        for idx, row in active_features.iterrows():
            impact = "meningkatkan" if row['Kontribusi SHAP'] > 0 else "menurunkan"
            arrow = "↑" if row['Kontribusi SHAP'] > 0 else "↓"
            color_badge = "🟢" if row['Kontribusi SHAP'] > 0 else "🔴"
            
            st.markdown(f"""
            {color_badge} **{row['Fitur']}**  
            {arrow} {impact.capitalize()} ACR sebesar **{abs(row['Kontribusi SHAP']):.2f} poin**
            """)
        
        st.write("---")
        
        st.markdown("#### 💡 Rekomendasi Strategis:")
        
        # Analisis berdasarkan hasil prediksi
        if pred_acr < 80:
            st.warning("""
            **⚠️ ACR di bawah 80% - Perlu Perbaikan:**
            
            1. **Tingkatkan Efisiensi Penyaluran:**
               - Review proses penyaluran dana
               - Identifikasi bottleneck operasional
               - Perkuat koordinasi dengan penerima manfaat
            
            2. **Optimalkan Strategi Pengumpulan:**
               - Sesuaikan target pengumpulan dengan kapasitas penyaluran
               - Diversifikasi sumber dana
            
            3. **Benchmark dengan UPZ Terbaik:**
               - Pelajari best practices dari UPZ dengan ACR tinggi di region yang sama
            """)
        elif pred_acr < 100:
            st.info("""
            **ℹ️ ACR 80-99% - Kinerja Baik, Bisa Ditingkatkan:**
            
            1. **Targetkan ACR ≥ 100%:**
               - Tingkatkan kecepatan penyaluran
               - Identifikasi program penyaluran baru
            
            2. **Jaga Pertumbuhan Stabil:**
               - Maintain momentum pertumbuhan pengumpulan
               - Perkuat hubungan dengan donatur
            
            3. **Monitor KPI:**
               - Tracking ACR bulanan
               - Set target improvement 5-10% per tahun
            """)
        else:
            st.success("""
            **✅ ACR ≥ 100% - Kinerja Excellent!**
            
            1. **Pertahankan Kinerja:**
               - Dokumentasikan best practices
               - Share knowledge dengan UPZ lain
            
            2. **Ekspansi Program:**
               - Kembangkan program penyaluran inovatif
               - Perluas jangkauan penerima manfaat
            
            3. **Sustainability:**
               - Pastikan pertumbuhan berkelanjutan
               - Jaga kualitas penyaluran
            """)
        
        st.write("---")
        
        st.markdown("#### 📊 Bandingkan dengan Data Aktual")
        
        # Cari UPZ dengan karakteristik serupa
        similar_upz = df[
            (df['Region'] == input_region) & 
            (df['Dominasi_Dana_2023'] == input_dominasi)
        ]
        
        if len(similar_upz) > 0:
            st.markdown(f"""
            **UPZ dengan karakteristik serupa (Region: {input_region}, Dominasi: {input_dominasi}):**
            """)
            
            comparison_df = similar_upz[['Wilayah', 'ACR_2023_Persen', 'Status_Kesehatan_2023', 'Pertumbuhan_Pengumpulan_Persen']].copy()
            comparison_df.columns = ['Wilayah', 'ACR Aktual (%)', 'Status', 'Pertumbuhan (%)']
            
            st.dataframe(comparison_df.style.format({
                'ACR Aktual (%)': '{:.2f}',
                'Pertumbuhan (%)': '{:.2f}'
            }), use_container_width=True)
            
            avg_acr_similar = similar_upz['ACR_2023_Persen'].mean()
            st.metric(
                "Rata-rata ACR UPZ Serupa",
                f"{avg_acr_similar:.2f}%",
                f"{pred_acr - avg_acr_similar:.2f}% vs prediksi Anda"
            )
        else:
            st.info("Tidak ada UPZ dengan karakteristik persis sama dalam data historis.")
        
        st.write("---")
        
        st.markdown("""
        ### 🎓 Kesimpulan
        
        Simulasi ini menunjukkan bahwa model machine learning dapat membantu:
        - 🎯 **Prediksi proaktif** kesehatan finansial UPZ
        - 🔍 **Transparansi penuh** tentang faktor-faktor yang mempengaruhi ACR
        - 💡 **Rekomendasi data-driven** untuk perbaikan strategi
        
        **Next Steps:**
        1. Validasi prediksi dengan data aktual
        2. Gunakan insight SHAP untuk perbaikan operasional
        3. Monitor ACR secara berkala dan adjust strategi
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Dashboard XAI BAZNAS</strong></p>
    <p>Powered by Streamlit, Scikit-learn & SHAP | Explainable AI for Financial Health Evaluation</p>
    <p>© 2025 | Data: BAZNAS Indonesia</p>
</div>
""", unsafe_allow_html=True)
