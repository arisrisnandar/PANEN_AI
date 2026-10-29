# ======================
# 1. IMPORT LIBRARY
# ======================

import pandas as pd
import numpy as np
import warnings
import re
import json
import optuna
from flask import Flask, request, jsonify, send_from_directory, Response, session, redirect
from functools import wraps
from dotenv import load_dotenv
load_dotenv()
from flask_cors import CORS
from flask_compress import Compress
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib
import os
import requests

warnings.filterwarnings('ignore')

# ── Pastikan working directory = folder file ini ──
# Perlu karena PythonAnywhere menjalankan app lewat WSGI dari direktori
# yang beda, sedangkan semua path dataset/model di bawah ini masih
# ditulis relatif (mis. "Dataset/harga", "xgboost_model.pkl", dst).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)
Compress(app)  # Otomatis gzip semua response JSON/text > 500 bytes

# ======================
# ADMIN CONFIG
# ======================
app.secret_key  = os.environ.get('SECRET_KEY', 'panen-ai-secret-2026')
ADMIN_USERNAME  = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD  = os.environ.get('ADMIN_PASSWORD', 'panen2026')
SUPABASE_URL    = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY    = os.environ.get('SUPABASE_KEY', '')
SUPABASE_BUCKET = os.environ.get('SUPABASE_BUCKET', 'panen-ai-uploads')

# Folder dataset
DIR_HARGA = 'Dataset/harga'
DIR_FSVA  = 'Dataset/kabupaten'
DIR_HUJAN = 'Dataset/hujan'

# Folder upload backup per komoditas/jenis
DIR_UPLOADS = {
    'cabai': 'Dataset/prediksi/uploads/cabai',
    'ayam':  'Dataset/prediksi/uploads/ayam',
    'sapi':  'Dataset/prediksi/uploads/sapi',
    'telur': 'Dataset/prediksi/uploads/telur',
    'hujan': 'Dataset/prediksi/uploads/hujan',
}
UPLOAD_METADATA_FILE = 'upload_metadata.json'

for _d in [DIR_HARGA, DIR_FSVA, DIR_HUJAN] + list(DIR_UPLOADS.values()):
    os.makedirs(_d, exist_ok=True)

_supabase = None
def get_supabase():
    """Return True kalau Supabase terkonfigurasi. Semua operasi Storage yang
    sesungguhnya lewat REST API langsung (lihat supabase_storage_* di bawah),
    BUKAN lewat library python 'supabase' -- karena versi terbarunya (2.31.0)
    punya bug internal ("'dict' object has no attribute 'text'") saat
    menangani response non-2xx dari server, bikin upload/list gagal diam-diam."""
    return bool(SUPABASE_URL and SUPABASE_KEY)

def _supabase_headers(content_type=None):
    h = {'Authorization': f'Bearer {SUPABASE_KEY}', 'apikey': SUPABASE_KEY}
    if content_type:
        h['Content-Type'] = content_type
    return h

def supabase_storage_upload(path, data: bytes, content_type='text/csv'):
    """Upload file ke Supabase Storage lewat REST API langsung."""
    if not get_supabase():
        return False
    url = f'{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}'
    headers = _supabase_headers(content_type)
    headers['x-upsert'] = 'true'
    r = requests.post(url, headers=headers, data=data, timeout=30)
    if r.status_code not in (200, 201):
        raise Exception(f"Upload gagal ({r.status_code}): {r.text[:300]}")
    return True

def supabase_storage_list(prefix):
    """List file di Supabase Storage lewat REST API langsung."""
    if not get_supabase():
        return []
    url = f'{SUPABASE_URL}/storage/v1/object/list/{SUPABASE_BUCKET}'
    headers = _supabase_headers('application/json')
    body = {'prefix': prefix, 'limit': 1000, 'offset': 0}
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code != 200:
        raise Exception(f"List gagal ({r.status_code}): {r.text[:300]}")
    return r.json()

def supabase_storage_download(path):
    """Download file dari Supabase Storage lewat REST API langsung."""
    if not get_supabase():
        return None
    url = f'{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}'
    headers = _supabase_headers()
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise Exception(f"Download gagal ({r.status_code}): {r.text[:300]}")
    return r.content

def supabase_storage_remove(paths):
    """Hapus file di Supabase Storage lewat REST API langsung."""
    if not get_supabase():
        return False
    url = f'{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}'
    headers = _supabase_headers('application/json')
    r = requests.delete(url, headers=headers, json={'prefixes': paths}, timeout=30)
    if r.status_code not in (200, 201):
        raise Exception(f"Hapus gagal ({r.status_code}): {r.text[:300]}")
    return True

# ======================
# RESTORE UPLOAD DARI SUPABASE (persistensi lintas restart)
# ======================
# Railway/Render/PythonAnywhere pakai disk yang SEMENTARA: tiap restart atau
# redeploy, folder Dataset/prediksi/uploads/* di lokal ke-reset kosong.
# Supabase Storage dipakai sebagai penyimpanan permanen. Fungsi ini menarik
# balik semua file yang pernah di-upload admin, SEBELUM server mulai jalan.
_uploads_restored = False

def _download_supabase_folder(sb, remote_prefix, local_dir):
    global _uploads_restored
    try:
        items = supabase_storage_list(remote_prefix)
    except Exception as e:
        print(f"[restore] Gagal list '{remote_prefix}': {e}")
        return
    for item in items or []:
        fname = item.get('name')
        if not fname or not fname.lower().endswith('.csv'):
            continue
        local_path = os.path.join(local_dir, fname)
        if os.path.exists(local_path):
            continue  # sudah ada lokal, skip
        try:
            data = supabase_storage_download(f'{remote_prefix}/{fname}')
            with open(local_path, 'wb') as f_out:
                f_out.write(data)
            _uploads_restored = True
            print(f"[restore] {local_path} dipulihkan dari Supabase")
        except Exception as e:
            print(f"[restore] Gagal download '{fname}': {e}")

def restore_uploads_from_supabase():
    """Tarik balik semua file upload admin dari Supabase Storage sebelum
    df dan cache dimuat, biar histori upload gak hilang tiap restart/redeploy."""
    sb = get_supabase()
    if not sb:
        print("[restore] Supabase belum dikonfigurasi, lewati restore.")
        return
    for _key, _dir in DIR_UPLOADS.items():
        _download_supabase_folder(sb, f'uploads/{_key}', _dir)
    _download_supabase_folder(sb, 'fsva', DIR_FSVA)

    # Restore metadata upload (kalau lokal belum ada / kosong)
    if not os.path.exists(UPLOAD_METADATA_FILE) or os.path.getsize(UPLOAD_METADATA_FILE) <= 2:
        try:
            data = supabase_storage_download('meta/upload_metadata.json')
            with open(UPLOAD_METADATA_FILE, 'wb') as f_out:
                f_out.write(data)
            print("[restore] upload_metadata.json dipulihkan dari Supabase")
        except Exception:
            pass  # belum pernah ada backup metadata, gak masalah

restore_uploads_from_supabase()

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

# ======================
# 2. LOAD SEMUA DATA HARIAN
# ======================

def _load_base_plus_uploads(base_path, upload_dir, dedup_subset=None):
    """
    Gabungkan file dasar (base_path, tidak pernah ditulis ulang/diubah) dengan
    semua file yang pernah diupload admin (upload_dir) -- di MEMORI setiap kali
    data di-load, tanpa menulis ulang satu file gabungan ke disk.

    Ini pola yang sama seperti FSVA: tiap file berdiri sendiri, hapus 1 file
    upload tidak pernah mengubah/merusak file lain.
    """
    dfs = [pd.read_csv(base_path)]
    if upload_dir and os.path.isdir(upload_dir):
        for fname in sorted(os.listdir(upload_dir)):
            if not fname.lower().endswith('.csv'):
                continue
            fpath = os.path.join(upload_dir, fname)
            try:
                dfs.append(pd.read_csv(fpath))
            except Exception as e:
                print(f"[load] Gagal baca upload '{fpath}': {e}")
    combined = pd.concat(dfs, ignore_index=True)

    # Normalisasi teks provinsi/komoditas: HANYA strip spasi ekstra di
    # depan/belakang -- TIDAK mengubah huruf besar/kecil, karena beberapa
    # nama provinsi punya akronim (DKI Jakarta, DI Yogyakarta) yang akan
    # RUSAK kalau di-title-case (jadi "Dki Jakarta", "Di Yogyakarta").
    # Spasi ekstra sendirian sudah cukup sering jadi penyebab dedup gagal.
    for col in ('provinsi', 'komoditas'):
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.strip()

    if dedup_subset and set(dedup_subset).issubset(combined.columns):
        before_n = len(combined)
        combined = combined.drop_duplicates(subset=dedup_subset, keep='last')
        if before_n != len(combined):
            print(f"[load] Dedup '{base_path}': {before_n} -> {len(combined)} baris")
    return combined

def load_and_preprocess_data():
    """Load dan preprocess data persis seperti syntax yang diberikan"""

    _dedup = ['tanggal', 'provinsi', 'komoditas']
    cabai_konsumen = _load_base_plus_uploads(
        "Dataset/prediksi/Cabai Rawit Merah_Konsumen_Harian.csv", DIR_UPLOADS.get('cabai'), _dedup
    )
    cabai_produsen = pd.read_csv(
        "Dataset/prediksi/Cabai Rawit Merah_Produsen_Harian.csv"
    )
    ayam_konsumen = _load_base_plus_uploads(
        "Dataset/prediksi/Daging Ayam Ras_Konsumen_Harian.csv", DIR_UPLOADS.get('ayam'), _dedup
    )
    ayam_produsen = pd.read_csv(
        "Dataset/prediksi/Daging Ayam Ras_Produsen_Harian.csv"
    )
    sapi_konsumen = _load_base_plus_uploads(
        "Dataset/prediksi/Daging Sapi Murni_Konsumen_Harian.csv", DIR_UPLOADS.get('sapi'), _dedup
    )
    sapi_produsen = pd.read_csv(
        "Dataset/prediksi/Daging Sapi Murni_Produsen_Harian.csv"
    )
    telur_konsumen = _load_base_plus_uploads(
        "Dataset/prediksi/Telur Ayam Ras_Konsumen_Harian.csv", DIR_UPLOADS.get('telur'), _dedup
    )
    telur_produsen = pd.read_csv(
        "Dataset/prediksi/Telur Ayam Ras_Produsen_Harian.csv"
    )

    # =========================================================
    # 3. LOAD DATA BULANAN
    # =========================================================

    bulanan_konsumen = pd.read_csv(
        "Dataset/prediksi/Harga_Konsumen_Bulanan.csv", sep=','
    )
    bulanan_produsen = pd.read_csv(
        "Dataset/prediksi/Harga_Produsen_Bulanan.csv", sep=','
    )

    # =========================================================
    # 4. LOAD DATA EKSTERNAL
    # =========================================================

    curah = _load_base_plus_uploads(
        "Dataset/prediksi/Curah_Hujan_Bulanan.csv", DIR_UPLOADS.get('hujan')
    )
    hbkn = pd.read_csv(
        "Dataset/prediksi/HBKN_Bulanan.csv"
    )

    # =========================================================
    # 5. GABUNG DATA HARIAN
    # =========================================================

    df_konsumen = pd.concat([
        cabai_konsumen,
        ayam_konsumen,
        sapi_konsumen,
        telur_konsumen
    ])

    df_produsen = pd.concat([
        cabai_produsen,
        ayam_produsen,
        sapi_produsen,
        telur_produsen
    ])

    print(f"[load] Baris per file konsumen -> cabai:{len(cabai_konsumen)} ayam:{len(ayam_konsumen)} "
          f"sapi:{len(sapi_konsumen)} telur:{len(telur_konsumen)} | total df_konsumen:{len(df_konsumen)}")
    print(f"[load] Baris per file produsen -> cabai:{len(cabai_produsen)} ayam:{len(ayam_produsen)} "
          f"sapi:{len(sapi_produsen)} telur:{len(telur_produsen)} | total df_produsen:{len(df_produsen)}")

    # =========================================================
    # 6. PRA-PEMROSESAN DATA HARIAN
    # =========================================================

    df_konsumen['tanggal'] = pd.to_datetime(
        df_konsumen['tanggal']
    )

    df_konsumen['harga'] = pd.to_numeric(
        df_konsumen['harga'].replace('#DIV/0!', np.nan),
        errors='coerce'
    )

    df_produsen['tanggal'] = pd.to_datetime(
        df_produsen['tanggal']
    )

    df_produsen['harga'] = pd.to_numeric(
        df_produsen['harga'].replace('#DIV/0!', np.nan),
        errors='coerce'
    )

    df_produsen.rename(
        columns={'harga':'harga_produsen'},
        inplace=True
    )

    # =========================================================
    # 7. MERGE HARGA HARIAN
    # =========================================================

    df = df_konsumen.merge(
        df_produsen[
            ['tanggal','komoditas','provinsi','harga_produsen']
        ],
        on=['tanggal','komoditas','provinsi'],
        how='left'
    )
    print(f"[load] Sesudah merge konsumen+produsen: {len(df)} baris")

    # =========================================================
    # 8. PRA-PEMROSESAN DATA BULANAN
    # =========================================================

    bulan_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    # KONSUMEN BULANAN
    with open("Dataset/prediksi/Harga_Konsumen_Bulanan.csv", encoding="utf-8-sig") as f:
        lines = f.readlines()

    KOLOM = ["Kode Provinsi", "Nama Provinsi", "Komoditas",
             "Bulan", "Tahun", "Produksi (Ton)", "Harga"]

    rows = []
    for line in lines[1:]:
        line = line.strip().strip('"')
        match = re.search(r'"?"?(Rp[\d,\.]+)"?"?\s*$', line)
        if match:
            harga = match.group(1)
            sisa  = line[:match.start()].strip().strip('"').rstrip(",")
        else:
            harga = ""
            sisa  = line.strip().strip('"')
        parts = sisa.split(",")
        if len(parts) >= 6:
            rows.append([parts[0].strip(), parts[1].strip(), parts[2].strip(),
                         parts[3].strip(), parts[4].strip(), parts[5].strip(), harga])

    bulanan_konsumen = pd.DataFrame(rows, columns=KOLOM)

    bulanan_konsumen["Harga"] = pd.to_numeric(bulanan_konsumen["Harga"].str.replace("Rp","").str.replace(",",""), errors="coerce")
    bulanan_konsumen["Produksi (Ton)"] = pd.to_numeric(bulanan_konsumen["Produksi (Ton)"].str.replace(",",""), errors="coerce")
    bulanan_konsumen["Kode Provinsi"] = pd.to_numeric(bulanan_konsumen["Kode Provinsi"], errors="coerce")

    bulanan_konsumen.rename(columns={
        'Kode Provinsi': 'kode_wilayah',
        'Nama Provinsi': 'provinsi',
        'Komoditas': 'komoditas',
        'Harga': 'harga_bulanan_konsumen',
        'Produksi (Ton)': 'produksi_ton',
        'Bulan': 'month',
        'Tahun': 'year'
    }, inplace=True)

    bulanan_konsumen['provinsi'] = bulanan_konsumen['provinsi'].str.strip().str.lower()
    bulanan_konsumen['komoditas'] = bulanan_konsumen['komoditas'].str.strip().str.lower()
    bulanan_konsumen['month'] = bulanan_konsumen['month'].str.strip().map(bulan_map).astype('Int64')
    bulanan_konsumen['year'] = pd.to_numeric(bulanan_konsumen['year'], errors='coerce').astype('Int64')
    bulanan_konsumen['harga_bulanan_konsumen'] = pd.to_numeric(bulanan_konsumen['harga_bulanan_konsumen'], errors='coerce')
    bulanan_konsumen['produksi_ton'] = pd.to_numeric(bulanan_konsumen['produksi_ton'], errors='coerce')
    bulanan_konsumen["provinsi"] = bulanan_konsumen["provinsi"].str.title()
    bulanan_konsumen["komoditas"] = bulanan_konsumen["komoditas"].str.title()

    # PRODUSEN BULANAN
    bulanan_produsen.rename(columns={
        'Kode Provinsi': 'kode_wilayah',
        'Nama Provinsi': 'provinsi',
        'Komoditas': 'komoditas',
        'Harga (Rp)': 'harga_bulanan_produsen',
        'Bulan': 'month',
        'Tahun': 'year'
    }, inplace=True)

    harga_produsen_str = bulanan_produsen['harga_bulanan_produsen'].astype(str)
    bulanan_produsen['harga_bulanan_produsen'] = pd.to_numeric(
        harga_produsen_str.str.replace("Rp","").str.replace(",",""), 
        errors="coerce"
    )

    bulanan_produsen['provinsi'] = bulanan_produsen['provinsi'].str.strip().str.lower()
    bulanan_produsen['komoditas'] = bulanan_produsen['komoditas'].str.strip().str.lower()
    bulanan_produsen['month'] = bulanan_produsen['month'].str.strip().map(bulan_map).astype('Int64')
    bulanan_produsen['year'] = pd.to_numeric(bulanan_produsen['year'], errors='coerce').astype('Int64')
    bulanan_produsen["provinsi"] = bulanan_produsen["provinsi"].str.title()
    bulanan_produsen["komoditas"] = bulanan_produsen["komoditas"].str.title()

    # =========================================================
    # 9. TAMBAH MONTH & YEAR
    # =========================================================

    df['month'] = df['tanggal'].dt.month
    df['year'] = df['tanggal'].dt.year

    # =========================================================
    # 10. MERGE DATA BULANAN
    # =========================================================

    df['month'] = df['month'].astype('Int64')
    df['year'] = df['year'].astype('Int64')

    df = df.merge(
        bulanan_konsumen[
            ['provinsi', 'komoditas', 'month', 'year', 'harga_bulanan_konsumen', 'produksi_ton']
        ],
        on=['provinsi','komoditas','month','year'],
        how='left'
    )

    df = df.merge(
        bulanan_produsen[
            ['provinsi', 'komoditas', 'month', 'year', 'harga_bulanan_produsen']
        ],
        on=['provinsi','komoditas','month','year'],
        how='left'
    )

    # =========================================================
    # 11. CURAH HUJAN
    # =========================================================

    curah['bulan'] = curah['Bulan'].map(bulan_map)
    curah['tanggal'] = pd.to_datetime(
        curah['Tahun'].astype(str) + '-' +
        curah['bulan'].astype(str) + '-01'
    )
    curah.rename(columns={
        'Kode Provinsi':'kode wilayah',
        'Nama Provinsi':'provinsi',
        'Curah Hujan':'curah_hujan'
    }, inplace=True)
    curah['month'] = curah['tanggal'].dt.month
    curah['year'] = curah['tanggal'].dt.year

    # =========================================================
    # 12. MERGE CURAH HUJAN
    # =========================================================

    df = df.merge(
        curah[['provinsi','month','year','curah_hujan']],
        on=['provinsi','month','year'],
        how='left'
    )

    # =========================================================
    # 13. MERGE HBKN
    # =========================================================

    hbkn['tanggal'] = pd.to_datetime(hbkn['tanggal'])
    df = df.merge(hbkn[['tanggal','hbkn']], on='tanggal', how='left')
    df['hbkn'] = df['hbkn'].fillna(0)

    # =========================================================
    # 14. SORT DATA
    # =========================================================

    df = df.sort_values(['tanggal', 'provinsi'])

    # =========================================================
    # 15. INTERPOLASI
    # =========================================================

    df['harga'] = df.groupby(['komoditas','provinsi'])['harga'].transform(
        lambda x: x.interpolate(method='linear', limit=30)
    )
    df['harga_produsen'] = df.groupby(['komoditas','provinsi'])['harga_produsen'].transform(
        lambda x: x.interpolate(method='linear', limit=30)
    )
    df['harga_bulanan_konsumen'] = df.groupby(['komoditas','provinsi'])['harga_bulanan_konsumen'].transform(
        lambda x: x.ffill().bfill()
    )
    df['harga_bulanan_produsen'] = df.groupby(['komoditas','provinsi'])['harga_bulanan_produsen'].transform(
        lambda x: x.ffill().bfill()
    )
    df['produksi_ton'] = df.groupby(['komoditas','provinsi'])['produksi_ton'].transform(
        lambda x: x.ffill().bfill()
    )

    # =========================================================
    # HANDLE MISSING VALUE
    # =========================================================

    print(f"[load] Sebelum dropna(harga): {len(df)} baris")
    df = df.dropna(subset=['harga'])
    print(f"[load] Sesudah dropna(harga): {len(df)} baris")
    df['harga_produsen'] = df['harga_produsen'].fillna(df['harga_produsen'].median())
    df['harga_bulanan_konsumen'] = df['harga_bulanan_konsumen'].fillna(df['harga_bulanan_konsumen'].median())
    df['harga_bulanan_produsen'] = df['harga_bulanan_produsen'].fillna(df['harga_bulanan_produsen'].median())
    df['produksi_ton'] = df['produksi_ton'].fillna(df['produksi_ton'].median())
    df['curah_hujan'] = df['curah_hujan'].fillna(df['curah_hujan'].median())

    # =========================================================
    # 16. WINSORIZING OUTLIER (clip persentil 1–99)
    # =========================================================

    winsor_cols = [
        'harga',
        'harga_produsen',
        'harga_bulanan_konsumen',
        'harga_bulanan_produsen',
        'produksi_ton',
        'curah_hujan',
    ]

    for col in winsor_cols:
        lo = df.groupby(['komoditas', 'provinsi'])[col].transform(
            lambda x: x.quantile(0.01)
        )
        hi = df.groupby(['komoditas', 'provinsi'])[col].transform(
            lambda x: x.quantile(0.99)
        )
        df[col] = df[col].clip(lower=lo, upper=hi)

    # =========================================================
    # 17. FEATURE ENGINEERING
    # =========================================================

    df['harga_harian_konsumen_lag_1'] = df.groupby(['komoditas','provinsi'])['harga'].shift(1)
    df['harga_harian_konsumen_lag_3'] = df.groupby(['komoditas','provinsi'])['harga'].shift(3)
    df['harga_harian_konsumen_lag_7'] = df.groupby(['komoditas','provinsi'])['harga'].shift(7)
    df['harga_harian__konsumen_ma_7'] = df.groupby(['komoditas','provinsi'])['harga'].transform(lambda x: x.rolling(7).mean())
    df['harga_harian__produsen_ma_7'] = df.groupby(['komoditas','provinsi'])['harga_produsen'].transform(lambda x: x.rolling(7).mean())
    df['harga_harian_produsen_lag_1'] = df.groupby(['provinsi', 'komoditas'])['harga_produsen'].shift(1)
    df['harga_harian_produsen_lag_3'] = df.groupby(['provinsi', 'komoditas'])['harga_produsen'].shift(3)
    df['harga_harian_produsen_lag_7'] = df.groupby(['provinsi', 'komoditas'])['harga_produsen'].shift(7)
    df['hujan_lag_1'] = df.groupby('provinsi')['curah_hujan'].shift(30)  
    df['hujan_lag_2'] = df.groupby('provinsi')['curah_hujan'].shift(60)  
    df['hujan_lag_3'] = df.groupby('provinsi')['curah_hujan'].shift(90)  
    df['harga_bulanan_konsumen_lag_1'] = df.groupby(['provinsi', 'komoditas'])['harga_bulanan_konsumen'].shift(30)
    df['harga_bulanan_konsumen_lag_3'] = df.groupby(['provinsi', 'komoditas'])['harga_bulanan_konsumen'].shift(90)
    df['harga_bulanan_produsen_lag_1'] = df.groupby(['provinsi', 'komoditas'])['harga_bulanan_produsen'].shift(30)
    df['harga_bulanan_produsen_lag_3'] = df.groupby(['provinsi', 'komoditas'])['harga_bulanan_produsen'].shift(90)
    df['harga_bulanan_konsumen_ma_3'] = df.groupby(['provinsi', 'komoditas'])['harga_bulanan_konsumen'].transform(lambda x: x.rolling(3).mean())
    df['harga_bulanan_produsen_ma_3'] = df.groupby(['provinsi', 'komoditas'])['harga_bulanan_produsen'].transform(lambda x: x.rolling(3).mean())
    df['hbkn_lag_1'] = df.groupby('provinsi')['hbkn'].shift(1)
    df['hbkn_lag_3'] = df.groupby('provinsi')['hbkn'].shift(3)
    df['hbkn_lag_7'] = df.groupby('provinsi')['hbkn'].shift(7)
    df['std_7_harga'] = df.groupby(['komoditas','provinsi'])['harga'].transform(lambda x: x.rolling(7).std())
    df['std_7_harga_produsen'] = df.groupby(['komoditas', 'provinsi'])['harga_produsen'].transform(lambda x: x.rolling(7).std())
    df['std_3_harga_bulanan_konsumen'] = df.groupby(['komoditas', 'provinsi'])['harga_bulanan_konsumen'].transform(lambda x: x.rolling(3).std())
    df['std_3_harga_bulanan_produsen'] = df.groupby(['komoditas', 'provinsi'])['harga_bulanan_produsen'].transform(lambda x: x.rolling(3).std())
    df['dayofweek'] = df['tanggal'].dt.dayofweek
    df['weekofyear'] = df['tanggal'].dt.isocalendar().week.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # =========================================================
    # 18. ENCODING KATEGORI
    # =========================================================

    # komoditas map
    komoditas_map = dict(enumerate(df['komoditas'].astype('category').cat.categories))
    provinsi_map  = dict(enumerate(df['provinsi'].astype('category').cat.categories))
    print(f"[load] Jumlah kategori komoditas: {len(komoditas_map)} -> {list(komoditas_map.values())}")
    print(f"[load] Jumlah kategori provinsi: {len(provinsi_map)}")

    df['komoditas'] = df['komoditas'].astype('category').cat.codes
    df['provinsi']  = df['provinsi'].astype('category').cat.codes

    # DROP NA
    print(f"[load] Sebelum dropna(fitur lag): {len(df)} baris")
    df = df.dropna(subset=[
        'harga_harian_konsumen_lag_1', 'harga_harian_konsumen_lag_3', 'harga_harian_konsumen_lag_7',
        'harga_harian__konsumen_ma_7', 'harga_harian__produsen_ma_7', 'harga_harian_produsen_lag_1',
        'harga_harian_produsen_lag_3', 'harga_harian_produsen_lag_7', 'hujan_lag_1', 'hujan_lag_2',
        'hujan_lag_3', 'harga_bulanan_konsumen_lag_1', 'harga_bulanan_konsumen_lag_3',
        'harga_bulanan_produsen_lag_1', 'harga_bulanan_produsen_lag_3', 'harga_bulanan_konsumen_ma_3',
        'harga_bulanan_produsen_ma_3', 'hbkn_lag_1', 'hbkn_lag_3', 'hbkn_lag_7', 'std_7_harga',
        'std_7_harga_produsen', 'std_3_harga_bulanan_konsumen', 'std_3_harga_bulanan_produsen'
    ])
    print(f"[load] Sesudah dropna(fitur lag): {len(df)} baris")

    return df, komoditas_map, provinsi_map

# =========================================================
# TRAIN MODEL DAN SAVE
# =========================================================

def train_and_save_model():
    """Train XGBoost model dan save ke file"""
    
    print("Loading and preprocessing data...")
    df, komoditas_map, provinsi_map = load_and_preprocess_data()
    
    features = [
        # Lagging harga harian konsumen
        'harga_harian_konsumen_lag_1',
        'harga_harian_konsumen_lag_3',
        'harga_harian_konsumen_lag_7',
        
        # Moving average harian
        'harga_harian__konsumen_ma_7',
        'harga_harian__produsen_ma_7',
        
        # Lagging harga harian produsen
        'harga_harian_produsen_lag_1',
        'harga_harian_produsen_lag_3',
        'harga_harian_produsen_lag_7',
        
        # Lagging curah hujan
        'hujan_lag_1',
        'hujan_lag_2',
        'hujan_lag_3',
        
        # Lagging dan moving average harga bulanan
        'harga_bulanan_konsumen_lag_1',
        'harga_bulanan_konsumen_lag_3',
        'harga_bulanan_produsen_lag_1',
        'harga_bulanan_produsen_lag_3',
        'harga_bulanan_konsumen_ma_3',
        'harga_bulanan_produsen_ma_3',
        
        # Lagging HBKN
        'hbkn_lag_1',
        'hbkn_lag_3',
        'hbkn_lag_7',
        
        # Standar deviasi
        'std_7_harga',
        'std_7_harga_produsen',
        'std_3_harga_bulanan_konsumen',
        'std_3_harga_bulanan_produsen',
        
        # Fitur kalender
        'dayofweek',
        'weekofyear',
        'month_sin',
        'month_cos',
        
        # Fitur tambahan
        'harga_produsen',
        'harga_bulanan_konsumen',
        'harga_bulanan_produsen',
        'produksi_ton',
        'curah_hujan',
        'hbkn',
        'komoditas',
        'provinsi'
    ]
    
    X = df[features]
    y = df['harga']
    
    print(f"Training model with {X.shape[1]} features and {X.shape[0]} samples...")

    # =========================================================
    # 19. HOLDOUT SPLIT — 80% train, 20% test (kronologis)
    # =========================================================

    split_idx = int(len(X) * 0.80)

    X_train_full = X.iloc[:split_idx]
    y_train_full = y.iloc[:split_idx]

    X_test_hold  = X.iloc[split_idx:]   # data paling baru, tidak disentuh selama tuning
    y_test_hold  = y.iloc[split_idx:]

    print(f"Holdout split: {len(X_train_full)} train / {len(X_test_hold)} test")

    # =========================================================
    # 20. OPTUNA — hanya pakai X_train_full
    # =========================================================

    def objective(trial):

        params = {

            'n_estimators': trial.suggest_int(
                'n_estimators',
                100,
                300
            ),

            'max_depth': trial.suggest_int(
                'max_depth',
                3,
                10
            ),

            'learning_rate': trial.suggest_float(
                'learning_rate',
                0.01,
                0.3
            ),

            'subsample': trial.suggest_float(
                'subsample',
                0.5,
                1.0
            ),

            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                0.5,
                1.0
            ),

            'objective':'reg:squarederror',

            'random_state':42
        }

        model = XGBRegressor(**params)

        tscv = TimeSeriesSplit(n_splits=3)

        rmse_list = []

        for train_idx, test_idx in tscv.split(X_train_full):

            X_train = X_train_full.iloc[train_idx]
            X_test  = X_train_full.iloc[test_idx]

            y_train = y_train_full.iloc[train_idx]
            y_test  = y_train_full.iloc[test_idx]

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            rmse = np.sqrt(
                mean_squared_error(y_test, pred)
            )

            rmse_list.append(rmse)

        return np.mean(rmse_list)

    study = optuna.create_study(
        direction='minimize'
    )

    study.optimize(
        objective,
        n_trials=20
    )

    # =========================================================
    # 21. MODEL FINAL
    # =========================================================

    print(study.best_params)

    model = XGBRegressor(
        **study.best_params,
        objective='reg:squarederror',
        random_state=42
    )
    
    # =========================================================
    # 22. CROSS VALIDATION — hanya pakai X_train_full
    # =========================================================

    tscv = TimeSeriesSplit(n_splits=5)

    rmse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmse_pct_scores = []
    mae_pct_scores = []

    for train_idx, test_idx in tscv.split(X_train_full):

        X_train = X_train_full.iloc[train_idx]
        X_test  = X_train_full.iloc[test_idx]

        y_train = y_train_full.iloc[train_idx]
        y_test  = y_train_full.iloc[test_idx]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        y_mean = np.mean(y_test)

        rmse_scores.append(
            (np.sqrt(mean_squared_error(y_test, pred)) / y_mean) * 100
        )

        mae_scores.append(
            (mean_absolute_error(y_test, pred) / y_mean) * 100
        )

        rmse_pct_scores.append(
            (np.sqrt(mean_squared_error(y_test, pred)) / y_mean) * 100
        )
        mae_pct_scores.append(
            (mean_absolute_error(y_test, pred) / y_mean) * 100
        )

        mape_scores.append(
            np.mean(
                np.abs((y_test - pred) / y_test)
            ) * 100
        )
        
        r2_scores.append(
            r2_score(y_test, pred)
        )
    
    # =========================================================
    # 25. EVALUASI HOLDOUT TEST SET
    # =========================================================

    # Fit dulu pada train_full untuk evaluasi holdout yang jujur
    model.fit(X_train_full, y_train_full)
    pred_hold = model.predict(X_test_hold)

    y_mean_hold = np.mean(y_test_hold)
    holdout_rmse = (np.sqrt(mean_squared_error(y_test_hold, pred_hold)) / y_mean_hold) * 100
    holdout_mae  = (mean_absolute_error(y_test_hold, pred_hold) / y_mean_hold) * 100
    holdout_mape = float(np.mean(np.abs((y_test_hold - pred_hold) / y_test_hold)) * 100)
    holdout_r2   = float(r2_score(y_test_hold, pred_hold))

    print("\n===== HASIL EVALUASI HOLDOUT TEST SET =====")
    print(f"RMSE : {holdout_rmse:.2f}%")
    print(f"MAE  : {holdout_mae:.2f}%")
    print(f"MAPE : {holdout_mape:.2f}%")
    print(f"R2   : {holdout_r2:.4f}")

    # =========================================================
    # 26. SAVE MODEL — refit pada seluruh data untuk deployment
    # =========================================================
    
    model.fit(X, y)
    joblib.dump(model, 'xgboost_model.pkl')
    print("Model saved to xgboost_model.pkl")

    # Save feature names
    with open('feature_names.json', 'w') as f:
        json.dump(features, f)

    # Save map
    with open('category_maps.json', 'w') as f:
        json.dump({
            'komoditas_map': komoditas_map,
            'provinsi_map': provinsi_map
        }, f)

    # Save metrics — pakai metrik holdout sebagai metrik utama (lebih jujur)
    with open('model_metrics.json', 'w') as f:
        json.dump({
            # Metrik holdout (ditampilkan di dashboard)
            'rmse': holdout_rmse,
            'mae':  holdout_mae,
            'mape': holdout_mape,
            'r2':   holdout_r2,
            # Metrik CV (referensi tambahan)
            'cv_rmse': float(np.mean(rmse_scores)),
            'cv_mae':  float(np.mean(mae_scores)),
            'cv_mape': float(np.mean(mape_scores)),
            'cv_r2':   float(np.mean(r2_scores)),
            'holdout_n': len(X_test_hold),
            'train_n':   len(X_train_full),
        }, f)

    df.to_parquet('df_cache.parquet')

    # =========================================================
    # 23. HASIL EVALUASI
    # =========================================================

    print("\n===== HASIL EVALUASI CV (Train Set) =====")
    print(f"RMSE : {np.mean(rmse_scores):.2f}%")
    print(f"MAE  : {np.mean(mae_scores):.2f}%")
    print(f"MAPE : {np.mean(mape_scores):.2f}%")
    print(f"R2   : {np.mean(r2_scores):.4f}")
    return model, features, rmse_scores, mae_scores, mape_scores, r2_scores, df, komoditas_map, provinsi_map

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading or training model...")

if os.path.exists('xgboost_model.pkl') and os.path.exists('feature_names.json') and os.path.exists('category_maps.json'):
    model = joblib.load('xgboost_model.pkl')
    
    with open('feature_names.json', 'r') as f:
        features = json.load(f)
    
    with open('category_maps.json', 'r') as f:
        maps = json.load(f)
        komoditas_map = {int(k): v for k, v in maps['komoditas_map'].items()}
        provinsi_map  = {int(k): v for k, v in maps['provinsi_map'].items()}

    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json', 'r') as f:
            saved_metrics = json.load(f)
    else:
        saved_metrics = {'rmse': 0, 'mae': 0, 'mape': 0, 'r2': 0}

    print("Model loaded from file")
    if os.path.exists('df_cache.parquet'):
        df = pd.read_parquet('df_cache.parquet')
        print("Data loaded from cache")
    else:
        df, _, _ = load_and_preprocess_data()
        df.to_parquet('df_cache.parquet')
        print("df_cache.parquet dibuat ulang dari CSV")

else:
    model, features, rmse_scores, mae_scores, mape_scores, r2_scores, df, komoditas_map, provinsi_map = train_and_save_model()
    with open('model_metrics.json', 'r') as f:
        saved_metrics = json.load(f)

# Get unique categories
komoditas_list      = list(komoditas_map.values())
provinsi_list       = list(provinsi_map.values())
komoditas_name2code = {v: k for k, v in komoditas_map.items()}
provinsi_name2code  = {v: k for k, v in provinsi_map.items()}

# =========================================================
# PRECOMPUTE CACHE
# =========================================================

import time as _time
import json as _json

_predict_all_cache = {}
_CACHE_TTL = 3600  # tidak dipakai lagi, cache berlaku selamanya
_CACHE_FILE = 'predict_all_cache.json'

def _load_cache_from_file():
    """Load cache dari file JSON saat startup."""
    global _predict_all_cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                saved = _json.load(f)
            for kom, val in saved.items():
                _predict_all_cache[kom] = {'data': val['data'], 'ts': val['ts']}
            print(f"[cache] Loaded {len(_predict_all_cache)} komoditas dari file cache")
        except Exception as e:
            print(f"[cache] Gagal load cache file: {e}")

def _save_cache_to_file():
    """Simpan cache ke file JSON."""
    try:
        to_save = {k: {'data': v['data'], 'ts': v['ts']} for k, v in _predict_all_cache.items()}
        with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
            _json.dump(to_save, f)
        print(f"[cache] Saved {len(to_save)} komoditas ke file cache")
    except Exception as e:
        print(f"[cache] Gagal simpan cache file: {e}")

def _compute_predict_all(komoditas_name, horizon=28):
    kom_code = komoditas_name2code.get(komoditas_name)
    if kom_code is None:
        return {}

    def risk_level(pct):
        if pct > 15:  return 'Krisis'
        if pct > 10:  return 'Siaga'
        if pct > 5:   return 'Waspada'
        return 'Aman'

    result = {}
    for prov_code, prov_name in provinsi_map.items():
        try:
            mask    = (df['provinsi'] == prov_code) & (df['komoditas'] == kom_code)
            df_filt = df[mask].sort_values('tanggal')
            if df_filt.empty:
                continue

            latest_row     = df_filt.iloc[-1]
            harga_terakhir = float(latest_row['harga'])

            price_window = df_filt['harga'].tail(7).tolist()
            prod_window  = df_filt['harga_produsen'].tail(7).tolist()
            while len(price_window) < 7: price_window.insert(0, price_window[0] if price_window else 0)
            while len(prod_window)  < 7: prod_window.insert(0, prod_window[0]   if prod_window  else 0)

            base_f   = {feat: float(latest_row[feat]) if feat in latest_row.index and pd.notna(latest_row[feat]) else 0.0
                        for feat in features}
            base_dow = int(latest_row['dayofweek'])
            base_woy = int(latest_row['weekofyear'])

            preds = []
            pw = list(price_window)
            dw = list(prod_window)
            for i in range(horizon):
                cf = base_f.copy()
                cf['harga_harian_konsumen_lag_1'] = pw[-1]
                cf['harga_harian_konsumen_lag_3'] = pw[-3]
                cf['harga_harian_konsumen_lag_7'] = pw[-7]
                cf['harga_harian__konsumen_ma_7'] = float(np.mean(pw[-7:]))
                cf['std_7_harga']                 = float(np.std(pw[-7:]))
                cf['harga_harian_produsen_lag_1'] = dw[-1]
                cf['harga_harian_produsen_lag_3'] = dw[-3]
                cf['harga_harian_produsen_lag_7'] = dw[-7]
                cf['harga_harian__produsen_ma_7'] = float(np.mean(dw[-7:]))
                cf['std_7_harga_produsen']        = float(np.std(dw[-7:]))
                cf['dayofweek']  = (base_dow + i + 1) % 7
                cf['weekofyear'] = base_woy + (base_dow + i + 1) // 7

                pred = float(model.predict(pd.DataFrame([cf])[features])[0])
                preds.append(pred)
                pw.append(pred); pw = pw[-7:]
                dw.append(pred * (dw[-1] / pw[-2] if pw[-2] != 0 else 0.8))
                dw = dw[-7:]

            minggu = {}
            for m in range(1, 5):
                start = (m - 1) * 7
                end   = m * 7
                sl    = preds[start:end]
                if sl:
                    avg = np.mean(sl)
                    pct = ((avg - harga_terakhir) / harga_terakhir * 100) if harga_terakhir else 0
                    minggu[f'minggu_{m}'] = {
                        'avg_price':  round(avg, 2),
                        'pct_change': round(pct, 2),
                        'risk_level': risk_level(pct),
                    }

            avg_all = np.mean(preds)
            pct_all = ((avg_all - harga_terakhir) / harga_terakhir * 100) if harga_terakhir else 0

            # Anomaly Score: seberapa jauh prediksi vs historis 30 hari terakhir
            hist_prices = df_filt['harga'].tail(30).tolist()
            hist_mean   = float(np.mean(hist_prices)) if hist_prices else harga_terakhir
            hist_std    = float(np.std(hist_prices))  if len(hist_prices) > 1 else 1.0
            if hist_std == 0: hist_std = 1.0
            anomaly_score = abs((avg_all - hist_mean) / hist_std)

            result[prov_name] = {
                'harga_terakhir': round(harga_terakhir, 2),
                'avg_prediksi':   round(avg_all, 2),
                'pct_change':     round(pct_all, 2),
                'risk_level':     risk_level(pct_all),
                'anomaly_score':  round(anomaly_score, 2),
                'minggu':         minggu,
                'predictions':    [round(p, 2) for p in preds],
            }
        except Exception as e:
            print(f"[precompute] Gagal {prov_name}: {e}")
            continue

    return dict(sorted(result.items(), key=lambda x: x[1]['pct_change'], reverse=True))


# EWS Horizon cache
_EWS_CACHE_FILE = 'ews_cache.json'
_ews_cache = {}

def _load_ews_cache():
    global _ews_cache
    if os.path.exists(_EWS_CACHE_FILE):
        try:
            with open(_EWS_CACHE_FILE, 'r', encoding='utf-8') as f:
                _ews_cache = _json.load(f)
            print(f"[ews-cache] Loaded dari file cache")
        except Exception as e:
            print(f"[ews-cache] Gagal load: {e}")

def _save_ews_cache():
    try:
        with open(_EWS_CACHE_FILE, 'w', encoding='utf-8') as f:
            _json.dump(_ews_cache, f)
        print("[ews-cache] Saved ke file cache")
    except Exception as e:
        print(f"[ews-cache] Gagal simpan: {e}")


# FSVA Nowcast cache
_FSVA_CACHE_FILE = 'fsva_nowcast_cache.json'
_fsva_cache = {}   # { tahun(int): [records] }

def _load_fsva_cache():
    global _fsva_cache
    if os.path.exists(_FSVA_CACHE_FILE):
        try:
            with open(_FSVA_CACHE_FILE, 'r', encoding='utf-8') as f:
                raw = _json.load(f)
            _fsva_cache = {int(k): v for k, v in raw.items()}
            print(f"[fsva-cache] Loaded {len(_fsva_cache)} tahun dari file cache")
        except Exception as e:
            print(f"[fsva-cache] Gagal load: {e}")

def _save_fsva_cache():
    try:
        with open(_FSVA_CACHE_FILE, 'w', encoding='utf-8') as f:
            _json.dump({str(k): v for k, v in _fsva_cache.items()}, f)
        print("[fsva-cache] Saved ke file cache")
    except Exception as e:
        print(f"[fsva-cache] Gagal simpan: {e}")

def precompute_all():
    # Load dari file dulu kalau ada
    _load_cache_from_file()

    # Cek komoditas mana yang belum ada di cache
    missing = [k for k in komoditas_list if k not in _predict_all_cache]

    if not missing:
        print("[precompute] Semua komoditas sudah ada di cache file, skip hitung ulang")
    else:
        print(f"[precompute] Menghitung {len(missing)} komoditas yang belum di-cache...")
        for kom in missing:
            print(f"[precompute] Menghitung: {kom}")
            data = _compute_predict_all(kom, horizon=28)
            _predict_all_cache[kom] = {'data': data, 'ts': _time.time()}  # ts dipertahankan untuk info saja
            print(f"[precompute] Selesai: {kom} ({len(data)} provinsi)")
        _save_cache_to_file()
        print("[precompute] Semua komoditas selesai!")

    # Precompute EWS horizon kalau belum ada
    _load_ews_cache()
    if not _ews_cache:
        print("[precompute] Menghitung EWS horizon...")
        _compute_and_cache_ews()
        print("[precompute] EWS selesai!")
    else:
        print("[precompute] EWS sudah ada di cache, skip hitung ulang")

    # Precompute FSVA nowcast kalau belum ada
    _load_fsva_cache()
    if not _fsva_cache:
        print("[precompute] Menghitung FSVA nowcast cache...")
        try:
            from flask import Response
            import glob
            # Panggil fungsi inti (bukan endpoint) langsung
            _build_fsva_cache()
            print("[precompute] FSVA nowcast selesai!")
        except Exception as e:
            print(f"[precompute] FSVA nowcast gagal: {e}")
    else:
        print("[precompute] FSVA cache sudah ada, skip hitung ulang")


import threading
threading.Thread(target=precompute_all, daemon=True).start()

# =========================================================
# API ENDPOINTS
# =========================================================

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information and performance metrics.
    Returns komoditas_map and provinsi_map as {name: code} so the frontend
    always sends the correct integer cat.code to the model, regardless of
    how the dict was ordered or serialised.
    """
    # komoditas_map / provinsi_map  =>  {code(int): name}
    # Invert to {name: code} for the frontend
    komoditas_name2code = {v: k for k, v in komoditas_map.items()}
    provinsi_name2code  = {v: k for k, v in provinsi_map.items()}

    return jsonify({
        'status': 'success',
        'features_count': len(features),
        'features': features[:10],
        # legacy plain lists (kept for compatibility)
        'komoditas_list': komoditas_list,
        'provinsi_list': provinsi_list,
        # NEW: name->code dicts so frontend can look up exact cat.code
        'komoditas_map': komoditas_name2code,
        'provinsi_map':  provinsi_name2code,
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'status': 'success',
        'rmse': saved_metrics.get('rmse', 0),
        'mae':  saved_metrics.get('mae', 0),
        'mape': saved_metrics['mape'],
        'r2':   saved_metrics['r2']
    })

def _get_latest_row(provinsi_name: str, komoditas_name: str):
    """
    Helper: ambil baris terbaru dari df untuk kombinasi provinsi & komoditas.
    Mengembalikan Series fitur siap pakai, atau raise ValueError kalau tidak ditemukan.
    """
    # Cari kode integer dari nama
    prov_code = None
    for code, name in provinsi_map.items():
        if name.lower() == provinsi_name.lower():
            prov_code = code
            break
    kom_code = None
    for code, name in komoditas_map.items():
        if name.lower() == komoditas_name.lower():
            kom_code = code
            break

    if prov_code is None:
        raise ValueError(f"Provinsi '{provinsi_name}' tidak ditemukan")
    if kom_code is None:
        raise ValueError(f"Komoditas '{komoditas_name}' tidak ditemukan")

    subset = df[(df['provinsi'] == prov_code) & (df['komoditas'] == kom_code)]
    if subset.empty:
        raise ValueError(f"Tidak ada data untuk {provinsi_name} - {komoditas_name}")

    latest = subset.sort_values('tanggal').iloc[-1]
    return latest, prov_code, kom_code


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict harga 1 hari ke depan.
    Frontend cukup kirim: { "provinsi": "Jawa Barat", "komoditas": "Beras" }
    Backend otomatis ambil fitur terbaru dari df.
    """
    try:
        data = request.json
        provinsi_name  = data.get('provinsi', '')
        komoditas_name = data.get('komoditas', '')

        latest, prov_code, kom_code = _get_latest_row(provinsi_name, komoditas_name)

        # Bangun input fitur dari baris terbaru
        input_data = {}
        for feature in features:
            if feature in latest.index:
                input_data[feature] = float(latest[feature]) if pd.notna(latest[feature]) else 0.0
            else:
                input_data[feature] = 0.0

        input_df   = pd.DataFrame([input_data])[features]
        prediction = float(model.predict(input_df)[0])

        confidence_lower = prediction * 0.92
        confidence_upper = prediction * 1.08

        return jsonify({
            'status':                      'success',
            'provinsi':                    provinsi_name,
            'komoditas':                   komoditas_name,
            'prediction':                  prediction,
            'prediction_formatted':        f"Rp {prediction:,.0f}",
            'confidence_lower':            float(confidence_lower),
            'confidence_upper':            float(confidence_upper),
            'confidence_lower_formatted':  f"Rp {confidence_lower:,.0f}",
            'confidence_upper_formatted':  f"Rp {confidence_upper:,.0f}",
            'last_data_date':              latest['tanggal'].strftime('%Y-%m-%d'),
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/get-latest-data', methods=['GET'])
def get_latest_data():
    """
    Get latest historical data for chart.
    Params: provinsi (nama), komoditas (nama), days (opsional, default 30)
    """
    try:
        provinsi_name  = request.args.get('provinsi', '')
        komoditas_name = request.args.get('komoditas', '')
        days           = int(request.args.get('days', 30))

        latest, prov_code, kom_code = _get_latest_row(provinsi_name, komoditas_name)

        df_filtered = df[
            (df['komoditas'] == kom_code) &
            (df['provinsi']  == prov_code)
        ].sort_values('tanggal').tail(days)

        chart_data = {
            'dates':          df_filtered['tanggal'].dt.strftime('%Y-%m-%d').tolist(),
            'prices':         df_filtered['harga'].tolist(),
            'produsen_prices': df_filtered['harga_produsen'].tolist(),
        }

        return jsonify({'status': 'success', 'data': chart_data})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/predict-week', methods=['POST'])
def predict_week():
    """
    Predict harga N hari ke depan dengan rolling window.
    Frontend cukup kirim: { "provinsi": "Jawa Barat", "komoditas": "Beras", "horizon": 7 }
    Backend otomatis ambil fitur terbaru dari df.
    """
    try:
        data           = request.json
        provinsi_name  = data.get('provinsi', '')
        komoditas_name = data.get('komoditas', '')
        horizon        = int(data.get('horizon', 7))

        latest, prov_code, kom_code = _get_latest_row(provinsi_name, komoditas_name)

        # Bangun base_features dari baris terbaru
        base_features = {}
        for feature in features:
            if feature in latest.index:
                base_features[feature] = float(latest[feature]) if pd.notna(latest[feature]) else 0.0
            else:
                base_features[feature] = 0.0

        # Sliding window harga konsumen (7 hari ke belakang)
        price_window = [
            base_features.get('harga_harian_konsumen_lag_7', base_features.get('harga_harian_konsumen_lag_1', 0)),
            base_features.get('harga_harian_konsumen_lag_7', base_features.get('harga_harian_konsumen_lag_1', 0)),
            base_features.get('harga_harian_konsumen_lag_7', base_features.get('harga_harian_konsumen_lag_1', 0)),
            base_features.get('harga_harian_konsumen_lag_7', base_features.get('harga_harian_konsumen_lag_1', 0)),
            base_features.get('harga_harian_konsumen_lag_3', base_features.get('harga_harian_konsumen_lag_1', 0)),
            base_features.get('harga_harian_konsumen_lag_3', base_features.get('harga_harian_konsumen_lag_1', 0)),
            base_features.get('harga_harian_konsumen_lag_1', 0),
        ]

        prod_window = [
            base_features.get('harga_harian_produsen_lag_7', base_features.get('harga_harian_produsen_lag_1', 0)),
            base_features.get('harga_harian_produsen_lag_7', base_features.get('harga_harian_produsen_lag_1', 0)),
            base_features.get('harga_harian_produsen_lag_7', base_features.get('harga_harian_produsen_lag_1', 0)),
            base_features.get('harga_harian_produsen_lag_7', base_features.get('harga_harian_produsen_lag_1', 0)),
            base_features.get('harga_harian_produsen_lag_3', base_features.get('harga_harian_produsen_lag_1', 0)),
            base_features.get('harga_harian_produsen_lag_3', base_features.get('harga_harian_produsen_lag_1', 0)),
            base_features.get('harga_harian_produsen_lag_1', 0),
        ]

        base_date_dow  = base_features.get('dayofweek', 0)
        base_date_woy  = base_features.get('weekofyear', 1)
        base_month_sin = base_features.get('month_sin', 0)
        base_month_cos = base_features.get('month_cos', 1)

        predictions = []
        for i in range(horizon):
            current_features = base_features.copy()

            # Update lag & MA dari rolling window
            current_features['harga_harian_konsumen_lag_1']  = price_window[-1]
            current_features['harga_harian_konsumen_lag_3']  = price_window[-3]
            current_features['harga_harian_konsumen_lag_7']  = price_window[-7] if len(price_window) >= 7 else price_window[0]
            current_features['harga_harian__konsumen_ma_7']  = float(np.mean(price_window[-7:]))
            current_features['harga_harian_produsen_lag_1']  = prod_window[-1]
            current_features['harga_harian_produsen_lag_3']  = prod_window[-3]
            current_features['harga_harian_produsen_lag_7']  = prod_window[-7] if len(prod_window) >= 7 else prod_window[0]
            current_features['harga_harian__produsen_ma_7']  = float(np.mean(prod_window[-7:]))
            current_features['std_7_harga']                  = float(np.std(price_window[-7:]))
            current_features['std_7_harga_produsen']         = float(np.std(prod_window[-7:]))

            # Update fitur kalender
            future_dow = (base_date_dow + i + 1) % 7
            future_woy = base_date_woy + (base_date_dow + i + 1) // 7
            current_features['dayofweek']  = future_dow
            current_features['weekofyear'] = future_woy
            current_features['month_sin']  = base_month_sin
            current_features['month_cos']  = base_month_cos

            input_df   = pd.DataFrame([current_features])[features]
            pred_price = float(model.predict(input_df)[0])

            predictions.append({
                'day':             i + 1,
                'price':           pred_price,
                'price_formatted': f"Rp {pred_price:,.0f}",
            })

            # Geser window
            price_window.append(pred_price)
            prod_window.append(pred_price * (prod_window[-1] / price_window[-2] if price_window[-2] != 0 else 0.8))

        return jsonify({
            'status':          'success',
            'provinsi':        provinsi_name,
            'komoditas':       komoditas_name,
            'horizon':         horizon,
            'last_data_date':  latest['tanggal'].strftime('%Y-%m-%d'),
            'predictions':     predictions,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400




# =========================================================
# SERVE HALAMAN HTML
# =========================================================

@app.route('/')
@app.route('/<path:filename>')
def serve_html(filename='utama.html'):
    # Hanya izinkan file .html dan .css untuk keamanan
    if not filename.endswith(('.html', '.css', '.js')):
        return "Not found", 404
    return send_from_directory('.', filename)

_geojson_cache = None  # Raw string, baca sekali saja

@app.route('/api/geojson', methods=['GET'])
def get_geojson():
    """Serve file shp_simplified.geojson — di-cache di memori, tidak baca disk ulang."""
    global _geojson_cache
    try:
        if _geojson_cache is None:
            with open('shp_simplified.geojson', 'r', encoding='utf-8') as f:
                _geojson_cache = f.read()
            print(f"[geojson] Loaded dari disk ({len(_geojson_cache)//1024} KB)")
        return Response(_geojson_cache, mimetype='application/json')
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'GeoJSON tidak ditemukan'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def _compute_dvi_per_provinsi():
    """
    Helper: hitung DVI per provinsi dari data terbaru di df dan data NCPR dari CSV FSVA.
    Mengembalikan dict { provinsi_name: { dvi, climate_risk, price_volatility, supply_gap, alert_level } }
    """
    import glob

    # ── 1. ClimateRisk per provinsi ──────────────────────────────────────────
    # Ambil curah hujan terbaru (bulan terakhir per provinsi)
    latest_hujan = (
        df.sort_values('tanggal')
          .groupby('provinsi')
          .tail(30)
          .groupby('provinsi')['curah_hujan']
          .mean()
    )
    # Rata-rata historis per provinsi (seluruh data)
    historis_hujan = df.groupby('provinsi')['curah_hujan'].mean()

    climate_risk_map = {}
    for prov_code in latest_hujan.index:
        terbaru   = latest_hujan.get(prov_code, 0)
        historis  = historis_hujan.get(prov_code, 1)
        cr        = 1 - (terbaru / historis) if historis != 0 else 0
        cr        = max(0, cr)  # min 0
        climate_risk_map[prov_code] = cr

    # ── 2. PriceVolatility per provinsi ─────────────────────────────────────
    # Prediksi 30 hari ke depan untuk semua komoditas, ambil CV (std/mean)
    price_vol_map = {}
    for prov_code in df['provinsi'].unique():
        prov_name = provinsi_map.get(int(prov_code), '')
        cv_list   = []
        for kom_code in df['komoditas'].unique():
            kom_name = komoditas_map.get(int(kom_code), '')
            try:
                subset = df[
                    (df['provinsi']  == prov_code) &
                    (df['komoditas'] == kom_code)
                ].sort_values('tanggal')
                if subset.empty:
                    continue
                latest_row = subset.iloc[-1]

                # Bangun base features
                base_f = {}
                for feat in features:
                    base_f[feat] = float(latest_row[feat]) if feat in latest_row.index and pd.notna(latest_row[feat]) else 0.0

                # Rolling prediction 30 hari
                pw = [
                    base_f.get('harga_harian_konsumen_lag_7', base_f.get('harga_harian_konsumen_lag_1', 0)),
                    base_f.get('harga_harian_konsumen_lag_7', base_f.get('harga_harian_konsumen_lag_1', 0)),
                    base_f.get('harga_harian_konsumen_lag_7', base_f.get('harga_harian_konsumen_lag_1', 0)),
                    base_f.get('harga_harian_konsumen_lag_7', base_f.get('harga_harian_konsumen_lag_1', 0)),
                    base_f.get('harga_harian_konsumen_lag_3', base_f.get('harga_harian_konsumen_lag_1', 0)),
                    base_f.get('harga_harian_konsumen_lag_3', base_f.get('harga_harian_konsumen_lag_1', 0)),
                    base_f.get('harga_harian_konsumen_lag_1', 0),
                ]
                dw = list(pw)
                base_dow = base_f.get('dayofweek', 0)
                base_woy = base_f.get('weekofyear', 1)
                base_msin = base_f.get('month_sin', 0)
                base_mcos = base_f.get('month_cos', 1)

                preds = []
                for i in range(30):
                    cf = base_f.copy()
                    cf['harga_harian_konsumen_lag_1'] = pw[-1]
                    cf['harga_harian_konsumen_lag_3'] = pw[-3]
                    cf['harga_harian_konsumen_lag_7'] = pw[-7] if len(pw) >= 7 else pw[0]
                    cf['harga_harian__konsumen_ma_7'] = float(np.mean(pw[-7:]))
                    cf['harga_harian_produsen_lag_1'] = dw[-1]
                    cf['harga_harian_produsen_lag_3'] = dw[-3]
                    cf['harga_harian_produsen_lag_7'] = dw[-7] if len(dw) >= 7 else dw[0]
                    cf['harga_harian__produsen_ma_7'] = float(np.mean(dw[-7:]))
                    cf['std_7_harga']                 = float(np.std(pw[-7:]))
                    cf['std_7_harga_produsen']        = float(np.std(dw[-7:]))
                    cf['dayofweek']  = (base_dow + i + 1) % 7
                    cf['weekofyear'] = base_woy + (base_dow + i + 1) // 7
                    cf['month_sin']  = base_msin
                    cf['month_cos']  = base_mcos
                    inp  = pd.DataFrame([cf])[features]
                    pred = float(model.predict(inp)[0])
                    preds.append(pred)
                    pw.append(pred)
                    dw.append(pred * (dw[-1] / pw[-2] if pw[-2] != 0 else 0.8))

                mean_p = np.mean(preds)
                std_p  = np.std(preds)
                cv     = std_p / mean_p if mean_p != 0 else 0
                cv_list.append(cv)
            except Exception:
                continue

        price_vol_map[prov_code] = float(np.mean(cv_list)) if cv_list else 0.0

    # ── 3. SupplyGap per provinsi dari NCPR CSV FSVA ─────────────────────────
    supply_gap_map = {}  # { provinsi_name_lower: supply_gap }
    fsva_files = sorted(glob.glob("Dataset/kabupaten/fsva kabupaten *.csv"))
    if fsva_files:
        # Pakai tahun terbaru
        latest_file = fsva_files[-1]
        try:
            df_fsva = pd.read_csv(latest_file)
            for _, row in df_fsva.iterrows():
                wilayah = str(row.get('Wilayah', ''))
                parts   = wilayah.split(' - ', 1)
                prov    = parts[0].strip().title() if len(parts) > 1 else wilayah.strip().title()
                ncpr_raw = row.get('NCPR', None)
                try:
                    ncpr = float(str(ncpr_raw).replace(',', '.').replace('%', '').strip())
                    sg   = max(0, 1 - ncpr)
                    prov_lower = prov.lower()
                    if prov_lower not in supply_gap_map:
                        supply_gap_map[prov_lower] = []
                    supply_gap_map[prov_lower].append(sg)
                except Exception:
                    continue
            # Rata-rata per provinsi
            supply_gap_map = {k: float(np.mean(v)) for k, v in supply_gap_map.items()}
        except Exception as e:
            print(f"Gagal baca FSVA untuk SupplyGap: {e}")

    # ── 4. Normalisasi CR, PV, SG ke [0,1] min-max across semua provinsi ──
    def minmax_normalize(d):
        vals = [v for v in d.values() if v is not None]
        if not vals: return d
        mn, mx = min(vals), max(vals)
        if mx == mn: return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    # Buat dict prov_name -> raw value untuk normalisasi
    cr_raw = {}
    pv_raw = {}
    sg_raw = {}
    for prov_code in df['provinsi'].unique():
        prov_name = provinsi_map.get(int(prov_code), '')
        cr_raw[prov_name] = climate_risk_map.get(prov_code, 0)
        pv_raw[prov_name] = price_vol_map.get(prov_code, 0)
        sg_raw[prov_name] = supply_gap_map.get(prov_name.lower(), 0)

    cr_norm = minmax_normalize(cr_raw)
    pv_norm = minmax_normalize(pv_raw)
    sg_norm = minmax_normalize(sg_raw)

    # ── 5. Gabung jadi DVI per provinsi ──────────────────────────────────────
    def alert_level(dvi):
        if dvi < 0.3:  return 'Aman'
        if dvi < 0.5:  return 'Waspada'
        if dvi < 0.7:  return 'Siaga'
        return 'Kritis'

    result = {}
    for prov_name in cr_norm:
        cr  = cr_norm.get(prov_name, 0)
        pv  = pv_norm.get(prov_name, 0)
        sg  = sg_norm.get(prov_name, 0)
        dvi = round((cr + pv + sg) / 3, 4)

        result[prov_name] = {
            'dvi':               dvi,
            'climate_risk':      round(cr, 4),
            'price_volatility':  round(pv, 4),
            'supply_gap':        round(sg, 4),
            'alert_level':       alert_level(dvi),
        }

    return result


@app.route('/api/ews', methods=['GET'])
def get_ews():
    """
    Hitung DVI (Dynamic Vulnerability Index) per provinsi.
    Returns: { provinsi: { dvi, climate_risk, price_volatility, supply_gap, alert_level } }
    """
    try:
        dvi_data = _compute_dvi_per_provinsi()
        return jsonify({'status': 'success', 'data': dvi_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/fsva-data', methods=['GET'])
def get_fsva_data():
    """
    Return FSVA nowcast dari cache. Hitung ulang kalau cache kosong.
    Hanya kirim field yang dibutuhkan peta untuk hemat bandwidth.
    """
    try:
        if not _fsva_cache:
            _load_fsva_cache()
        if not _fsva_cache:
            print("[fsva-data] Cache kosong, menghitung...")
            _build_fsva_cache()

        # Slim down: hanya field yang dipakai peta.html
        SLIM_FIELDS = {'Kabupaten', 'Provinsi', 'nowcast_score', 'fsva_score', 'delta', 'nowcast_kelas', 'kelas', 'delta_breakdown'}
        slim_cache = {
            str(tahun): [
                {k: v for k, v in rec.items() if k in SLIM_FIELDS}
                for rec in records
            ]
            for tahun, records in _fsva_cache.items()
        }

        return jsonify({'status': 'success', 'data': slim_cache})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _build_fsva_cache():
    """Hitung FSVA nowcast untuk semua tahun & simpan ke cache."""
    global _fsva_cache
    import glob as _g

    BOBOT = {
        'NCPR': 1/9,
        'Kemiskinan (%)': 1/27,
        'Pengeluaran Pangan (%)': 1/27,
        'Tanpa Listrik (%)': 1/27,
        'Tanpa Air Bersih (%)': 1/45,
        'Lama Sekolah Perempuan (tahun)': 1/45,
        'Rasio Tenaga Kesehatan': 1/45,
        'Angka Harapan Hidup (tahun)': 1/45,
        'Stunting (%)': 1/45,
    }
    NEG_COLS = ['NCPR','Kemiskinan (%)','Pengeluaran Pangan (%)','Rasio Tenaga Kesehatan',
                'Tanpa Listrik (%)','Tanpa Air Bersih (%)','Stunting (%)']
    POS_COLS = ['Lama Sekolah Perempuan (tahun)','Angka Harapan Hidup (tahun)']

    def klasifikasi(score):
        if score <= 1.5: return 'Sangat Rentan'
        if score <= 2.5: return 'Rentan'
        if score <= 3.5: return 'Agak Rentan'
        if score <= 4.5: return 'Agak Tahan'
        if score <= 5.5: return 'Tahan'
        return 'Sangat Tahan'

    def minmax_score(vals, invert):
        nums = [v for v in vals if v is not None and not np.isnan(v)]
        if not nums: return [None] * len(vals)
        mn, mx = min(nums), max(nums)
        if mx == mn: return [3.0] * len(vals)
        result = []
        for v in vals:
            if v is None or np.isnan(v):
                result.append(None)
            elif invert:
                result.append(1 + 5 * (mx - v) / (mx - mn))
            else:
                result.append(1 + 5 * (v - mn) / (mx - mn))
        return result

    def process_df(df_raw, delta_map):
        records = []
        for _, r in df_raw.iterrows():
            wilayah = str(r.get('Wilayah', ''))
            parts = wilayah.split(' - ', 1)
            prov = parts[0].strip().title() if len(parts) > 1 else ''
            kab  = parts[1].strip().title() if len(parts) > 1 else wilayah.strip().title()

            def pn(col):
                v = r.get(col, None)
                if v is None or (isinstance(v, float) and np.isnan(v)): return None
                try: return float(str(v).replace(',','.').replace('%','').strip())
                except: return None

            records.append({
                'Kabupaten': kab,
                'Provinsi':  prov,
                'NCPR':                          pn('NCPR'),
                'Kemiskinan (%)':                pn('Kemiskinan (%)'),
                'Pengeluaran Pangan (%)':        pn('Pengeluaran Pangan (%)'),
                'Tanpa Listrik (%)':             pn('Tanpa Listrik (%)'),
                'Tanpa Air Bersih (%)':          pn('Tanpa Air Bersih (%)'),
                'Lama Sekolah Perempuan (tahun)':pn('Lama Sekolah Perempuan (tahun)'),
                'Rasio Tenaga Kesehatan':        pn('Rasio Tenaga Kesehatan'),
                'Angka Harapan Hidup (tahun)':   pn('Angka Harapan Hidup (tahun)'),
                'Stunting (%)':                  pn('Stunting (%)'),
            })

        skor = {}
        for col in NEG_COLS + POS_COLS:
            vals = [rec[col] for rec in records]
            if any(v is not None for v in vals):
                skor[col] = minmax_score(vals, col in NEG_COLS)

        total_bobot = sum(BOBOT[c] for c in skor)

        for i, rec in enumerate(records):
            score = 0
            for col, s in skor.items():
                if s[i] is not None:
                    score += s[i] * BOBOT[col]
            fsva_struktural = round(score / total_bobot, 4) if total_bobot else 0
            rec['fsva_score'] = fsva_struktural
            rec['kelas']      = klasifikasi(fsva_struktural)

            prov_name = rec['Provinsi']
            sinyal    = delta_map.get(prov_name, {})
            pct_harga = sinyal.get('pct_harga', 0)
            ch_pct    = sinyal.get('ch_pct', 0)
            ncpr_val  = rec.get('NCPR', 1.0) or 1.0

            skor_harga = 0.0
            skor_cuaca = 0.0
            skor_ncpr  = 0.0

            # ── Harga: dari pct_harga (tren prediksi) ──
            # Harga naik tajam = kondisi memburuk → delta turun
            if pct_harga > 25:   skor_harga = -0.6
            elif pct_harga > 15: skor_harga = -0.4
            elif pct_harga > 10: skor_harga = -0.2

            # Harga turun tajam = kondisi membaik → delta naik
            elif pct_harga < -25: skor_harga = 0.6
            elif pct_harga < -15: skor_harga = 0.4
            elif pct_harga < -10: skor_harga = 0.2

            # ── Cuaca: curah hujan ekstrem (kekeringan/banjir) = memburuk → delta turun ──
            if ch_pct < -30:   skor_cuaca = -0.5
            elif ch_pct < -20: skor_cuaca = -0.3
            elif ch_pct > 50:  skor_cuaca = -0.2

            # Curah hujan normal/kondusif = kondisi membaik → delta naik
            elif -10 <= ch_pct <= 10: skor_cuaca = 0.2

            # ── NCPR: rasio rendah = kondisi memburuk → delta turun ──
            if ncpr_val < 0.5:   skor_ncpr = -0.5
            elif ncpr_val < 0.7: skor_ncpr = -0.3
            elif ncpr_val < 1.0: skor_ncpr = -0.1

            # NCPR tinggi = ketahanan/ketersediaan baik → delta naik
            elif ncpr_val > 1.5:  skor_ncpr = 0.5
            elif ncpr_val > 1.2:  skor_ncpr = 0.3
            elif ncpr_val > 1.0:  skor_ncpr = 0.1

            delta = round(skor_harga + skor_cuaca + skor_ncpr, 4)
            nowcast = round(max(1.0, min(6.0, fsva_struktural + delta)), 4)
            rec['nowcast_score'] = nowcast
            rec['nowcast_kelas'] = klasifikasi(nowcast)
            rec['delta']         = delta
            rec['delta_breakdown'] = {
                'harga': round(skor_harga, 4),
                'cuaca': round(skor_cuaca, 4),
                'ncpr':  round(skor_ncpr, 4),
            }

        return records

    # Bangun delta_map
    delta_map = {}
    try:
        KOM_LIST = ['Cabai Rawit Merah', 'Daging Ayam Ras', 'Daging Sapi Murni', 'Telur Ayam Ras']
        pct_per_prov = {}
        as_per_prov  = {}
        for kom in KOM_LIST:
            cached = _predict_all_cache.get(kom)
            if not cached:
                print(f'[fsva-build] Cache {kom} belum siap, skip')
                continue
            for prov_name, val in cached['data'].items():
                pct = val.get('pct_change', 0)
                ast = val.get('anomaly_score', 0) or 0
                pct_per_prov.setdefault(prov_name, []).append(abs(pct))
                as_per_prov.setdefault(prov_name, []).append(ast)

        ch_pct_prov = {}
        try:
            latest_hujan   = df.sort_values('tanggal').groupby('provinsi').tail(30).groupby('provinsi')['curah_hujan'].mean()
            historis_hujan = df.groupby('provinsi')['curah_hujan'].mean()
            for prov_code in latest_hujan.index:
                terbaru  = float(latest_hujan.get(prov_code, 0))
                historis = float(historis_hujan.get(prov_code, 1))
                pct_ch   = round((terbaru / historis * 100) - 100, 1) if historis != 0 else 0
                prov_nm  = provinsi_map.get(int(prov_code), '')
                if prov_nm: ch_pct_prov[prov_nm] = pct_ch
        except Exception as e_ch:
            print(f'[fsva-build] Gagal hitung curah hujan: {e_ch}')

        all_provs = set(list(pct_per_prov.keys()) + list(ch_pct_prov.keys()))
        for prov in all_provs:
            delta_map[prov] = {
                'pct_harga':     float(max(pct_per_prov[prov])) if prov in pct_per_prov else 0.0,
                'anomaly_score': float(max(as_per_prov[prov]))  if prov in as_per_prov  else 0.0,
                'ch_pct':        ch_pct_prov.get(prov, 0.0),
            }
        print(f'[fsva-build] delta_map: {len(delta_map)} provinsi')
    except Exception as e:
        print(f'[fsva-build] Gagal delta_map: {e}')

    result = {}
    files  = _g.glob("Dataset/kabupaten/fsva kabupaten *.csv")
    for path in sorted(files):
        match = re.search(r'(\d{4})', os.path.basename(path))
        if not match: continue
        tahun = int(match.group(1))
        try:
            df_raw        = pd.read_csv(path)
            result[tahun] = process_df(df_raw, delta_map)
        except Exception as e:
            print(f"[fsva-build] Gagal baca {path}: {e}")

    _fsva_cache.update(result)
    _save_fsva_cache()
    print(f"[fsva-build] Selesai: {len(result)} tahun, total {sum(len(v) for v in result.values())} kabupaten")


# =========================================================
# ENDPOINT BARU 1: PREDICT ALL PROVINSI (untuk heatmap + top N)
# =========================================================

@app.route('/api/predict-all-provinsi', methods=['GET'])
def predict_all_provinsi():
    """
    Prediksi harga semua provinsi untuk 1 komoditas.
    Pakai cache precompute jika tersedia, hitung ulang jika tidak.
    """
    try:
        komoditas_name = request.args.get('komoditas', '')
        horizon        = int(request.args.get('horizon', 28))

        if not komoditas_name:
            return jsonify({'status': 'error', 'message': 'Parameter komoditas wajib diisi'}), 400

        # Cek cache dulu
        cached = _predict_all_cache.get(komoditas_name)
        if cached:  # cache berlaku selamanya, hapus file json untuk reset manual
            return jsonify({
                'status':    'success',
                'komoditas': komoditas_name,
                'horizon':   horizon,
                'data':      cached['data'],
                'from_cache': True,
            })

        # Cache belum ada → hitung dulu
        print(f"[predict-all] Cache miss untuk {komoditas_name}, menghitung...")
        data = _compute_predict_all(komoditas_name, horizon)
        _predict_all_cache[komoditas_name] = {'data': data, 'ts': _time.time()}

        return jsonify({
            'status':    'success',
            'komoditas': komoditas_name,
            'horizon':   horizon,
            'data':      data,
            'from_cache': False,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


# =========================================================

# =========================================================
# COMPUTE EWS HORIZON
# =========================================================

def _compute_and_cache_ews():
    import glob as _glob
    try:
        latest_hujan   = (df.sort_values("tanggal")
                           .groupby("provinsi").tail(30)
                           .groupby("provinsi")["curah_hujan"].mean())
        historis_hujan = df.groupby("provinsi")["curah_hujan"].mean()

        climate_risk_map    = {}
        curah_hujan_pct_map = {}
        for prov_code in latest_hujan.index:
            terbaru  = float(latest_hujan.get(prov_code, 0))
            historis = float(historis_hujan.get(prov_code, 1))
            cr  = max(0.0, 1.0 - (terbaru / historis)) if historis != 0 else 0.0
            pct = round((terbaru / historis * 100) - 100, 1) if historis != 0 else 0
            climate_risk_map[prov_code] = cr
            prov_name = provinsi_map.get(int(prov_code), "")
            curah_hujan_pct_map[prov_name] = pct

        supply_gap_map = {}
        ncpr_map       = {}
        fsva_files = sorted(_glob.glob("Dataset/kabupaten/fsva kabupaten *.csv"))
        if fsva_files:
            try:
                df_fsva = pd.read_csv(fsva_files[-1])
                for _, row in df_fsva.iterrows():
                    wilayah  = str(row.get("Wilayah", ""))
                    parts    = wilayah.split(" - ", 1)
                    prov     = parts[0].strip().title() if len(parts) > 1 else wilayah.strip().title()
                    ncpr_raw = row.get("NCPR", None)
                    try:
                        ncpr = float(str(ncpr_raw).replace(",", ".").strip())
                        sg   = max(0.0, 1.0 - ncpr)
                        pk   = prov.lower()
                        if pk not in supply_gap_map:
                            supply_gap_map[pk] = []
                            ncpr_map[pk]       = []
                        supply_gap_map[pk].append(sg)
                        ncpr_map[pk].append(ncpr)
                    except:
                        continue
                supply_gap_map = {k: float(np.mean(v)) for k, v in supply_gap_map.items()}
                ncpr_map       = {k: round(float(np.mean(v)), 3) for k, v in ncpr_map.items()}
            except Exception as e:
                print(f"[ews] Gagal baca FSVA: {e}")

        HORIZONS    = {"h7": 7, "h14": 14, "h21": 21}
        STATUS_ORD  = ["Aman", "Waspada", "Siaga", "Kritis"]

        def calc_pv(prov_code, horizon_days):
            cv_list = []
            pct_per_kom = {}
            for kom_code in df["komoditas"].unique():
                kom_name = komoditas_map.get(int(kom_code), "")
                try:
                    mask = (df["provinsi"] == prov_code) & (df["komoditas"] == kom_code)
                    df_f = df[mask].sort_values("tanggal")
                    if df_f.empty: continue
                    lr  = df_f.iloc[-1]
                    ht  = float(lr["harga"])
                    pw  = df_f["harga"].tail(7).tolist()
                    dw  = df_f["harga_produsen"].tail(7).tolist()
                    while len(pw) < 7: pw.insert(0, pw[0] if pw else 0)
                    while len(dw) < 7: dw.insert(0, dw[0] if dw else 0)
                    bf  = {feat: float(lr[feat]) if feat in lr.index and pd.notna(lr[feat]) else 0.0 for feat in features}
                    bdow = int(lr["dayofweek"])
                    bwoy = int(lr["weekofyear"])
                    preds = []
                    for i in range(horizon_days):
                        cf = bf.copy()
                        cf["harga_harian_konsumen_lag_1"] = pw[-1]
                        cf["harga_harian_konsumen_lag_3"] = pw[-3]
                        cf["harga_harian_konsumen_lag_7"] = pw[-7]
                        cf["harga_harian__konsumen_ma_7"] = float(np.mean(pw[-7:]))
                        cf["std_7_harga"]                 = float(np.std(pw[-7:]))
                        cf["harga_harian_produsen_lag_1"] = dw[-1]
                        cf["harga_harian_produsen_lag_3"] = dw[-3]
                        cf["harga_harian_produsen_lag_7"] = dw[-7]
                        cf["harga_harian__produsen_ma_7"] = float(np.mean(dw[-7:]))
                        cf["std_7_harga_produsen"]        = float(np.std(dw[-7:]))
                        cf["dayofweek"]  = (bdow + i + 1) % 7
                        cf["weekofyear"] = bwoy + (bdow + i + 1) // 7
                        pred = float(model.predict(pd.DataFrame([cf])[features])[0])
                        preds.append(pred)
                        pw.append(pred); pw = pw[-7:]
                        dw.append(pred * (dw[-1] / pw[-2] if pw[-2] != 0 else 0.8)); dw = dw[-7:]
                    mp = np.mean(preds)
                    sp = np.std(preds)
                    cv_list.append(sp / mp if mp != 0 else 0)
                    pct = ((mp - ht) / ht * 100) if ht else 0
                    pct_per_kom[kom_name] = round(pct, 2)
                except:
                    continue
            return float(np.mean(cv_list)) if cv_list else 0.0, pct_per_kom

        def status_dvi(dvi):
            if dvi < 0.3: return "Aman"
            if dvi < 0.5: return "Waspada"
            if dvi < 0.7: return "Siaga"
            return "Kritis"

        def indikator_dominan(cr, pv, sg, pct_kom, ch_pct, ncpr):
            # Urutkan 3 komponen dari dominan ke tidak
            komponen = sorted([
                ("ClimateRisk", cr),
                ("PriceVolatility", pv),
                ("SupplyGap", sg),
            ], key=lambda x: x[1], reverse=True)

            hasil = []
            for nama, nilai in komponen:
                if nama == "PriceVolatility" and pct_kom:
                    top  = max(pct_kom.items(), key=lambda x: x[1])
                    nm   = top[0].replace(" Merah","").replace(" Ras","").replace(" Murni","")
                    hasil.append(f"Harga {nm} +{top[1]:.1f}%")
                elif nama == "ClimateRisk":
                    hasil.append(f"Curah hujan {ch_pct:+.1f}% dari normal")
                else:
                    hasil.append(f"NCPR {ncpr:.2f}")
            return hasil  # list 3 item

        # ── Kumpulkan raw values dulu untuk normalisasi ──
        raw_cr = {}
        raw_sg = {}
        raw_pv = {h: {} for h in HORIZONS}
        pct_kom_per_prov = {}
        prov_list_ews = list(df["provinsi"].unique())

        for prov_code in prov_list_ews:
            prov_name = provinsi_map.get(int(prov_code), "")
            raw_cr[prov_name] = climate_risk_map.get(prov_code, 0.0)
            raw_sg[prov_name] = supply_gap_map.get(prov_name.lower(), 0.0)
            pct_kom_per_prov[prov_name] = {}
            for h_key, h_days in HORIZONS.items():
                pv, pct_kom = calc_pv(prov_code, h_days)
                raw_pv[h_key][prov_name] = pv
                if h_key == "h7":
                    pct_kom_per_prov[prov_name] = pct_kom

        # ── Normalisasi min-max per komponen ──
        def _mm(d):
            vals = list(d.values())
            mn, mx = min(vals), max(vals)
            if mx == mn: return {k: 0.0 for k in d}
            return {k: (v - mn) / (mx - mn) for k, v in d.items()}

        norm_cr = _mm(raw_cr)
        norm_sg = _mm(raw_sg)
        norm_pv = {h: _mm(raw_pv[h]) for h in HORIZONS}

        result = {}
        for prov_code in prov_list_ews:
            prov_name = provinsi_map.get(int(prov_code), "")
            cr   = norm_cr.get(prov_name, 0.0)
            sg   = norm_sg.get(prov_name, 0.0)
            ncpr = ncpr_map.get(prov_name.lower(), 0.0)
            ch_pct = curah_hujan_pct_map.get(prov_name, 0.0)
            pct_kom_h7 = pct_kom_per_prov.get(prov_name, {})

            h_result   = {}
            for h_key, h_days in HORIZONS.items():
                pv  = norm_pv[h_key].get(prov_name, 0.0)
                dvi = round((cr + pv + sg) / 3, 4)
                h_result[h_key] = {"dvi": dvi, "status": status_dvi(dvi), "pv": round(pv, 4)}

            all_statuses = [h_result[h]["status"] for h in HORIZONS]
            level = max(all_statuses, key=lambda s: STATUS_ORD.index(s))

            dl_map   = {"h7": "H-7", "h14": "H-14", "h21": "H-21"}
            deadline = "H-21"
            for h_key in ["h7", "h14", "h21"]:
                if h_result[h_key]["status"] == level:
                    deadline = dl_map[h_key]
                    break

            indikator_list = indikator_dominan(cr, h_result["h7"]["pv"], sg, pct_kom_h7, ch_pct, ncpr)
            result[prov_name] = {
                "h7": h_result["h7"], "h14": h_result["h14"], "h21": h_result["h21"],
                "level": level, "deadline": deadline, "indikator": indikator_list,
                "detail": {
                    "climate_risk": round(cr, 4), "supply_gap": round(sg, 4),
                    "ncpr": ncpr, "curah_hujan_pct": ch_pct, "harga_komoditas": pct_kom_h7,
                }
            }

        sorted_result = dict(sorted(result.items(), key=lambda x: STATUS_ORD.index(x[1]["level"]), reverse=True))
        _ews_cache.update(sorted_result)
        _save_ews_cache()
        print(f"[ews] Selesai: {len(sorted_result)} provinsi")

    except Exception as e:
        print(f"[ews] Error saat compute: {e}")


@app.route('/api/ews-horizon', methods=['GET'])
def get_ews_horizon():
    """Return EWS horizon dari cache. Hitung ulang jika cache kosong."""
    try:
        # Load cache dari file kalau belum ada di memori
        if not _ews_cache:
            _load_ews_cache()

        # Kalau masih kosong, hitung sekarang
        if not _ews_cache:
            print("[ews-horizon] Cache kosong, menghitung...")
            _compute_and_cache_ews()

        return jsonify({
            'status': 'success',
            'data':   _ews_cache,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400



# ======================
# ADMIN ENDPOINTS
# ======================

@app.route('/admin/login', methods=['GET'])
def admin_login_page():
    if session.get('admin_logged_in'):
        return redirect('/admin')
    return send_from_directory('.', 'admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login():
    data     = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        session['admin_username']  = username
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Username atau password salah'}), 401

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.clear()
    return jsonify({'status': 'success'})

@app.route('/admin', methods=['GET'])
@app.route('/admin/', methods=['GET'])
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect('/admin/login')
    return send_from_directory('.', 'admin.html')

@app.route('/admin/check', methods=['GET'])
def admin_check():
    if not session.get('admin_logged_in'):
        return jsonify({'status': 'error'}), 401
    return jsonify({'status': 'ok', 'username': session.get('admin_username', 'admin')})

@app.route('/admin/status', methods=['GET'])
@admin_required
def admin_status():
    try:
        import json as _j
        last_date  = str(df['tanggal'].max().date()) if not df.empty else '—'
        total_rows = len(df)
        fsva_files = sorted([f for f in os.listdir(DIR_FSVA) if f.endswith('.csv')])
        last_year  = '—'
        total_kab  = 0
        if fsva_files:
            m = re.search(r'(\d{4})', fsva_files[-1])
            if m: last_year = m.group(1)
            try:
                total_kab = len(pd.read_csv(os.path.join(DIR_FSVA, fsva_files[-1])))
            except Exception as _e_sb:
                print(f"[supabase] Gagal upload/sync: {_e_sb}")
        mape, r2 = '—', '—'
        try:
            with open('model_metrics.json') as f:
                met  = _j.load(f)
                mape = round(met.get('mape', 0), 2)
                r2   = round(met.get('r2', 0), 4)
        except Exception as _e_sb:
            print(f"[supabase] Gagal upload/sync: {_e_sb}")
        return jsonify({
            'status': 'success',
            'last_harga_date':  last_date,
            'total_harga_rows': total_rows,
            'last_fsva_year':   last_year,
            'total_fsva_kab':   total_kab,
            'model_mape':       mape,
            'model_r2':         r2,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── List file per jenis ──
@app.route('/admin/files/<jenis>', methods=['GET'])
@admin_required
def admin_list_files(jenis):
    dirs = {'harga': DIR_HARGA, 'fsva': DIR_FSVA, 'hujan': DIR_HUJAN}
    if jenis not in dirs:
        return jsonify({'status': 'error', 'message': 'Jenis tidak valid'}), 400
    try:
        folder = dirs[jenis]
        files  = []
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.csv'): continue
            fpath = os.path.join(folder, fname)
            stat  = os.stat(fpath)
            files.append({
                'name':     fname,
                'size_kb':  round(stat.st_size / 1024, 1),
                'modified': pd.Timestamp(stat.st_mtime, unit='s').strftime('%d %b %Y %H:%M'),
            })
        return jsonify({'status': 'success', 'files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Hapus file ──
@app.route('/admin/files/<jenis>/<filename>', methods=['DELETE'])
@admin_required
def admin_delete_file(jenis, filename):
    global df
    dirs = {'harga': DIR_HARGA, 'fsva': DIR_FSVA, 'hujan': DIR_HUJAN}
    if jenis not in dirs:
        return jsonify({'status': 'error', 'message': 'Jenis tidak valid'}), 400
    try:
        fpath = os.path.join(dirs[jenis], filename)
        if not os.path.exists(fpath):
            return jsonify({'status': 'error', 'message': 'File tidak ditemukan'}), 404

        os.remove(fpath)

        # Hapus dari Supabase juga kalau ada
        sb = get_supabase()
        if sb:
            try:
                supabase_storage_remove([f'{jenis}/{filename}'])
            except Exception as _e_sb:
                print(f"[supabase] Gagal upload/sync: {_e_sb}")

        # Rebuild df_cache kalau hapus harga atau hujan
        if jenis in ['harga', 'hujan']:
            df, _, _ = load_and_preprocess_data()
            df.to_parquet('df_cache.parquet', index=False)

        # Invalidate cache hasil hitungan (predict/ews/fsva) — kalau tidak,
        # dashboard masih nampilin hasil lama sampai server di-restart
        global _predict_all_cache, _ews_cache, _fsva_cache
        for f_name in ['predict_all_cache.json', 'ews_cache.json', 'fsva_nowcast_cache.json']:
            if os.path.exists(f_name):
                os.remove(f_name)
        _predict_all_cache = {}
        _ews_cache         = {}
        _fsva_cache        = {}
        threading.Thread(target=precompute_all, daemon=True).start()

        return jsonify({'status': 'success', 'message': f'{filename} berhasil dihapus, cache sedang diperbarui'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Upload harga ──
# Mapping komoditas → nama file konsumen harian
KOMODITAS_FILE_MAP = {
    'cabai':  'Cabai Rawit Merah_Konsumen_Harian.csv',
    'ayam':   'Daging Ayam Ras_Konsumen_Harian.csv',
    'sapi':   'Daging Sapi Murni_Konsumen_Harian.csv',
    'telur':  'Telur Ayam Ras_Konsumen_Harian.csv',
}
KOMODITAS_VALID = {
    'cabai':  'Cabai Rawit Merah',
    'ayam':   'Daging Ayam Ras',
    'sapi':   'Daging Sapi Murni',
    'telur':  'Telur Ayam Ras',
}

def _upload_harga_komoditas(komoditas_key):
    """Helper upload harga per komoditas."""
    import io
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Tidak ada file'}), 400

        file    = request.files['file']
        content = file.read().decode('utf-8-sig')
        df_new  = pd.read_csv(io.StringIO(content))

        # Validasi kolom
        missing = [c for c in ['tanggal','provinsi','komoditas','harga'] if c not in df_new.columns]
        if missing:
            return jsonify({'status': 'error', 'message': f'Kolom tidak lengkap: {missing}'}), 400

        # Validasi komoditas
        valid_kom = KOMODITAS_VALID[komoditas_key]
        invalid   = df_new[df_new['komoditas'] != valid_kom]
        if not invalid.empty:
            return jsonify({'status': 'error', 'message': f'Komoditas harus "{valid_kom}", ditemukan nilai lain'}), 400

        # Simpan file asli ke folder uploads/komoditas
        # (file ini BERDIRI SENDIRI, tidak digabung/ditulis ulang ke file lain
        # -- load_and_preprocess_data() yang menggabungkannya di memori setiap
        # kali data di-load, sama seperti pola FSVA. Ini bikin hapus 1 file
        # tidak pernah bisa merusak/mengubah file lain.)
        upload_dir  = DIR_UPLOADS[komoditas_key]
        save_name   = file.filename  # nama file bebas dari admin
        backup_path = os.path.join(upload_dir, save_name)
        df_new.to_csv(backup_path, index=False)

        # Simpan metadata
        meta = load_upload_metadata()
        if komoditas_key not in meta:
            meta[komoditas_key] = []
        meta[komoditas_key].append({
            'filename':   save_name,
            'rows':       len(df_new),
            'uploaded_at': _time.strftime('%Y-%m-%d %H:%M:%S'),
            'komoditas':  valid_kom,
        })
        save_upload_metadata(meta)

        # Simpan ke Supabase (permanen, biar gak hilang tiap restart/redeploy)
        sb = get_supabase()
        if sb:
            try:
                with open(backup_path, 'rb') as f_raw:
                    supabase_storage_upload(
                        f'uploads/{komoditas_key}/{save_name}',
                        f_raw.read(),
                        'text/csv'
                    )
                with open(UPLOAD_METADATA_FILE, 'rb') as f_meta:
                    supabase_storage_upload(
                        'meta/upload_metadata.json',
                        f_meta.read(),
                        'application/json'
                    )
            except Exception as _e_sb:
                print(f"[supabase] Gagal upload/sync: {_e_sb}")

        # Reload df + invalidate semua cache
        threading.Thread(target=reload_data_and_cache, daemon=True).start()
        return jsonify({
            'status': 'success',
            'message': f'{len(df_new)} baris {valid_kom} berhasil ditambahkan, cache sedang diperbarui',
            'filename': save_name,
            'rows': len(df_new),
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/admin/upload/harga/cabai', methods=['POST'])
@admin_required
def admin_upload_harga_cabai():
    return _upload_harga_komoditas('cabai')

@app.route('/admin/upload/harga/ayam', methods=['POST'])
@admin_required
def admin_upload_harga_ayam():
    return _upload_harga_komoditas('ayam')

@app.route('/admin/upload/harga/sapi', methods=['POST'])
@admin_required
def admin_upload_harga_sapi():
    return _upload_harga_komoditas('sapi')

@app.route('/admin/upload/harga/telur', methods=['POST'])
@admin_required
def admin_upload_harga_telur():
    return _upload_harga_komoditas('telur')

# Endpoint lama tetap ada untuk backward compatibility
@app.route('/admin/upload/harga', methods=['POST'])
@admin_required
def admin_upload_harga():
    return jsonify({'status': 'error', 'message': 'Gunakan endpoint per komoditas: /admin/upload/harga/cabai, /ayam, /sapi, /telur'}), 400


# =========================================================
# METADATA MANAGEMENT (tracking file upload per komoditas)
# =========================================================

def load_upload_metadata():
    if os.path.exists(UPLOAD_METADATA_FILE):
        try:
            with open(UPLOAD_METADATA_FILE, 'r', encoding='utf-8') as f:
                return _json.load(f)
        except:
            pass
    return {}

def save_upload_metadata(meta):
    with open(UPLOAD_METADATA_FILE, 'w', encoding='utf-8') as f:
        _json.dump(meta, f, ensure_ascii=False, indent=2)

# (rebuild_harga_file & rebuild_hujan_file dihapus -- sekarang setiap file
#  upload berdiri sendiri dan digabung di memori oleh
#  load_and_preprocess_data()/_load_base_plus_uploads(), sama seperti pola
#  FSVA. File dasar di Dataset/prediksi/*.csv tidak pernah ditulis ulang lagi,
#  jadi hapus 1 file upload tidak akan pernah merusak/mengubah data lain.)


# =========================================================
# RELOAD DATA + INVALIDATE CACHE (dipanggil setelah upload)
# =========================================================

def reload_data_and_cache():
    """
    Reload df dari CSV terbaru, hapus semua cache dashboard,
    lalu trigger precompute ulang di background.
    """
    global df, komoditas_map, provinsi_map, komoditas_name2code, provinsi_name2code
    try:
        print("[reload] Memuat ulang data dari CSV terbaru...")
        df_lama_jumlah = len(df) if 'df' in globals() and df is not None else 0
        df_baru, komoditas_map_baru, provinsi_map_baru = load_and_preprocess_data()

        # ── Pengaman: kalau hasil reload kosong/anomali drastis, JANGAN
        # dipakai -- pertahankan data lama yang masih valid. Tanpa ini,
        # sekali proses load gagal (mis. karena file CSV korup sesaat),
        # dashboard langsung nampilin 0 provinsi ke semua orang.
        if len(df_baru) == 0 or (df_lama_jumlah > 0 and len(df_baru) < df_lama_jumlah * 0.5):
            print(f"[reload] DITOLAK! df baru cuma {len(df_baru)} baris (sebelumnya {df_lama_jumlah}). "
                  f"Data lama DIPERTAHANKAN, cache TIDAK dihapus. Cek CSV sumbernya!")
            return

        df, komoditas_map, provinsi_map = df_baru, komoditas_map_baru, provinsi_map_baru
        komoditas_name2code = {v: k for k, v in komoditas_map.items()}
        provinsi_name2code  = {v: k for k, v in provinsi_map.items()}
        print(f"[reload] df berhasil dimuat: {len(df)} baris")
    except Exception as e:
        print(f"[reload] Gagal reload df: {e}")
        return

    # Hapus semua cache dashboard
    cache_files = [
        'predict_all_cache.json',
        'ews_cache.json',
        'fsva_nowcast_cache.json',
    ]
    for f_name in cache_files:
        if os.path.exists(f_name):
            os.remove(f_name)
            print(f"[reload] Cache dihapus: {f_name}")

    # Reset in-memory cache
    global _predict_all_cache, _ews_cache
    _predict_all_cache = {}
    _ews_cache         = {}

    # Precompute ulang di background
    print("[reload] Memulai precompute ulang...")
    threading.Thread(target=precompute_all, daemon=True).start()


# ── Upload FSVA ──
@app.route('/admin/upload/fsva', methods=['POST'])
@admin_required
def admin_upload_fsva():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Tidak ada file'}), 400
        file    = request.files['file']
        content = file.read().decode('utf-8')
        import io
        df_fsva = pd.read_csv(io.StringIO(content))
        missing = [c for c in ['Wilayah','NCPR'] if c not in df_fsva.columns]
        if missing:
            return jsonify({'status': 'error', 'message': f'Kolom tidak lengkap: {missing}'}), 400
        # Ekstrak tahun dari nama file (fsva_kabupaten_2024.csv → 2024)
        import re as _re
        tahun_match = _re.search(r'(20\d{2})', file.filename)
        tahun_str   = tahun_match.group(1) if tahun_match else '0000'
        # Simpan dengan format nama yang sesuai glob: "fsva kabupaten YYYY.csv"
        save_name = f'fsva kabupaten {tahun_str}.csv'
        save_path = os.path.join(DIR_FSVA, save_name)
        with open(save_path, 'w', encoding='utf-8') as f_out:
            f_out.write(content)
        sb = get_supabase()
        if sb:
            try:
                supabase_storage_upload(
                    f'fsva/{save_name}', content.encode(),
                    'text/csv'
                )
            except Exception as _e_sb:
                print(f"[supabase] Gagal upload/sync: {_e_sb}")
        # Invalidate cache nowcast & EWS + fsva cache
        for f_name in ['fsva_nowcast_cache.json', 'ews_cache.json']:
            if os.path.exists(f_name):
                os.remove(f_name)
        global _ews_cache, _fsva_cache
        _ews_cache   = {}
        _fsva_cache  = {}
        threading.Thread(target=precompute_all, daemon=True).start()
        return jsonify({'status': 'success', 'message': f'{len(df_fsva)} kabupaten FSVA {tahun_str} tersimpan, cache sedang diperbarui'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Upload curah hujan ──
@app.route('/admin/upload/curah-hujan', methods=['POST'])
@admin_required
def admin_upload_hujan():
    global df
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Tidak ada file'}), 400
        file    = request.files['file']
        content = file.read().decode('utf-8')
        import io
        df_hujan = pd.read_csv(io.StringIO(content))
        missing  = [c for c in ['Nama Provinsi','Tahun','Bulan','Curah Hujan'] if c not in df_hujan.columns]
        if missing:
            return jsonify({'status': 'error', 'message': f'Kolom tidak lengkap: {missing}'}), 400

        # Simpan file asli ke folder uploads/hujan (berdiri sendiri, tidak
        # digabung/ditulis ulang -- sama seperti pola FSVA)
        upload_dir  = DIR_UPLOADS['hujan']
        save_name   = file.filename
        backup_path = os.path.join(upload_dir, save_name)
        df_hujan.to_csv(backup_path, index=False)

        # Simpan metadata
        meta = load_upload_metadata()
        if 'hujan' not in meta:
            meta['hujan'] = []
        meta['hujan'].append({
            'filename':   save_name,
            'rows':       len(df_hujan),
            'uploaded_at': _time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        save_upload_metadata(meta)

        sb = get_supabase()
        if sb:
            try:
                supabase_storage_upload(
                    f'uploads/hujan/{save_name}', content.encode(),
                    'text/csv'
                )
                with open(UPLOAD_METADATA_FILE, 'rb') as f_meta:
                    supabase_storage_upload(
                        'meta/upload_metadata.json',
                        f_meta.read(),
                        'application/json'
                    )
            except Exception as _e_sb:
                print(f"[supabase] Gagal upload/sync: {_e_sb}")
        # Reload df + invalidate cache dashboard
        threading.Thread(target=reload_data_and_cache, daemon=True).start()
        return jsonify({'status': 'success', 'message': f'{len(df_hujan)} baris baru curah hujan ditambahkan, cache sedang diperbarui'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── List file upload per komoditas/jenis ──
@app.route('/admin/uploads/<jenis>', methods=['GET'])
@admin_required
def list_uploads(jenis):
    try:
        meta = load_upload_metadata()
        files = meta.get(jenis, [])
        return jsonify({'status': 'success', 'data': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Hapus file upload per komoditas/jenis ──
@app.route('/admin/uploads/<jenis>/<filename>', methods=['DELETE'])
@admin_required
def delete_upload(jenis, filename):
    try:
        upload_dir = DIR_UPLOADS.get(jenis)
        if not upload_dir:
            return jsonify({'status': 'error', 'message': 'Jenis tidak valid'}), 400

        # Hapus file fisik lokal
        file_path = os.path.join(upload_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Update metadata
        meta = load_upload_metadata()
        if jenis in meta:
            meta[jenis] = [f for f in meta[jenis] if f['filename'] != filename]
        save_upload_metadata(meta)

        # Hapus juga dari Supabase (KALAU TIDAK, file akan kembali lagi
        # saat server restart karena restore_uploads_from_supabase())
        sb = get_supabase()
        if sb:
            try:
                supabase_storage_remove([f'uploads/{jenis}/{filename}'])
                with open(UPLOAD_METADATA_FILE, 'rb') as f_meta:
                    supabase_storage_upload(
                        'meta/upload_metadata.json',
                        f_meta.read(),
                        'application/json'
                    )
            except Exception as _e_sb:
                print(f"[supabase] Gagal upload/sync: {_e_sb}")

        # Reload cache (load_and_preprocess_data() otomatis gabungin ulang
        # base+upload yang tersisa di memori -- gak perlu rebuild file apapun)
        threading.Thread(target=reload_data_and_cache, daemon=True).start()

        return jsonify({'status': 'success', 'message': f'{filename} berhasil dihapus, cache sedang diperbarui'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Kalau ada file upload yang berhasil dipulihkan dari Supabase tadi,
#    reload df + cache SEBELUM server melayani request. Tidak perlu rebuild
#    apapun -- load_and_preprocess_data() otomatis menggabungkan file dasar +
#    semua file di folder uploads/ setiap kali dipanggil. ──
if _uploads_restored:
    try:
        print("[restore] Reload data setelah restore Supabase...")
        reload_data_and_cache()
    except Exception as e:
        print(f"[restore] Gagal reload data: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# ── Entry point untuk PythonAnywhere (WSGI) ──
# PythonAnywhere mengimpor variabel bernama 'application' dari file ini,
# BUKAN menjalankan app.run(). Baris di atas (__main__) tidak akan
# terpanggil saat dijalankan via WSGI, jadi ini wajib ada.
application = app