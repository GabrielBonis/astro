import io
import os
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import requests
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from skyfield.api import Loader
from skyfield.data import hipparcos
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
S3_BUCKET = "astro-gbonis"
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=S3_REGION
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

load = Loader('/tmp')
eph = None
stars = None
bright_stars = None
constellation_lines = []

# ---------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------
def format_coords(lat, lon):
    lat_dir = 'S' if lat < 0 else 'N'
    lon_dir = 'W' if lon < 0 else 'E'
    return f"{abs(lat):.4f}° {lat_dir}   |   {abs(lon):.4f}° {lon_dir}"

def format_title(title):
    spaced = ""
    for char in title.upper():
        if char == " ":
            spaced += "   "
        else:
            spaced += char + " "
    return spaced.strip()

def ensure_data_exists(filename, remote_url=None, is_skyfield_native=False):
    if not os.path.exists('/tmp'): os.makedirs('/tmp', exist_ok=True)
    local_path = os.path.join('/tmp', filename)
    s3_key = f'data/{filename}'

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    try:
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        return local_path
    except Exception as e:
        print(f"Aviso: Não achou {filename} no S3. Tentando fallback... Erro: {e}")

    try:
        if is_skyfield_native:
            load.download(filename)
            return local_path
            
        elif remote_url:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(remote_url, headers=headers, timeout=15)
            if resp.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(resp.content)
                return local_path
    except:
        pass
    
    return None

def startup_data():
    global eph, stars, bright_stars, constellation_lines
    if eph is not None: return

    ensure_data_exists('de421.bsp', is_skyfield_native=True)
    eph = load('de421.bsp')

    hip_path = ensure_data_exists('hip_main.dat', is_skyfield_native=True)
    if hip_path:
        import pandas as pd # Garante que o pandas está disponível para o hipparcos
        with load.open('hip_main.dat') as f:
            stars = hipparcos.load_dataframe(f)
        stars = stars.sort_index()
        bright_stars = stars[stars['magnitude'] <= 6.5].copy()

    fab_url = "https://raw.githubusercontent.com/Stellarium/stellarium/eb47095a9282cf6b981f6e37fe1ea3a3ae0fd167/skycultures/modern_st/constellationship.fab"
    fab_path = ensure_data_exists('constellationship.fab', fab_url)

    constellation_lines = []
    if fab_path and os.path.exists(fab_path) and stars is not None:
        with open(fab_path, 'r', encoding='utf-8') as f:
            for linha in f:
                partes = linha.strip().split()
                if len(partes) < 4: continue
                try:
                    hips = [int(h) for h in partes[2:]]
                    for i in range(len(hips) - 1):
                        h1, h2 = hips[i], hips[i+1]
                        if h1 in stars.index and h2 in stars.index:
                            s1, s2 = stars.loc[h1], stars.loc[h2]
                            constellation_lines.append([(s1['ra_hours'], s1['dec_degrees']), (s2['ra_hours'], s2['dec_degrees'])])
                except: continue

# ---------------------------------------------------------------------
# MATH & GEOLOCATION (COM PROJEÇÃO OVAL)
# ---------------------------------------------------------------------
def project_oval(ra_hours, dec_deg, lst, lat_rad, flatten_factor=1.1):
    dec_rad = np.radians(dec_deg)
    ha_rad = np.radians((lst - ra_hours) * 15.0)
    sin_alt = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    sin_az = -np.sin(ha_rad) * np.cos(dec_rad) / np.cos(alt_rad)
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_alt) / (np.cos(lat_rad) * np.cos(alt_rad))
    az = np.arctan2(sin_az, cos_az)

    r = (np.pi / 2.0 - alt_rad) / (np.pi / 2.0)
    
    x = r * np.sin(az)
    y = r * np.cos(az)

    x_oval = x * flatten_factor
    y_oval = y

    return x_oval, y_oval, alt_rad

def obter_nome_local(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
        resp = requests.get(url, headers={'User-Agent': 'Gbonis_Astro_v4'}, timeout=5)
        if resp.status_code == 200:
            addr = resp.json().get('address', {})
            city = addr.get('city') or addr.get('town') or addr.get('state') or "TERRA"
            return f"{city}, {addr.get('country', 'BRASIL')}".upper()
    except: pass
    return "SÃO PAULO, BRASIL"

class SkyMapRequest(BaseModel):
    lat: float
    lon: float
    date: str
    title: str = "O NASCER DE UMA ESTRELA"

@app.post("/sky-map")
async def generate_sky_map(req: SkyMapRequest):
    startup_data()
    ts = load.timescale()
    try:
        dt_base = datetime.strptime(req.date, "%Y-%m-%d")
        t = ts.utc(dt_base.year, dt_base.month, dt_base.day, 22 + 3, 0)
        lst = t.gast + (req.lon / 15.0)
        lat_rad = np.radians(req.lat)
        FLATTEN_FACTOR = 1.1

        fig = plt.figure(figsize=(10, 13), facecolor='black')
        ax = fig.add_axes([0.05, 0.35, 0.9, 0.6])
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_xlim(-FLATTEN_FACTOR - 0.05, FLATTEN_FACTOR + 0.05)
        ax.set_ylim(-1.05, 1.05)

        oval_border = Ellipse((0, 0), width=2*FLATTEN_FACTOR, height=2, 
                              edgecolor='white', facecolor='none', linewidth=1.2, alpha=0.9, zorder=3)
        ax.add_patch(oval_border)

        for line in constellation_lines:
            (r1, d1), (r2, d2) = line
            x1, y1, alt1_rad = project_oval(r1, d1, lst, lat_rad, FLATTEN_FACTOR)
            x2, y2, alt2_rad = project_oval(r2, d2, lst, lat_rad, FLATTEN_FACTOR)

            if alt1_rad > 0 and alt2_rad > 0:
                ax.plot([x1, x2], [y1, y2], color='white', alpha=0.4, lw=0.4, zorder=1)

        if bright_stars is not None:
            ra_vals = bright_stars['ra_hours'].values
            dec_vals = bright_stars['dec_degrees'].values
            
            x_s, y_s, alt_s_rad = project_oval(ra_vals, dec_vals, lst, lat_rad, FLATTEN_FACTOR)
            
            mask = alt_s_rad > 0
            mags = bright_stars['magnitude'].values[mask]
            
            s_sizes = np.clip((6.5 - mags) ** 1.9, 0.05, 12)
            ax.scatter(x_s[mask], y_s[mask], s=s_sizes, color='white', edgecolors='none', alpha=1.0, zorder=2)

        plt.text(0.5, 0.22, format_title(req.title), color='white', size=22, ha='center', transform=fig.transFigure, weight='light')
        
        city_str = obter_nome_local(req.lat, req.lon)
        date_str = dt_base.strftime('%d/%m/%Y')
        coords_str = format_coords(req.lat, req.lon)

        plt.text(0.5, 0.15, city_str, color='white', size=11, ha='center', alpha=0.8, transform=fig.transFigure, weight='light')
        plt.text(0.5, 0.11, date_str, color='white', size=10, ha='center', alpha=0.7, transform=fig.transFigure, weight='light')
        plt.text(0.5, 0.07, coords_str, color='white', size=9, ha='center', alpha=0.5, transform=fig.transFigure, weight='light')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        file_key = f"maps/{uuid.uuid4()}.png"
        s3_client.put_object(Bucket=S3_BUCKET, Key=file_key, Body=buf, ContentType='image/png')
        return {"url": f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_key}"}
        
    except Exception as e:
        return {"erro": str(e)}

@app.get("/geocode")
async def geocode(q: str):
    return requests.get(f"https://nominatim.openstreetmap.org/search?format=json&q={q}&limit=5",
                        headers={'User-Agent': 'Gbonis_Astro_v4'}).json()