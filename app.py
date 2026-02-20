import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import requests
from fastapi import FastAPI, Response
from pydantic import BaseModel
from skyfield.api import Loader
from skyfield.data import hipparcos
from datetime import datetime

app = FastAPI()

load = Loader('/tmp')
eph = None
stars = None
bright_stars = None
constellation_pairs_radec = []

def startup_data():
    global eph, stars, bright_stars, constellation_pairs_radec
    if eph is None:
        eph = load('de421.bsp')
        with load.open(hipparcos.URL) as f:
            stars = hipparcos.load_dataframe(f)
        bright_stars = stars[stars['magnitude'] <= 5.8].copy()
        
        url = "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern_st/constellationship.fab"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                constellation_pairs_radec = []
                for linha in resp.text.splitlines():
                    partes = linha.split()
                    if len(partes) < 3: continue
                    num = int(partes[1])
                    for i in range(num):
                        h1, h2 = int(partes[2+i*2]), int(partes[2+i*2+1])
                        if h1 in stars.index and h2 in stars.index:
                            s1, s2 = stars.loc[h1], stars.loc[h2]
                            constellation_pairs_radec.append([(s1['ra_hours'], s1['dec_degrees']), (s2['ra_hours'], s2['dec_degrees'])])
        except: pass

def obter_nome_local(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
        headers = {'User-Agent': f'AstroFrame_{datetime.now().timestamp()}'}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            addr = resp.json().get('address', {})
            cidade = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('suburb') or addr.get('state')
            pais = addr.get('country', '')
            if cidade and pais: return f"{cidade}, {pais}".upper()
    except: pass
    return "COORDENADAS SELECIONADAS"

class SkyMapRequest(BaseModel):
    lat: float
    lon: float
    date: str
    title: str = "O NASCER DE UMA ESTRELA"

def project(ra_hours, dec_deg, lst, lat_rad):
    dec_rad = np.radians(dec_deg)
    ha_rad = np.radians((lst - ra_hours) * 15.0)
    sin_alt = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    sin_az = -np.sin(ha_rad) * np.cos(dec_rad) / np.cos(alt_rad)
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_alt) / (np.cos(lat_rad) * np.cos(alt_rad))
    return (np.arctan2(sin_az, cos_az) + 2 * np.pi) % (2 * np.pi), np.degrees(alt_rad)

@app.post("/sky-map")
async def generate_sky_map(req: SkyMapRequest):
    startup_data()
    ts = load.timescale()
    try:
        dt = datetime.strptime(req.date, "%Y-%m-%d")
        t = ts.utc(dt.year, dt.month, dt.day)
        lst = t.gast + (req.lon / 15.0)
        lat_rad = np.radians(req.lat)

        az, alt = project(bright_stars['ra_hours'].values, bright_stars['dec_degrees'].values, lst, lat_rad)
        mask = alt > 0
        x, y, mag = az[mask], 90 - alt[mask], bright_stars['magnitude'].values[mask]

        DPI = 300
        fig = plt.figure(figsize=(12, 18), facecolor='#050508')
        fig.subplots_adjust(bottom=0.25, top=0.92)
        
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor('#050508')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 90)
        ax.grid(True, color='white', alpha=0.08, linewidth=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['polar'].set_color('white')
        ax.spines['polar'].set_linewidth(1.2)

        for (r1, d1), (r2, d2) in constellation_pairs_radec:
            az1, alt1 = project(r1, d1, lst, lat_rad)
            az2, alt2 = project(r2, d2, lst, lat_rad)
            if alt1 > 0 and alt2 > 0:
                xc, yc = [az1, az2], [90-alt1, 90-alt2]
                if abs(az1-az2) > np.pi:
                    if az1 > az2: xc[1] += 2*np.pi
                    else: xc[0] += 2*np.pi
                ax.plot(xc, yc, color='white', alpha=0.2, linewidth=0.6, zorder=2)

        sizes = (1.2 + (6 - mag) * 0.3) * (DPI / 250)
        ax.scatter(x, y, s=sizes*3, color='white', alpha=0.1, edgecolors='none', zorder=3)
        ax.scatter(x, y, s=sizes, color='white', alpha=1.0, edgecolors='none', zorder=4)
        ax.scatter(0, 0, s=25, color='#050508', edgecolors='white', linewidth=1.2, zorder=5)

        t_font = {'fontfamily': 'sans-serif', 'fontweight': 'light'}
        b_font = {'fontfamily': 'sans-serif', 'fontweight': 'normal'}
        c_font = {'fontfamily': 'monospace', 'fontweight': 'light'}

        plt.text(0.5, 0.18, " ".join(req.title.upper()), color='white', size=24, ha='center', transform=fig.transFigure, **t_font)
        
        local_name = obter_nome_local(req.lat, req.lon)
        plt.text(0.5, 0.13, " ".join(local_name), color='white', ha='center', transform=fig.transFigure, size=11, alpha=0.9, **b_font)
        
        data_ext = dt.strftime('%d de %B de %Y').upper()
        plt.text(0.5, 0.10, " ".join(data_ext), color='white', ha='center', transform=fig.transFigure, size=10, alpha=0.7, **b_font)
        
        lat_dir, lon_dir = ('S' if req.lat < 0 else 'N'), ('W' if req.lon < 0 else 'E')
        coords = f"{abs(req.lat):.4f}° {lat_dir}   |   {abs(req.lon):.4f}° {lon_dir}"
        plt.text(0.5, 0.06, coords, color='white', ha='center', transform=fig.transFigure, size=9, alpha=0.35, **c_font)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#050508', bbox_inches='tight', pad_inches=0.6, dpi=DPI)
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        return {"erro": str(e)}