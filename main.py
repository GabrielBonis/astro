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
eph = load('de421.bsp')
ts = load.timescale()

with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

bright_stars = stars[stars['magnitude'] <= 5.5].copy()

CONST_URLS = [
    "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/western/constellationship.fab",
    "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern_st/constellationship.fab",
    "https://gist.githubusercontent.com/gacarrillor/e7fc425b5443617afd294b7ba6169864/raw/"
]
constellation_pairs_radec = []

for url in CONST_URLS:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        for linha in response.text.splitlines():
            if linha.startswith("#") or not linha.strip(): continue
            partes = linha.split()
            if len(partes) < 3: continue
            num_linhas = int(partes[1])
            for i in range(num_linhas):
                h1, h2 = int(partes[2 + i*2]), int(partes[2 + i*2 + 1])
                if h1 in stars.index and h2 in stars.index:
                    star1, star2 = stars.loc[h1], stars.loc[h2]
                    constellation_pairs_radec.append([
                        (star1['ra_hours'], star1['dec_degrees']),
                        (star2['ra_hours'], star2['dec_degrees'])
                    ])
        break
    except:
        pass

def obter_nome_local(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
        resp = requests.get(url, headers={'User-Agent': 'SkyMapAPI/1.0'}, timeout=5)
        if resp.status_code == 200:
            address = resp.json().get('address', {})
            cidade = address.get('city') or address.get('town') or address.get('village') or address.get('municipality', '')
            pais = address.get('country', '')
            if cidade and pais: return f"{cidade}, {pais}".upper()
            elif cidade: return cidade.upper()
    except:
        pass
    return "LOCALIZAÇÃO ESPECÍFICA"

class SkyMapRequest(BaseModel):
    lat: float
    lon: float
    date: str
    title: str = "O CÉU NAQUELE MOMENTO"

def project_to_altaz(ra_hours, dec_deg, lst, lat_rad):
    dec_rad = np.radians(dec_deg)
    ha_rad = np.radians((lst - ra_hours) * 15.0)
    sin_alt = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    alt_degrees = np.degrees(alt_rad)
    sin_az = -np.sin(ha_rad) * np.cos(dec_rad) / np.cos(alt_rad)
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_alt) / (np.cos(lat_rad) * np.cos(alt_rad))
    az_rad = np.arctan2(sin_az, cos_az)
    return (az_rad + 2 * np.pi) % (2 * np.pi), alt_degrees

@app.post("/sky-map")
async def generate_sky_map(req: SkyMapRequest):
    try:
        dt_obj = datetime.strptime(req.date, "%Y-%m-%d")
        t = ts.utc(dt_obj.year, dt_obj.month, dt_obj.day)
        lst = t.gast + (req.lon / 15.0)
        lat_rad = np.radians(req.lat)

        x_stars, alt_stars = project_to_altaz(
            bright_stars['ra_hours'].values, 
            bright_stars['dec_degrees'].values, 
            lst, lat_rad
        )
        visible_mask = alt_stars > 0
        y_stars = 90 - alt_stars[visible_mask]
        x_stars = x_stars[visible_mask]
        mag_stars = bright_stars['magnitude'].values[visible_mask]

        DPI = 300
        FIG_WIDTH, FIG_HEIGHT = 12, 18
        BG_COLOR, ACCENT_COLOR = '#050508', '#FFFFFF'

        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG_COLOR)
        fig.subplots_adjust(bottom=0.25, top=0.95, left=0.05, right=0.95)
        
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor(BG_COLOR)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 90)

        ax.grid(True, color=ACCENT_COLOR, alpha=0.1, linewidth=0.5, linestyle='-')
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.spines['polar'].set_color(ACCENT_COLOR)
        ax.spines['polar'].set_linewidth(1.5)

        for (ra1, dec1), (ra2, dec2) in constellation_pairs_radec:
            az1, alt1 = project_to_altaz(ra1, dec1, lst, lat_rad)
            az2, alt2 = project_to_altaz(ra2, dec2, lst, lat_rad)
            if alt1 > -2 or alt2 > -2: 
                x_coords, y_coords = [az1, az2], [90 - alt1, 90 - alt2]
                if abs(az1 - az2) > np.pi:
                    if az1 > az2: x_coords[1] += 2*np.pi
                    else: x_coords[0] += 2*np.pi
                ax.plot(x_coords, y_coords, color=ACCENT_COLOR, alpha=0.25, linewidth=0.7, zorder=2)

        base_size = np.clip(6 - mag_stars, 0.5, None)
        size = (base_size ** 1.3) * (DPI / 200) 

        glow = ax.scatter(x_stars, y_stars, s=size*2, color=ACCENT_COLOR, alpha=0.08, edgecolors='none', zorder=3, marker='o')
        glow.set_path_effects([path_effects.withSimplePatchShadow(offset=(0,0), shadow_rgbFace=ACCENT_COLOR, alpha=0.2, rho=DPI*0.01)])
        ax.scatter(x_stars, y_stars, s=size, color=ACCENT_COLOR, alpha=1.0, edgecolors='none', zorder=4, marker='o')
        
        ax.scatter(0, 0, s=size.mean()*2.5, color=BG_COLOR, edgecolors=ACCENT_COLOR, linewidth=1.2, zorder=5, marker='o')

        title_font = {'fontfamily': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'], 'fontweight': 'light'}
        body_font = {'fontfamily': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'], 'fontweight': 'normal'}
        coord_font = {'fontfamily': ['Consolas', 'Liberation Mono', 'Courier New', 'monospace'], 'fontweight': 'normal'}
        
        plt.text(0.5, 0.17, " ".join(req.title.upper()), color=ACCENT_COLOR, size=24 * (DPI/150), ha='center', transform=fig.transFigure, **title_font)
        plt.text(0.5, 0.12, " ".join(obter_nome_local(req.lat, req.lon)), color=ACCENT_COLOR, ha='center', transform=fig.transFigure, size=11 * (DPI/150), alpha=0.9, **body_font)
        plt.text(0.5, 0.09, " ".join(dt_obj.strftime('%d/%m/%Y')), color=ACCENT_COLOR, ha='center', transform=fig.transFigure, size=10 * (DPI/150), alpha=0.6, **body_font)
        
        lat_dir, lon_dir = ('S' if req.lat < 0 else 'N'), ('W' if req.lon < 0 else 'E')
        plt.text(0.5, 0.04, f"{abs(req.lat):.4f}° {lat_dir}   |   {abs(req.lon):.4f}° {lon_dir}", color=ACCENT_COLOR, ha='center', transform=fig.transFigure, size=9 * (DPI/150), alpha=0.35, **coord_font)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0.6, dpi=DPI)
        plt.close(fig)
        buf.seek(0)
        
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        import traceback
        return {"erro": str(e), "detalhes": traceback.format_exc()}