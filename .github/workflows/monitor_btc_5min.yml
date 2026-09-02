#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de Mercado BTC/EUR — versión GitHub Actions (email, sin depender del Mac)
====================================================================================

QUÉ ES ESTO
------------
Versión de monitor_estrategia_btc.py adaptada para correr en GitHub Actions:
en vez de un bucle infinito que revisa cada 15 minutos y avisa con una
notificación de macOS, este script hace UNA sola comprobación cada vez que
se ejecuta (el "bucle" lo gestiona el propio GitHub Actions, con un cron
que lo lanza cada 15 minutos), y en vez de notificación de macOS, envía un
EMAIL cuando encuentra una señal.

Funciona aunque tu ordenador esté apagado, porque corre en los servidores
de GitHub, no en tu Mac.

MISMA ESTRATEGIA VALIDADA que la versión de escritorio:
  1. Acumulación: rango ≤0,60% en ventana de 2h.
  2. Filtro de tendencia: precio > EMA50 y RSI(14) > 50.
  3. Filtro de funding rate: NO en el 10% más alto de sus últimas ~66
     liquidaciones (Binance, con Bybit como respaldo si Binance falla).
  4. Máximo 1 aviso al día (estado guardado en monitor_estado.json, que
     este script actualiza y el workflow de GitHub Actions se encarga de
     confirmar en el repositorio para que persista entre ejecuciones).
  5. Orden: entrada 0,17% sobre el techo de zona, SL en el suelo real de
     la zona, TP a 1x el riesgo (ratio 1:1, validado como el mejor de los
     probados: 53,8%/57,4% de acierto según periodo, profit factor 1,18-1,36).

RESULTADOS: en bruto, SIN comisiones reales aplicadas — pendiente de
confirmar cómo se cobra la comisión de una orden de ruptura en tu bróker.

VARIABLES DE ENTORNO NECESARIAS (configúralas como "Secrets" del repositorio
de GitHub, en Settings → Secrets and variables → Actions):
    GMAIL_USER            tu dirección de Gmail remitente
    GMAIL_APP_PASSWORD    contraseña de aplicación de Gmail (no tu contraseña normal)
    EMAIL_TO              a qué dirección quieres que lleguen los avisos
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests
except ImportError:
    requests = None

try:
    import yagmail
except ImportError:
    yagmail = None


# --------------------------------------------------------------------------
# CONFIGURACIÓN (idéntica a la versión de escritorio)
# --------------------------------------------------------------------------
TICKER = "BTC-EUR"
INTERVAL = "15m"          # marco de la ESTRATEGIA (no se toca)
FETCH_INTERVAL = "5m"     # marco de DESCARGA (para comprobar cada 5 min)
PERIOD = "60d"

WINDOW = 8
RANGE_THRESHOLD = 0.006
ENTRY_BUFFER = 0.001718
RR = 1.0

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_URL_BYBIT = "https://api.bybit.com/v5/market/funding/history"
FUNDING_SYMBOL = "BTCUSDT"
FUNDING_LOOKBACK = 200
FUNDING_EXTREME_PCTL = 0.90
STALE_THRESHOLD = 0.0015

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "monitor_estado_5min.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "monitor_historial_5min.log")

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
EMAIL_TO = os.environ.get("EMAIL_TO")


# --------------------------------------------------------------------------
# UTILIDADES
# --------------------------------------------------------------------------
def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def send_email(subject: str, body: str):
    if yagmail is None:
        log("[AVISO] Falta yagmail; no se puede enviar el email. Instala con: pip install yagmail")
        return
    if not (GMAIL_USER and GMAIL_APP_PASSWORD and EMAIL_TO):
        log("[AVISO] Faltan variables de entorno GMAIL_USER / GMAIL_APP_PASSWORD / EMAIL_TO. No se envía email.")
        return
    try:
        yag = yagmail.SMTP(GMAIL_USER, GMAIL_APP_PASSWORD)
        yag.send(to=EMAIL_TO, subject=subject, contents=body)
        log(f"Email enviado a {EMAIL_TO}: {subject}")
    except Exception as e:
        log(f"[ERROR] No se pudo enviar el email: {e}")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_alert_date": None}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# --------------------------------------------------------------------------
# DATOS E INDICADORES (idéntico a la versión de escritorio)
# --------------------------------------------------------------------------
def fetch_price_data() -> pd.DataFrame:
    """Descarga velas de 5m y las agrega a 15m alineadas al reloj. La última vela
    de 15m puede estar en formación; se marca en df.attrs['last_candle_closed']."""
    if yf is None:
        raise ImportError("Falta yfinance. Instálalo con: pip install yfinance")
    df5 = yf.download(TICKER, interval=FETCH_INTERVAL, period=PERIOD, progress=False)
    if isinstance(df5.columns, pd.MultiIndex):
        df5.columns = df5.columns.get_level_values(0)
    df5 = df5.rename(columns={"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"})
    if df5.index.tz is None:
        df5.index = df5.index.tz_localize("UTC")
    df5 = df5[["open", "high", "low", "close", "volume"]].dropna()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df15 = df5.resample("15min", label="left", closed="left").agg(agg).dropna()

    last_bucket_start = df15.index[-1]
    n_5m_in_last = int(((df5.index >= last_bucket_start) &
                        (df5.index < last_bucket_start + pd.Timedelta("15min"))).sum())
    df15.attrs["last_candle_closed"] = (n_5m_in_last >= 3)
    df15.attrs["n_5m_in_last"] = n_5m_in_last
    return df15


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out[avg_loss == 0] = 100
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = _ema(df["close"], 50)
    df["rsi14"] = _rsi(df["close"], 14)
    return df


def fetch_funding_percentile() -> dict | None:
    if requests is None:
        return None
    try:
        resp = requests.get(FUNDING_URL, params={"symbol": FUNDING_SYMBOL, "limit": FUNDING_LOOKBACK}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            rates = np.array([float(d["fundingRate"]) for d in data])
            current = rates[-1]
            pct = float((rates < current).mean())
            return dict(current=current, percentile=pct, is_extreme=pct >= FUNDING_EXTREME_PCTL, source="Binance")
    except Exception as e:
        log(f"Aviso: Binance funding rate no disponible ({e}). Probando Bybit...")

    try:
        resp = requests.get(FUNDING_URL_BYBIT, params={"category": "linear", "symbol": FUNDING_SYMBOL,
                                                         "limit": FUNDING_LOOKBACK}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("result", {}).get("list", [])
        if data:
            rates = np.array([float(d["fundingRate"]) for d in reversed(data)])
            current = rates[-1]
            pct = float((rates < current).mean())
            return dict(current=current, percentile=pct, is_extreme=pct >= FUNDING_EXTREME_PCTL, source="Bybit")
    except Exception as e:
        log(f"Aviso: Bybit funding rate tampoco disponible ({e}). Se omite ese filtro esta vez.")
    return None


# --------------------------------------------------------------------------
# DETECCIÓN DE LA SEÑAL (idéntico a la versión de escritorio, incluida la
# corrección de "ruptura ya avanzada")
# --------------------------------------------------------------------------
def detect_signal(df: pd.DataFrame) -> dict:
    if len(df) < WINDOW + 55:
        return dict(status="datos_insuficientes", detail=f"Solo {len(df)} velas disponibles.")

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    ema50 = df["ema50"].values
    rsi14 = df["rsi14"].values
    n = len(df)
    i = n - 1

    window_high = high[i - WINDOW + 1: i + 1].max()
    window_low = low[i - WINDOW + 1: i + 1].min()
    current_range = (window_high - window_low) / window_low

    if current_range <= RANGE_THRESHOLD:
        return dict(status="en_acumulacion", zone_high=round(float(window_high), 2),
                    zone_low=round(float(window_low), 2), price=round(float(close[i]), 2),
                    range_pct=round(current_range * 100, 2))

    lookback_high = high[i - WINDOW - 4: i - 3]
    lookback_low = low[i - WINDOW - 4: i - 3]
    if len(lookback_high) < WINDOW:
        return dict(status="sin_patron")

    zone_high = lookback_high.max()
    zone_low = lookback_low.min()
    zone_range = (zone_high - zone_low) / zone_low

    if not (zone_range <= RANGE_THRESHOLD and close[i] > zone_high and close[i] > zone_low):
        return dict(status="sin_patron")

    price_above_ema50 = close[i] > ema50[i]
    rsi_above_50 = rsi14[i] > 50
    if not (price_above_ema50 and rsi_above_50):
        return dict(status="sin_patron", detail="Ruptura detectada pero no cumple EMA50+RSI.")

    funding = fetch_funding_percentile()
    if funding is not None and funding["is_extreme"]:
        return dict(status="filtrado_por_funding", zone_high=round(float(zone_high), 2),
                    zone_low=round(float(zone_low), 2), funding=funding)

    entry = round(zone_high * (1 + ENTRY_BUFFER), 2)
    sl = round(float(zone_low), 2)
    risk = entry - sl
    tp = round(entry + RR * risk, 2)

    current_price = float(close[i])
    if current_price > entry * (1 + STALE_THRESHOLD):
        distancia_pct = (current_price - entry) / entry * 100
        return dict(
            status="ruptura_avanzada", zone_high=round(float(zone_high), 2), zone_low=sl,
            price=round(current_price, 2), entry_teorica=entry, tp_teorico=tp,
            distancia_pct=round(distancia_pct, 2), timestamp=df.index[i],
        )

    return dict(
        status="señal", zone_high=round(float(zone_high), 2), zone_low=sl,
        price=round(current_price, 2), entry=entry, tp=tp, sl=sl,
        rsi=round(float(rsi14[i]), 1), funding=funding, timestamp=df.index[i],
        provisional=not df.attrs.get("last_candle_closed", True),
    )


# --------------------------------------------------------------------------
# EJECUCIÓN (una sola pasada, no bucle — el cron de GitHub Actions es el bucle)
# --------------------------------------------------------------------------
def format_email_body(res: dict) -> str:
    lines = [
        f"Zona de acumulación: {res['zone_low']} - {res['zone_high']}",
        f"Precio de ENTRADA sugerido: {res['entry']}",
        f"TAKE PROFIT (TP): {res['tp']}  ({(res['tp']/res['entry']-1)*100:+.2f}% desde entrada)",
        f"STOP LOSS (SL): {res['sl']}  ({(res['sl']/res['entry']-1)*100:+.2f}% desde entrada)",
        f"RSI(14): {res['rsi']}",
    ]
    if res.get("funding"):
        lines.append(f"Funding rate: percentil {res['funding']['percentile']*100:.0f}% ({res['funding']['source']})")
    lines.append("")
    lines.append("Recuerda: esto es un aviso, no una orden automática. Revisa antes de operar.")
    lines.append("Resultados validados en bruto (sin comisiones reales aplicadas).")
    return "\n".join(lines)


def main():
    if yf is None:
        log("ERROR: falta yfinance en este entorno.")
        sys.exit(1)

    state = load_state()
    today = datetime.now().strftime("%Y-%m-%d")

    df = fetch_price_data()
    df = add_indicators(df)
    res = detect_signal(df)

    if res["status"] == "señal":
        provisional = res.get("provisional", False)
        signal_key = f"{today}_{res['zone_low']}_{res['zone_high']}"

        if provisional:
            # Email PROVISIONAL: la vela de 15m aún no ha cerrado. Solo uno por señal.
            if state.get("last_provisional_key") == signal_key:
                log("Señal provisional ya enviada por email para esta vela en formación. Espera al cierre.")
                return
            if state.get("last_alert_date") == today:
                log("Ya hubo aviso confirmado hoy; no se envía provisional adicional.")
                return
            log("⚠️ SEÑAL PROVISIONAL (vela de 15m en formación)")
            body = ("AVISO PROVISIONAL — la vela de 15m todavía se está formando y la señal "
                    "puede deshacerse antes del cierre. Úsalo para prepararte, no para entrar "
                    "en firme todavía. Si al cerrar la vela se mantiene, recibirás un segundo "
                    "email de CONFIRMACIÓN.\n\n" + format_email_body(res))
            log(body)
            send_email(f"[BTC/EUR] Señal PROVISIONAL — TP {res['tp']} / SL {res['sl']}", body)
            state["last_provisional_key"] = signal_key
            save_state(state)
        else:
            # Email CONFIRMADO: la vela de 15m ha cerrado cumpliendo las condiciones.
            if state.get("last_alert_date") == today:
                log("Señal confirmada, pero ya se avisó hoy (máx. 1 confirmado/día). Se omite.")
                return
            log("🔔 SEÑAL DE ENTRADA CONFIRMADA")
            body = "SEÑAL CONFIRMADA (la vela de 15m ha cerrado cumpliendo las condiciones).\n\n" + format_email_body(res)
            log(body)
            send_email(f"[BTC/EUR] Señal CONFIRMADA — TP {res['tp']} / SL {res['sl']}", body)
            state["last_alert_date"] = today
            save_state(state)

    elif res["status"] == "en_acumulacion":
        log(f"En acumulación ({res['zone_low']}-{res['zone_high']}, rango {res['range_pct']}%). Sin ruptura aún.")

    elif res["status"] == "ruptura_avanzada":
        log(f"Ruptura detectada pero el precio ({res['price']}) ya está {res['distancia_pct']:.2f}% por "
            f"encima de la entrada teórica ({res['entry_teorica']}). Se descarta, no se avisa.")

    elif res["status"] == "filtrado_por_funding":
        log(f"Ruptura detectada pero descartada: funding rate en zona extrema "
            f"(percentil {res['funding']['percentile']*100:.0f}%).")

    elif res["status"] == "datos_insuficientes":
        log(f"Datos insuficientes: {res.get('detail', '')}")

    else:
        log("Sin patrón relevante en este momento.")


if __name__ == "__main__":
    main()
