# -*- coding: utf-8 -*-
"""
Crypto S/R Analyzer — Streamlit + Bybit (публичное API, без ключей).

Вход: тикер монеты (например BTCUSDT).
Выход: зоны поддержки/сопротивления + сценарии ЛОНГ/ШОРТ (вход, цели, стоп).
"""

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BYBIT_API = "https://api.bybit.com"

INTERVALS = {"15 мин": "15", "1 час": "60", "4 часа": "240", "1 день": "D"}
SWING_K = 3          # «плечо» фрактала (свечи слева/справа)
N_KLINES = 320
ZONES_PER_SIDE = 3

st.set_page_config(page_title="Crypto S/R Analyzer — Bybit", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
#  Данные: публичные эндпоинты Bybit v5 (регистрация и ключи не нужны)        #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=45, show_spinner=False)
def fetch_klines(symbol: str, category: str, interval: str, limit: int = N_KLINES) -> pd.DataFrame:
    resp = requests.get(
        f"{BYBIT_API}/v5/market/klines",
        params={"category": category, "symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0 or not data["result"]["list"]:
        raise ValueError(data.get("retMsg", "По этому тикеру нет данных"))
    df = pd.DataFrame(
        data["result"]["list"],
        columns=["start", "open", "high", "low", "close", "volume", "turnover"],
    )
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df["start"] = pd.to_datetime(df["start"].astype(np.int64), unit="ms")
    return df.sort_values("start").reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ticker(symbol: str, category: str) -> dict:
    resp = requests.get(
        f"{BYBIT_API}/v5/market/tickers",
        params={"category": category, "symbol": symbol},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0 or not data["result"]["list"]:
        raise ValueError(data.get("retMsg", "Тикер не найден"))
    return data["result"]["list"][0]


# --------------------------------------------------------------------------- #
#  Индикаторы и построение зон                                                 #
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn)
    return out.fillna(50).clip(0, 100)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def find_swings(df: pd.DataFrame, k: int = SWING_K):
    """Индексы фрактальных максимумов/минимумов."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    hi_idx, lo_idx = [], []
    for i in range(k, len(df) - k):
        if highs[i] >= highs[i - k: i + k + 1].max():
            hi_idx.append(i)
        if lows[i] <= lows[i - k: i + k + 1].min():
            lo_idx.append(i)
    return hi_idx, lo_idx


def cluster_zones(levels, tol: float):
    """Склеиваем близкие уровни в зоны (min, max, вес)."""
    groups = []
    for lv in sorted(levels, reverse=True):
        if groups and abs(lv - float(np.mean(groups[-1]))) <= tol:
            groups[-1].append(lv)
        else:
            groups.append([lv])
    return [(min(g), max(g), len(g)) for g in groups]


def pad_zones(zones, side, price, a, n=ZONES_PER_SIDE):
    """Если реальных зон меньше n — достраиваем проекцией через ATR."""
    out = list(zones)
    while len(out) < n:
        if side == "res":
            base = out[-1][0] if out else price + 1.2 * a
            out.append((base + 1.5 * a, base + 2.0 * a, 0))
        else:
            base = out[-1][1] if out else price - 1.2 * a
            out.append((base - 2.0 * a, base - 1.5 * a, 0))
    return out[:n]


def analyze(df: pd.DataFrame) -> dict:
    close = df["close"]
    price = float(close.iloc[-1])
    a = float(atr(df).iloc[-1])

    hi_idx, lo_idx = find_swings(df)
    levels = [float(df["high"].iat[i]) for i in hi_idx]
    levels += [float(df["low"].iat[i]) for i in lo_idx]
    zones = cluster_zones(levels, tol=0.6 * a)

    eps = 0.15 * a
    above = sorted([z for z in zones if z[0] > price + eps], key=lambda z: z[0])
    below = sorted([z for z in zones if z[1] < price - eps], key=lambda z: -z[1])
    inside = [z for z in zones if z[0] - eps <= price <= z[1] + eps]

    R = pad_zones(above, "res", price, a)
    S = pad_zones(below, "sup", price, a)
    R1, R2, R3 = R
    S1, S2, S3 = S

    # эвристика «вероятности» по тренду и импульсу (НЕ статистика!)
    e20 = float(ema(close, 20).iloc[-1])
    e50 = float(ema(close, 50).iloc[-1])
    r = float(rsi(close).iloc[-1])
    mom = (price / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0.0
    score = (10 if price > e20 else -10) + (8 if e20 > e50 else -8)
    score += 4 if r > 55 else (-4 if r < 45 else 0)
    score += float(np.clip(mom * 2, -8, 8))
    long_prob = int(np.clip(round(50 + score), 30, 70))
    short_prob = 100 - long_prob

    # ЛОНГ: пробой ближайшего сопротивления и закрепление
    long_entry = (R1[1], R1[1] + 0.30 * a)
    long_tps = list(np.maximum.accumulate([R2[1] + 0.05 * a, R3[1], R3[1] + 1.5 * a]))
    long_stop = S1[0] - 0.05 * a

    # ШОРТ: отскок/реакция от ближайшего сопротивления
    short_entry = (R1[0], R1[1] + 0.10 * a)
    short_tps = list(np.minimum.accumulate([S1[1], S2[1], S3[1]]))
    short_stop = R2[1] + 0.05 * a

    return dict(
        price=price, atr=a, rsi=r, ema20=e20, ema50=e50,
        zones_R=R, zones_S=S, inside=inside,
        long_prob=long_prob, short_prob=short_prob,
        long_entry=long_entry, long_tps=long_tps, long_stop=long_stop,
        short_entry=short_entry, short_tps=short_tps, short_stop=short_stop,
    )


# --------------------------------------------------------------------------- #
#  Оформление                                                                  #
# --------------------------------------------------------------------------- #
def fmt(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    if abs(x) >= 1:
        return f"{x:,.2f}".rstrip("0").rstrip(".")
    return f"{x:.6f}".rstrip("0").rstrip(".")


def pct(x: float, base: float) -> str:
    return f"{(x / base - 1) * 100:+.1f}%"


def card_html(title, subtitle, prob, color, bg, lines):
    rows = "".join(
        f"<tr><td style='padding:2px 14px 2px 0;color:#888;white-space:nowrap'>{k}</td>"
        f"<td style='font-family:ui-monospace,monospace'>{v}</td></tr>"
        for k, v in lines
    )
    return f"""
    <div style="border:1px solid {color};border-radius:10px;background:{bg};
                padding:14px 16px;margin-top:8px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b style="color:{color};font-size:17px">{title}</b>
        <span style="background:{color};color:#fff;border-radius:6px;
                     padding:2px 10px;font-size:13px">вероятность {prob}%</span>
      </div>
      <div style="color:#888;font-size:12px;margin:2px 0 8px">{subtitle}</div>
      <table style="font-size:14px;border-spacing:0">{rows}</table>
    </div>"""


# --------------------------------------------------------------------------- #
#  График                                                                      #
# --------------------------------------------------------------------------- #
def build_chart(df: pd.DataFrame, info: dict, symbol: str) -> go.Figure:
    fig = go.Figure(
        go.Candlestick(
            x=df["start"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=symbol,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
    )
    t0, t1 = df["start"].iloc[0], df["start"].iloc[-1]
    pad = (t1 - t0) * 0.17
    xr = [t0, t1 + pad]
    price = info["price"]

    def add_zone(lo, hi, fill, name, descr, pct_val):
        fig.add_shape(type="rect", x0=xr[0], x1=xr[1], y0=lo, y1=hi,
                      fillcolor=fill, line=dict(width=0), layer="below")
        text = f"<b>{name}</b>&nbsp;&nbsp;{fmt(lo)} – {fmt(hi)}&nbsp;&nbsp;{pct_val:+.1f}%"
        if descr:
            text += f"<br><span style='font-size:10px;color:#888'>{descr}</span>"
        fig.add_annotation(x=xr[1], xref="x", xanchor="right",
                           y=(lo + hi) / 2, yref="y",
                           text=text, showarrow=False, align="right",
                           font=dict(size=11, color="#333"),
                           bgcolor="rgba(255,255,255,0.65)")

    res_names = ["ЛОКАЛЬНОЕ СОПРОТИВЛЕНИЕ", "СОПРОТИВЛЕНИЕ", "ГЛОБАЛЬНОЕ СОПРОТИВЛЕНИЕ"]
    res_descr = ["первая зона для шорта", "возможен отскок / ложный пробой",
                 "основная ликвидность, фиксация прибыли"]
    sup_names = ["СЛЕДУЮЩАЯ ПОДДЕРЖКА", "ПОДДЕРЖКА", "КЛЮЧЕВАЯ ПОДДЕРЖКА"]
    sup_descr = ["где возможен отскок", "", "глобальный уровень"]

    for i, (lo, hi, _) in enumerate(info["zones_R"]):
        j = min(i, 2)
        add_zone(lo, hi, "rgba(239,83,80,0.10)", res_names[j], res_descr[j],
                 (hi - price) / price * 100)
    for i, (lo, hi, _) in enumerate(info["zones_S"]):
        j = min(i, 2)
        add_zone(lo, hi, "rgba(38,166,154,0.10)", sup_names[j], sup_descr[j],
                 (lo - price) / price * 100)

    if info["inside"]:
        lo, hi, _ = info["inside"][0]
        for y in (lo, hi):
            fig.add_shape(type="line", x0=xr[0], x1=xr[1], y0=y, y1=y,
                          line=dict(color="#1e88e5", dash="dash", width=1.2),
                          layer="below")
        fig.add_annotation(x=xr[0], xref="x", xanchor="left", y=hi, yref="y",
                           yanchor="bottom", showarrow=False,
                           text=f"<b>ЗОНА ПОДДЕРЖКИ / РЕАКЦИИ</b>&nbsp;&nbsp;{fmt(lo)} – {fmt(hi)}",
                           font=dict(size=11, color="#1e88e5"),
                           bgcolor="rgba(255,255,255,0.65)")

    fig.add_shape(type="line", x0=xr[0], x1=xr[1], y0=price, y1=price,
                  line=dict(color="#111", dash="dot", width=1))
    fig.add_annotation(x=xr[1], xref="x", xanchor="right", y=price, yref="y",
                       yanchor="bottom", showarrow=False,
                       text=f"<b>{fmt(price)}</b>",
                       font=dict(color="#fff", size=11),
                       bgcolor="#111", borderpad=3)

    fig.update_layout(
        height=640, margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, hovermode="x unified",
        xaxis=dict(range=xr, showgrid=False),
        yaxis=dict(side="right", gridcolor="#eeeeee", tickformat=",.6~f"),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# --------------------------------------------------------------------------- #
#  UI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    st.title("📊 Crypto S/R Analyzer — Bybit")

    with st.sidebar:
        st.header("Параметры")
        raw = st.text_input("Монета", value="BTCUSDT",
                            help="Например: BTC, ETH, SOL или сразу BTCUSDT").strip().upper()
        if not raw.endswith(("USDT", "USDC", "USD", "EUR")):
            raw += "USDT"
        symbol = raw
        category = st.selectbox(
            "Рынок", ["spot", "linear"],
            format_func=lambda v: {"spot": "Спот", "linear": "Фьючерсы USDT-M"}[v],
        )
        interval = st.selectbox("Таймфрейм", list(INTERVALS), index=1)
        st.caption("Котировки — публичное API Bybit, ключи не нужны.")

    try:
        df = fetch_klines(symbol, category, INTERVALS[interval])
        ticker = fetch_ticker(symbol, category)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось загрузить данные по {symbol}: {exc}")
        st.stop()

    info = analyze(df)
    price = info["price"]
    chg = float(ticker.get("price24hPcnt", 0)) * 100
    color = "#26a69a" if chg >= 0 else "#ef5350"

    c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
    c1.markdown(
        f"<span style='font-size:34px;font-weight:700;color:{color}'>{fmt(price)}</span>"
        f"<span style='font-size:14px;color:#888'>&nbsp;&nbsp;{symbol} · {interval} · "
        f"за сутки {chg:+.2f}%</span>",
        unsafe_allow_html=True,
    )
    c2.metric("Макс. 24ч", fmt(float(ticker["highPrice24h"])))
    c3.metric("Мин. 24ч", fmt(float(ticker["lowPrice24h"])))
    c4.metric("ATR 14", fmt(info["atr"]))
    st.caption(
        f"RSI(14): {info['rsi']:.0f} · EMA20 {fmt(info['ema20'])} "
        f"({'выше' if info['ema20'] > info['ema50'] else 'ниже'} EMA50 {fmt(info['ema50'])})"
    )

    st.plotly_chart(build_chart(df, info, symbol), use_container_width=True)

    le_lo, le_hi = info["long_entry"]
    se_lo, se_hi = info["short_entry"]
    e_long = (le_lo + le_hi) / 2
    e_short = (se_lo + se_hi) / 2

    long_lines = [
        ("Вход:", f"{fmt(le_lo)} – {fmt(le_hi)} (после закрепления)"),
        ("Цели:", "<br>".join(f"{fmt(t)} ({pct(t, e_long)})" for t in info["long_tps"])),
        ("Стоп:", f"ниже {fmt(info['long_stop'])} ({pct(info['long_stop'], e_long)})"),
        ("Условия:", f"пробой и ретест зоны {fmt(le_lo)}"),
    ]
    short_lines = [
        ("Вход:", f"{fmt(se_lo)} – {fmt(se_hi)} (отскок/реакция)"),
        ("Цели:", "<br>".join(f"{fmt(t)} ({pct(t, e_short)})" for t in info["short_tps"])),
        ("Стоп:", f"выше {fmt(info['short_stop'])} ({pct(info['short_stop'], e_short)})"),
        ("Условия:", "сигнал на разворот в зоне"),
    ]

    col_l, col_s = st.columns(2)
    with col_l:
        st.markdown(
            card_html("ЛОНГ", "только после подтверждения", info["long_prob"],
                      "#26a69a", "rgba(38,166,154,0.08)", long_lines),
            unsafe_allow_html=True,
        )
    with col_s:
        st.markdown(
            card_html("ШОРТ", "при подтверждении", info["short_prob"],
                      "#ef5350", "rgba(239,83,80,0.08)", short_lines),
            unsafe_allow_html=True,
        )

    if info["inside"]:
        lo, hi, _ = info["inside"][0]
        st.markdown(
            f"<div style='border:1px solid #ef5350;border-radius:10px;"
            f"background:rgba(239,83,80,0.07);padding:10px 14px;margin-top:12px'>"
            f"<b>⚠️ СЕЙЧАС ЦЕНА ВНУТРИ ЗОНЫ {fmt(lo)} – {fmt(hi)}. ЖДЁМ ПОДТВЕРЖДЕНИЯ.</b><br>"
            f"<span style='color:#888;font-size:13px'>Входить в середине диапазона не рекомендуется.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Цена вне зон — работаем от ближайших уровней по сценариям выше.")

    st.caption(
        "«Вероятности» — эвристика по тренду/импульсу, а не статистическая оценка. "
        "Не финансовый совет."
    )


if __name__ == "__main__":
    main()
