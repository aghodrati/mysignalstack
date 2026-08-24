"""Two interactive views:
- Compare: click to add/remove tickers from quick-pick categories (companies,
  semiconductors, software, crypto, metals, oil & gas, defense, S&P sectors)
  and/or type other tickers, then compare cumulative return against SPY since
  a chosen date.
- Correlations: across the whole universe of tracked tickers, which pairs move
  together over a lookback window, and among those, which have diverged from
  their usual relationship right now.

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import html
import os
import re
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from dotenv import load_dotenv

# Loads .env into os.environ explicitly -- `uv run` does NOT do this automatically (confirmed
# live: `uv run python3 -c "import os; 'UPSTASH_REDIS_REST_URL' in os.environ"` prints False
# without this), so every finance.* module reading its secrets via plain os.environ.get (the
# convention finance.llm's OPENROUTER_API_KEY already uses) was silently getting nothing locally
# unless the shell itself had sourced .env first. load_dotenv() never overrides a var that's
# already set in the real environment, so this is safe layered under Streamlit Cloud's own
# secrets-bridging below.
load_dotenv()

from finance.backtest import buy_and_hold, rebalance_dates, run_backtest
from finance.claims import load_claims
from finance.correlation import divergence_now, pairwise_correlation
from finance.data import (
    get_earnings_history,
    get_insider_transactions,
    get_institutional_ownership,
    get_intraday_closes,
    get_major_holders,
    get_open_close,
    get_prices,
    get_upgrades_downgrades,
    get_volume,
)
from finance.dip import find_dip_trades, find_pending_dips
from finance.intraday import average_intraday_path
from finance.metrics import summary_table
from finance.momentum import (
    Direction,
    equal_weight_universe_weight_func,
    picks_by_rebalance,
    random_n_average_equity,
    top_n_momentum_weight_func,
)
from finance.loop_a_config import ticker_sectors, tracked_universe
from finance.newsloop import (
    CONCENTRATION_PROFILES,
    HORIZON_PROFILES,
    RISK_PROFILES,
    RULE_NAME,
    TYPE_COMPANY as TYPE_COMPANY_DISCOVERY,
    TYPE_THEME as TYPE_THEME_DISCOVERY,
    get_article_archive,
    load_discovery_candidates,
    save_discovery_candidates,
)
from finance.overnight import decompose_returns, summarize as summarize_overnight
from finance.panel import FACTOR_COLUMNS as PANEL_FACTOR_COLUMNS
from finance.pead import Direction as PeadDirection, find_earnings_streak_trades, find_pead_trades
from finance.portfolio import (
    closed_trades,
    create_portfolio,
    current_state,
    delete_portfolio,
    list_portfolios,
    load_meta,
    load_trades,
    save_rule_settings,
    undo_last_trade,
    valuation_history,
)
from finance.rules import (
    RuleConfig,
    analyst_momentum_candidates,
    dip_candidates,
    diverged_pairs_candidates,
    earnings_streak_candidates,
    run_rules,
)
from finance.ranking import (
    FACTOR_CATEGORIES,
    FACTOR_DESCRIPTIONS,
    HIGHER_IS_BETTER,
    build_factor_table,
    category_scores,
    composite_score,
    percentile_rank_table,
)
from finance.positions import open_positions
from finance.earnings_calls import latest_earnings_preview, load_earnings_call_history
from finance.fundamentals import load_fundamental_history
from finance.macro import (
    FINNHUB_PROXY_SYMBOLS,
    MACRO_SERIES,
    MONTHLY_NARRATIVE_SERIES,
    NARRATIVE_QUERIES,
    NARRATIVE_SERIES,
    latest_narrative,
    macro_snapshot,
)

# Which provider a series' narrative was grounded in (finance.macro.FINNHUB_PROXY_SYMBOLS vs.
# NARRATIVE_QUERIES) -- shown on the monthly card's footer so the source is visible per-card, not
# just in the page-level caption.
_NARRATIVE_SOURCE_LABEL: dict[str, str] = {
    **{key: "Finnhub" for key in FINNHUB_PROXY_SYMBOLS},
    **{key: "GNews" for key in NARRATIVE_QUERIES},
}
from finance import read_state
from finance.thesis import list_tickers_with_thesis, load_ticker_thesis
from finance.universe import QUICK_PICK_CATEGORIES, SP500_BENCHMARK, load_custom_tickers, save_custom_tickers

# Set on the hosted deployment only (e.g. a Streamlit Community Cloud secret).
# The Portfolio tab writes trade history to local disk, which a git-backed
# host doesn't persist across redeploys -- hidden there to avoid silently
# losing trades, not because the feature is broken.
HOSTED = os.environ.get("FINANCE_APP_HOSTED", "") == "1"

# Diverging pair for correlation (-1..+1, 0 = neutral gray).
DIVERGING_COLORSCALE = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
SP500_LINE_COLOR = "#e34948"
# Categorical slots 1 (blue) & 2 (orange) for the overnight/intraday pair.
OVERNIGHT_COLOR = "#2a78d6"
INTRADAY_COLOR = "#eb6834"

# ticker -> display name, deduped across every quick-pick category except S&P
# sectors (a sector ETF structurally contains many of the individual stocks in
# the universe, so those pairs dominate "most correlated" without meaning
# anything) and Defense (correlates on political/news catalysts rather than
# an economic relationship worth pairs-trading).
CORRELATION_EXCLUDED_CATEGORIES = {"S&P sectors", "Defense"}
TICKER_TO_NAME: dict[str, str] = {}
for _category, _options in QUICK_PICK_CATEGORIES.items():
    if _category in CORRELATION_EXCLUDED_CATEGORIES:
        continue
    for _name, _ticker in _options.items():
        TICKER_TO_NAME.setdefault(_ticker, _name)

# Equity-only categories for the momentum universe (excludes crypto, metals,
# and S&P sector ETFs, which aren't "stocks" in the hypothesis's sense).
STOCK_EXCLUDED_CATEGORIES = {"Crypto", "Metals", "S&P sectors"}
STOCK_TICKER_TO_NAME: dict[str, str] = {}
for _category, _options in QUICK_PICK_CATEGORIES.items():
    if _category in STOCK_EXCLUDED_CATEGORIES:
        continue
    for _name, _ticker in _options.items():
        STOCK_TICKER_TO_NAME.setdefault(_ticker, _name)

# Bridges Streamlit Cloud's secrets (st.secrets, set via the app's dashboard, not a real env var by
# default) into plain os.environ -- every finance.* module (finance.llm's OPENROUTER_API_KEY,
# finance.read_state's UPSTASH_REDIS_REST_URL/TOKEN) reads its secrets via os.environ.get, the same
# convention whether running locally (via .env) or deployed, so those modules stay Streamlit-
# agnostic. A key already set as a real env var (local dev) is left alone. st.secrets itself raises
# (lazily, on first access -- not when the object is obtained) if there's no secrets.toml anywhere
# (plain local dev via .env, no Streamlit secrets file at all) -- a normal, expected setup here,
# not an error, so each access is individually guarded rather than one try around the whole loop.
for _secret_key in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"):
    if _secret_key in os.environ:
        continue
    try:
        _secret_value = st.secrets[_secret_key]
    except Exception:
        continue
    os.environ[_secret_key] = _secret_value

st.set_page_config(page_title="Market comparisons", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    "<style>"
    # On a narrow (phone/iPad-width) screen Streamlit's sidebar otherwise takes up the full
    # viewport width while open, hiding all of the main content behind it -- a third of the width
    # leaves the page context visible/reachable at a glance instead. `!important` on width (only)
    # since Streamlit's own resize-drag feature sets an inline `style="width:...px"` on this
    # element, which without `!important` here would otherwise win over a plain stylesheet rule.
    # Deliberately NOT forcing min-width/max-width here: Streamlit's collapse animation drives the
    # sidebar closed by setting inline min-width/max-width to 0 (see its own collapsed-state
    # styling), and the browser always clamps a used `width` between min-width and max-width
    # regardless of `!important` on `width` itself -- forcing min-width alongside it (as a previous
    # version of this rule did) fought that clamp, leaving the sidebar stuck open and visible even
    # when "collapsed", and starving the main content of the freed flexbox space.
    "@media (max-width: 1024px){"
    "[data-testid='stSidebar']{width:33vw !important}"
    "}"
    "[data-testid='stSidebar'] div.block-container{padding-top:0}"
    "[data-testid='stSidebarUserContent']{padding-top:0}"
    "[data-testid='stSidebar'] .stExpander{margin-bottom:0}"
    "[data-testid='stSidebar'] div[data-testid='stExpanderDetails']{padding-top:0.1rem;padding-bottom:0.1rem}"
    "[data-testid='stSidebar'] summary{padding-top:0.15rem;padding-bottom:0.15rem;min-height:0}"
    "[data-testid='stSidebar'] summary p{font-size:0.78rem;font-weight:600}"
    "[data-testid='stSidebar'] button p{font-size:0.72rem}"
    "[data-testid='stSidebar'] [data-testid='stPills'] button{padding:0.1rem 0.5rem}"
    "[data-testid='stSidebar'] [data-testid='stVerticalBlock']{gap:0.25rem}"
    # Research trail entries (ticker-thesis history, article claims) are the only nested expanders
    # in the app (a trail entry expander inside a wrapping one, sometimes with a further-nested
    # "Source article text") -- this selector reaches exactly those, giving them a light tint
    # distinguishing them from top-level text without touching any other expander in the app.
    "[data-testid='stExpanderDetails'] [data-testid='stExpander']"
    "{background-color:rgba(151,166,195,0.15);border-radius:0.5rem}"
    # Per-claim Direction/Confidence/Expected return/Horizon metrics (Theses -> ticker -> Claims ->
    # a single claim) are nested three levels deep and shown once per claim, often many per ticker --
    # full-size st.metric styling (meant for a page's headline numbers) is too heavy repeated that
    # often, so shrink just these, leaving every other metric in the app (portfolio totals, the
    # ticker-level thesis, aggregation/fundamental history) untouched.
    "[class*='st-key-claim_metrics_'] [data-testid='stMetricValue']{font-size:1.4rem}"
    "[class*='st-key-claim_metrics_'] [data-testid='stMetricLabel'] p{font-size:0.8rem}"
    # Per-category fundamental-factor metrics (Growth/Valuation/.../Ownership) pack several raw
    # values into one metric's value string -- shrink so a multi-value string like "16.3x · 0.6x ·
    # 19.6x" fits without wrapping awkwardly.
    "[class*='st-key-fundamental_factors_'] [data-testid='stMetricValue']{font-size:0.85rem}"
    "[class*='st-key-fundamental_factors_'] [data-testid='stMetricLabel'] p{font-size:0.8rem}"
    # News/Fundamental confidence are the *inputs* to the headline Blended confidence metric above
    # them -- shrink so they read as a breakdown/footnote, not equal-weight siblings.
    "[class*='st-key-confidence_breakdown_'] [data-testid='stMetricValue']{font-size:0.85rem}"
    "[class*='st-key-confidence_breakdown_'] [data-testid='stMetricLabel'] p{font-size:0.75rem}"
    # Claim cards inside the Claims dialog -- lightly tinted to read as distinct cards while
    # scrolling, default size/spacing otherwise.
    "[class*='st-key-claim_card_']{background-color:rgba(151,166,195,0.10);border-radius:0.5rem}"
    # Below 1024px (phone and iPad-width tablets alike), keep Streamlit's
    # native header/sidebar controls untouched -- stExpandSidebarButton (the
    # only way to reopen a collapsed sidebar on a touch device) is rendered
    # *inside* stHeader, so hiding stHeader unconditionally was silently
    # deleting a touch device's only way to open the sidebar at all.
    # Only strip these on wide (desktop) viewports, where the sidebar (on the
    # Explore page) stays permanently visible and this chrome is genuinely
    # unused. stHeader itself is NOT hidden (unlike before this app had
    # multiple pages) -- st.navigation(position="top")'s page switcher
    # renders *inside* stHeader, so hiding it would hide the only way to
    # switch pages on desktop. No custom block-container padding-top either
    # (there used to be one, sized for a *hidden* header) -- Streamlit's own
    # default padding already accounts for the header's real height now that
    # it's visible again; a smaller custom value was making the page's own
    # top content render underneath the header.
    "@media (min-width: 1024px){"
    "[data-testid='stSidebarHeader']{display:none}"
    "[data-testid='stSidebarCollapseButton']{display:none}"
    "[data-testid='stExpandSidebarButton']{display:none}"
    "}"
    "</style>",
    unsafe_allow_html=True,
)

# Module-level so every render_*_tab function below can read them as plain globals -- only
# page_explore() (see bottom of file) ever assigns them, since only tabs living on that page
# use them; page_home()'s tabs (Research/This Week/Portfolio) never touch sidebar state.
picked_tickers: set[str] = set()
tickers_input: str = ""


def _render_equal_width_tab_css(key: str, n_tabs: int) -> None:
    """Equal-width segments wrapped onto exactly two rows -- st.segmented_control otherwise
    sizes each button to fit its own label (so "This Week" is visibly wider than "Rank") and
    packs everything onto one line. A grid with ceil(n/2) columns fills row 1 first, then row 2,
    regardless of how many tabs exist. Scoped to the widget's own .st-key-{key} class (the class
    Streamlit adds for any widget created with an explicit key=) -- st.pills elsewhere (e.g. the
    sidebar's ticker category pickers) render through this exact same stButtonGroup component, so
    an unscoped rule here would squeeze those into the same grid too. Each page's own segmented
    control passes its own key, since the two pages have different tab counts.
    """
    cols = -(-n_tabs // 2)  # ceil division
    st.markdown(
        f"""
        <style>
        .st-key-{key} div[data-testid="stButtonGroup"] > div {{
            display: grid;
            grid-template-columns: repeat({cols}, 1fr);
            width: 100%;
        }}
        .st-key-{key} div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{
            width: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_compare_tab() -> None:
    start_date = st.date_input(
        "Since", value=dt.date(2026, 1, 1), max_value=dt.date.today(), key="compare_since"
    )
    typed_tickers = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    tickers = sorted(picked_tickers | typed_tickers)

    all_tickers = sorted(set(tickers) | {SP500_BENCHMARK})
    with st.spinner("Fetching prices..."):
        prices = get_prices(all_tickers, start=start_date.isoformat())
    prices = prices.loc[prices.index >= pd.Timestamp(start_date)]

    combined = prices.dropna(axis=1, how="all")
    missing = set(all_tickers) - set(combined.columns)
    if missing:
        st.warning(f"No data found for: {', '.join(sorted(missing))}")

    if combined.empty:
        st.error("No price data available for this selection/date range.")
        return

    cumulative_return = combined / combined.bfill().iloc[0] - 1

    fig = go.Figure()
    for col in cumulative_return.columns:
        is_benchmark = col == SP500_BENCHMARK
        fig.add_trace(
            go.Scatter(
                x=cumulative_return.index,
                y=cumulative_return[col],
                name=col,
                line=dict(width=3, color=SP500_LINE_COLOR) if is_benchmark else dict(width=2),
            )
        )
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Cumulative return",
        xaxis_title="Date",
        hovermode="x unified",
        legend_title_text="",
        height=450,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch", key="chart_compare")

    summary = pd.DataFrame({"Total return": cumulative_return.iloc[-1]})
    summary[f"vs {SP500_BENCHMARK}"] = cumulative_return.iloc[-1] - cumulative_return[SP500_BENCHMARK].iloc[-1]
    summary = summary.sort_values("Total return", ascending=False)
    st.dataframe(summary.style.format("{:+.2%}"), width="stretch", key="table_compare")


def _select_page_ticker(ticker: str) -> None:
    st.session_state["ticker_page_selected_ticker"] = ticker
    st.session_state["ticker_page_view"] = "ticker"
    # Picked up once, right after this rerun, by _maybe_close_sidebar_on_mobile -- see that
    # function for why this needs a real DOM click rather than anything Streamlit exposes directly.
    st.session_state["_close_sidebar_after_ticker_pick"] = True


def _maybe_close_sidebar_on_mobile() -> None:
    """On a narrow (phone/iPad-width) screen, closes the sidebar right after a ticker is picked
    from it -- Streamlit has no Python-level API to collapse the sidebar on demand (only
    `initial_sidebar_state` at page load), so this reaches into the real page DOM the same
    same-origin way components/card_feed/index.html does, and clicks the sidebar's own native
    collapse button. Desktop is left alone (window.parent.innerWidth check) since there the sidebar
    isn't in the way and closing it on every pick would just be annoying.

    Fragile by nature -- `aria-label`/`data-testid` names are Streamlit-internal, not a public API,
    and could rename on a future Streamlit upgrade. Tries a few known-shape selectors defensively;
    if none match, this silently no-ops rather than breaking the page (see the try/catch in the JS).
    Only fires once per ticker pick (session_state flag consumed here), not on every later rerun.
    """
    if not st.session_state.pop("_close_sidebar_after_ticker_pick", False):
        return
    components.html(
        """
        <script>
        (function() {
            try {
                if (window.parent.innerWidth >= 1024) return;  // desktop -- leave the sidebar open
                var doc = window.parent.document;
                var sidebar = doc.querySelector('[data-testid="stSidebar"]');
                if (!sidebar) return;
                var btn = sidebar.querySelector('button[aria-label*="lose sidebar" i]')
                    || sidebar.querySelector('[data-testid="stSidebarCollapseButton"] button')
                    || sidebar.querySelector('[data-testid="stSidebarCollapseButton"]');
                if (btn) btn.click();
            } catch (e) {}
        })();
        </script>
        """,
        height=0, width=0,
    )


def _select_recent_view() -> None:
    st.session_state["ticker_page_view"] = "recent"


def _select_read_view() -> None:
    st.session_state["ticker_page_view"] = "read"


def _select_favorites_view() -> None:
    st.session_state["ticker_page_view"] = "favorites"


def _select_discovery_view() -> None:
    st.session_state["ticker_page_view"] = "discovery"


def _select_macro_view(series_key: str | None = None) -> None:
    """Switches to the Macro dashboard -- `series_key=None` (the "Macro"/"All" entries) shows
    every series, exactly as before per-series filtering existed; a specific key (clicking e.g.
    "Gold" in the sidebar's Macro expander) narrows the page down to just that series' own card(s).
    See _render_macro_page's own use of "ticker_page_macro_series".
    """
    st.session_state["ticker_page_view"] = "macro"
    st.session_state["ticker_page_macro_series"] = series_key


def _create_portfolio_clicked() -> None:
    # Runs before the rerun that re-instantiates the "portfolio_selected"
    # selectbox widget -- setting its session_state key here (rather than in
    # the `if st.button(...)` body below, which runs *after* that widget has
    # already been instantiated this pass) avoids Streamlit's "cannot modify
    # a widget's session_state after it's instantiated" exception.
    try:
        name = st.session_state["portfolio_new_name"]
        create_portfolio(name, st.session_state["portfolio_new_cash"])
        save_rule_settings(
            name.strip(), RULE_NAME,
            {
                "risk_profile": st.session_state["portfolio_new_risk_profile"],
                "concentration": st.session_state["portfolio_new_concentration"],
                "horizon": st.session_state["portfolio_new_horizon"],
            },
        )
        st.session_state["portfolio_selected"] = name.strip()
        st.session_state["_portfolio_create_error"] = None
        st.rerun()  # closes the dialog and returns to the main page with the new portfolio selected
    except ValueError as exc:
        st.session_state["_portfolio_create_error"] = str(exc)


def _delete_portfolio_clicked(name: str) -> None:
    delete_portfolio(name)
    del st.session_state["portfolio_selected"]


def _md(text: str) -> str:
    """Escapes a `$` before handing LLM-generated prose to st.write/st.markdown/st.caption --
    those render two `$` as inline LaTeX math (KaTeX), and financial text is full of dollar
    amounts, so a claim/thesis/summary mentioning e.g. "$30 million to $50 million" gets the
    text *between* the two dollar signs silently typeset as math instead of displayed literally
    (mangled spacing, unexpected color/styling). Every dynamic piece of text that ultimately
    comes from an LLM or an article (claims, context, thesis, reasoning, catalysts, invalidation,
    risks, article summaries) should be wrapped in this before display.
    """
    return text.replace("$", "\\$")


# Unit each finance.fundamentals raw factor is stored in -- see finance.ranking's own row[...]
# assignments (some are already *100 percentages, some raw ratios/dollars/counts) -- used only to
# format the fundamental card's per-category factor summary, not to re-derive/rank anything.
_FUNDAMENTAL_FACTOR_UNIT: dict[str, str] = {
    "revenue_growth": "pct100", "earnings_growth": "pct100",
    "forward_pe": "x", "peg_ratio": "x", "ev_to_revenue": "x",
    "operating_margin": "pct100", "fcf_margin": "pct100",
    "analyst_upside": "pct", "revisions_trend": "pct", "net_upgrades": "count",
    "institutional_flow": "pct", "insider_flow": "$", "low_short_interest": "pct",
}


def _format_factor(name: str, value: float) -> str:
    unit = _FUNDAMENTAL_FACTOR_UNIT.get(name, "")
    if unit == "pct100":
        return f"{value * 100:+.1f}%"
    if unit == "pct":
        return f"{value:+.1f}%"
    if unit == "x":
        return f"{value:.1f}x"
    if unit == "$":
        return f"${value:+,.0f}"
    if unit == "count":
        return f"{value:+.0f}"
    return f"{value:.2f}"


# The single most decision-relevant factor per category, used to boil the fundamental card's
# per-category summary down to one number instead of listing every raw factor -- a true composite
# would need percentile-ranking against a universe (deliberately not done for a single-ticker check,
# see finance.fundamentals's own module docstring), so this picks the most commonly-cited headline
# metric for each category instead of averaging incompatible units together.
_CATEGORY_HEADLINE_FACTOR: dict[str, str] = {
    "Growth": "revenue_growth",
    "Valuation": "forward_pe",
    "Quality": "operating_margin",
    "Sentiment": "analyst_upside",
    "Ownership": "institutional_flow",
}

# Human-readable label per raw factor key (finance.ranking.FACTOR_CATEGORIES) -- shown on the
# fundamental card's front instead of the category name (e.g. "Forward P/E", not "Valuation"),
# since the category alone doesn't say which actual metric drove the number.
_FACTOR_LABEL: dict[str, str] = {
    "revenue_growth": "Revenue Growth", "earnings_growth": "Earnings Growth",
    "forward_pe": "Forward P/E", "peg_ratio": "PEG Ratio", "ev_to_revenue": "EV/Revenue",
    "operating_margin": "Operating Margin", "fcf_margin": "FCF Margin",
    "analyst_upside": "Analyst Upside", "revisions_trend": "Analyst Revisions", "net_upgrades": "Net Upgrades",
    "institutional_flow": "Institutional Flow", "insider_flow": "Insider Flow", "low_short_interest": "Low Short Interest",
}

_FUNDAMENTAL_STYLE: dict[str, tuple[str, str]] = {
    "long": ("\U0001f7e2", "Fundamentals: Long"),
    "short": ("\U0001f534", "Fundamentals: Short"),
    "neutral": ("⚪", "Fundamentals: Neutral"),
}


# Per-ticker company domain, for a real logo (via a free logo API keyed by domain, not ticker) --
# covers config_loop_a.json's tracked universe as of this writing; a ticker added later or with no
# entry here just shows its plain name, no logo (never an error -- see _ticker_logo_html's onerror
# fallback). Only usable in raw-HTML contexts (st.markdown(unsafe_allow_html=True)), not native
# st.button, which only supports an emoji/Material-icon glyph, not an arbitrary image.
_TICKER_LOGO_DOMAINS = {
    "AAPL": "apple.com",
    "MSFT": "microsoft.com",
    "GOOGL": "google.com",
    "AMZN": "amazon.com",
    "META": "meta.com",
    "TSLA": "tesla.com",
    "ORCL": "oracle.com",
    "UBER": "uber.com",
    "RDDT": "reddit.com",
    # BRK-B (berkshirehathaway.com) omitted -- confirmed 404 against DuckDuckGo's icon index (that
    # famously bare-bones site apparently has no crawlable favicon), same for CVX/IREN below.
    "NVDA": "nvidia.com",
    "TSM": "tsmc.com",
    "ASML": "asml.com",
    "AMD": "amd.com",
    "ARM": "arm.com",
    "QCOM": "qualcomm.com",
    "MU": "micron.com",
    "INTC": "intel.com",
    "NBIS": "nebius.com",
    "AVGO": "broadcom.com",
    "XOM": "exxonmobil.com",
    "PLTR": "palantir.com",
    "LMT": "lockheedmartin.com",
    "QBTS": "dwavesys.com",
    "RGTI": "rigetti.com",
    "OKLO": "oklo.com",
    "SPCX": "spacex.com",
    "CRWV": "coreweave.com",
    "SKHY": "skhynix.com",
    "CBRS": "cerebras.ai",
}


def _ticker_logo_html(ticker: str, size_em: float = 1.3) -> str:
    """A small inline <img> for `ticker`'s company logo (DuckDuckGo's free favicon-by-domain
    service -- logo.clearbit.com, tried first, turned out to be dead: Clearbit shut down their free
    public Logo API after the HubSpot acquisition, confirmed via DNS no longer resolving at all,
    not just a sandbox networking quirk), or "" if no domain is mapped for it
    (_TICKER_LOGO_DOMAINS) -- callers just prepend this, no conditional needed on their end.
    `onerror` hides the tag entirely if the icon 404s, rather than showing a broken-image icon.
    """
    domain = _TICKER_LOGO_DOMAINS.get(ticker)
    if not domain:
        return ""
    return (
        f'<img src="https://icons.duckduckgo.com/ip3/{html.escape(domain)}.ico" '
        f'style="height:{size_em}em;width:{size_em}em;object-fit:contain;vertical-align:middle;'
        f'margin-right:0.35em;border-radius:0.25em;" onerror="this.style.display=\'none\'">'
    )

# A single light, minimal neutral tint for every card regardless of direction -- direction is
# carried by the arrow's own color instead (see _DIRECTION_ARROW), not the card background.
# Low-alpha so it reads fine in both Streamlit's light and dark themes (a solid hex background
# would only look right in one).
_KEEP_CARD_BACKGROUND = "rgba(151,166,195,0.08)"

# Minimal thin Unicode arrows (not emoji) colored to signal direction -- same green/red the rest
# of the app already uses (PANEL_PALETTE, SP500_LINE_COLOR).
_DIRECTION_ARROW = {
    "long": '<span style="color:#1baf7a">↑</span>',
    "short": '<span style="color:#e34948">↓</span>',
}


def _article_summaries() -> dict[str, str]:
    """link -> Stage A's own general one-paragraph summary (finance.newsloop.extract_event's
    "summary" field), for every article that got one -- read straight from events.json (the
    permanent Stage A archive, already committed/tracked), not a separate cache. An article
    processed before this field existed, or whose Stage A call failed to produce one, is just
    absent here.
    """
    return {
        link: entry["event"]["summary"]
        for link, entry in get_article_archive().items()
        if entry.get("event") and entry["event"].get("summary")
    }


# Both defined in finance.read_state now, not here -- finance.macro/fundamentals/earnings_calls
# also need to compute a card's id (to mark it unread again when a refresh overwrites what that id
# already pointed to), so the id recipe and the single hardcoded user live in one shared place
# rather than being duplicated per module.
_CURRENT_USER = read_state.CURRENT_USER
_card_id = read_state.card_id


def _mark_card_read(card_id: str) -> None:
    read_state.mark_read(_CURRENT_USER, card_id)


def _mark_card_unread(card_id: str) -> None:
    read_state.mark_unread(_CURRENT_USER, card_id)


def _mark_card_favorite(card_id: str) -> None:
    read_state.mark_favorite(_CURRENT_USER, card_id)


def _mark_card_unfavorite(card_id: str) -> None:
    read_state.mark_unfavorite(_CURRENT_USER, card_id)


# Dispatch table for whatever action the card_feed component reports back (see
# _render_keep_card_grid) -- one shared table rather than an if/elif chain, since the set of
# possible actions is closed and each is just a single-argument (card_id) call. "discard" is
# defined further down (needs _group_discovery_candidates/_discovery_card_id, not yet defined at
# this point in the file) and added to this same table right after.
_CARD_ACTIONS = {
    "read": _mark_card_read,
    "unread": _mark_card_unread,
    "favorite": _mark_card_favorite,
    "unfavorite": _mark_card_unfavorite,
}


def _flip_card_html(card_body: str, back_html: str | None) -> str:
    """One card's outer HTML -- a plain div if there's nothing to flip to (`back_html` is None),
    otherwise a <details>-based 3D flip card (see components/card_feed/index.html for the CSS that
    drives the flip). Shared by claim cards and fundamental cards so both flip the same way.

    back_html is wrapped in its own inner "keep-flip-back-scroll" div, separate from the
    "keep-flip-back" div that actually carries the transform/backface-visibility -- the iframe CSS
    never touches that inner div at all (harmless there), but the native-mode CSS
    (_inject_native_card_css) needs it: putting `overflow-y: auto` on the SAME element as a 3D
    `transform` is a real Chromium rendering bug (confirmed empirically -- content silently painted
    past its own clipped box instead of scrolling, with no visible box background for the
    overflow), so the scroll constraint has to live one level down from the transformed element.
    """
    if back_html is None:
        return f'<div class="keep-card" style="background:{_KEEP_CARD_BACKGROUND}">{card_body}</div>'
    return (
        f'<details class="keep-card-flip">'
        f'<summary class="keep-flip-summary"><div class="keep-flip-inner">'
        f'<div class="keep-card keep-flip-front" style="background:{_KEEP_CARD_BACKGROUND}">{card_body}</div>'
        f'<div class="keep-card keep-flip-back" style="background:{_KEEP_CARD_BACKGROUND}">'
        f'<div class="keep-flip-back-scroll">{back_html}</div>'
        f"</div>"
        f"</div></summary>"
        f"</details>"
    )


def _claim_card_html(c, article_summary: str | None) -> str:
    arrow = _DIRECTION_ARROW.get(c.direction, "➖")
    metrics_bits = [f"Importance {c.importance}/10", f"Confidence {c.confidence:.0%}"]
    if c.trade_worthy:
        metrics_bits.append(f"Return {c.expected_return_pct:+.1f}%")
        metrics_bits.append(f"{c.expected_horizon_days}d horizon")
    metrics_html = html.escape(" · ".join(metrics_bits))
    context_html = f'<div class="keep-card-context">{html.escape(c.context)}</div>' if c.context else ""
    logo_html = _ticker_logo_html(c.ticker, size_em=1.4)
    card_body = (
        f'<div class="keep-card-source">{logo_html}#{html.escape(c.source or "unknown")} '
        f'#{html.escape(c.ticker)}</div>'
        f'<div class="keep-card-claim">{arrow} {html.escape(c.claim)}</div>'
        f"{context_html}"
        f'<div class="keep-card-meta">{metrics_html} · {c.created.isoformat()}</div>'
    )
    # Always flips now -- the source link belongs on every card regardless of whether Stage A also
    # produced a summary (never generated on demand, that'd be a fresh LLM call per card); the
    # summary block above it is still conditional on that.
    back_bits = []
    if article_summary:
        back_bits.append(
            f'<div class="keep-card-summary-title">\U0001f4f0 Article summary</div>'
            f'<div class="keep-card-summary">{html.escape(article_summary)}</div>'
        )
    back_bits.append(
        f'<div class="keep-card-summary-title">\U0001f517 Source</div>'
        f'<div class="keep-card-summary"><a href="{html.escape(c.source_link)}" target="_blank" '
        f'rel="noopener noreferrer">{html.escape(c.source_title or c.source_link)}</a></div>'
    )
    back_html = "".join(back_bits)
    return _flip_card_html(card_body, back_html)


_CARD_FEED_COMPONENT = components.declare_component(
    "card_feed", path=str(Path(__file__).parent / "components" / "card_feed"),
)


def _inject_card_feed_iframe_css() -> None:
    """Strips whatever default chrome Streamlit/the browser puts around a custom component's own
    iframe (a border, a drop shadow, an opaque background) so the card_feed panel reads as part of
    the page rather than a visibly separate box. Covers both the selector recent Streamlit versions
    use (`data-testid="stCustomComponentV1"`) and a plain `iframe[title=...]` fallback, since the
    exact wrapper markup isn't a documented/stable API -- belt and suspenders, not a real dependency
    on either matching. Safe to call once per grid rendered (just another <style> tag).
    """
    st.markdown(
        """
        <style>
        div[data-testid="stCustomComponentV1"], iframe[title*="card_feed"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_native_card_css() -> None:
    """The card visual language (.keep-card/.keep-card-source/.keep-card-claim/.keep-flip-* etc.)
    lives ONLY inside components/card_feed/index.html's own <style> tag today -- scoped to that
    iframe's isolated document, invisible to the main Streamlit page. _render_keep_card_grid_native
    renders the exact same card HTML directly into the page instead, so without this, every card
    falls back to unstyled default browser rendering (no rounded card background/border, no muted
    source/meta colors, a plain default <details> disclosure triangle on the flip cards) -- not a
    deliberate redesign, just the CSS never having been carried over. Mirrors that file's rules
    closely but drops what doesn't apply outside an iframe: the component's own internal DOM
    structure (#feed/#grid/.card-cell/.card-star/.card-action-btn -- native mode has none of that,
    using plain st.container/st.button instead) and the hardcoded body text color (the iframe needed
    that since its document has nothing to inherit from; here `color: inherit` alone already picks
    up Streamlit's own theme-correct text color from its real ancestor elements, light or dark).
    Idempotent/cheap -- safe to call once per grid rendered, same as _inject_card_feed_iframe_css.
    """
    # User-adjustable via _render_card_display_settings' popover -- every card font-size below is
    # expressed as calc(base * var(--card-font-scale)) rather than a plain rem value, so one number
    # rescales every card on the page at once. The variable itself is injected via a tiny separate
    # f-string (not the whole block below) specifically to avoid having to escape every literal
    # `{`/`}` in the much larger static CSS that follows.
    font_scale = st.session_state.get("card_font_scale", 1.0)
    st.markdown(f"<style>:root {{ --card-font-scale: {font_scale}; }}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        /* Chrome's scroll-anchoring feature repositions scroll on the FIRST layout shift it sees
        near/above the viewport after a subtree mounts (confirmed empirically -- the first Fav/Read
        click after navigating to a ticker page jumps ~100-500px even on an already-settled page,
        every later click on the same mount is a no-op). The card grid gets fully remounted every
        time you navigate away and back, so this isn't a one-time cost -- disabling anchoring here
        is the fix, not working around one particular mutation. */
        [data-testid="stMain"] { overflow-anchor: none; }
        .keep-card, .card-action-btn { color: inherit; }
        .keep-card {
            border-radius: 0.6rem 0.6rem 0 0;
            padding: 0.9rem 1rem;
            background: rgba(151,166,195,0.08);
            border: 1px solid rgba(151,166,195,0.15);
            border-bottom: none;
        }
        .keep-card-source {
            font-size: calc(0.75rem * var(--card-font-scale)); color: #1baf7a; font-weight: 600;
            margin-bottom: 0.3rem;
        }
        .keep-card-claim {
            font-size: calc(0.92rem * var(--card-font-scale)); font-weight: 600; line-height: 1.35;
            margin-bottom: 0.4rem;
        }
        .keep-card-context {
            font-size: calc(0.82rem * var(--card-font-scale)); opacity: 0.85; line-height: 1.4;
            margin-bottom: 0.5rem;
        }
        .keep-card-meta { font-size: calc(0.72rem * var(--card-font-scale)); opacity: 0.65; }

        details.keep-card-flip summary { list-style: none; cursor: pointer; display: block; }
        details.keep-card-flip summary::-webkit-details-marker { display: none; }
        details.keep-card-flip summary::marker { content: ""; }
        /* Both faces always occupy the same shared grid cell (like the iframe version), so the
           card is exactly as tall as whichever face is taller and NEVER changes size when flipped
           -- no internal scrolling anywhere, full content always shown either way. The 3D rotateY
           flip animation is safe again now: the only reason it was dropped earlier was a real
           Chromium bug where an overflow:auto element fails to clip whenever any ancestor has a 3D
           transform (an earlier version capped+scrolled the back face and leaked content past its
           box) -- moot now that neither face scrolls at all, so nothing here conflicts with the
           transform. backface-visibility is what actually swaps which face renders as the parent
           rotates (each face is only visible for the half of the rotation where it faces the
           viewer) -- .keep-flip-back carries its own counter-rotation so it reads right-side-up
           once the parent has turned 180deg, not mirrored. */
        .keep-flip-summary { perspective: 1200px; }
        .keep-flip-inner {
            display: grid;
            width: 100%;
            transition: transform 0.5s cubic-bezier(0.4, 0.2, 0.2, 1);
            transform-style: preserve-3d;
        }
        details[open] .keep-flip-inner { transform: rotateY(180deg); }
        .keep-flip-front, .keep-flip-back {
            grid-area: 1 / 1;
            backface-visibility: hidden;
            -webkit-backface-visibility: hidden;
        }
        .keep-flip-back { transform: rotateY(180deg); }
        .keep-card-summary-title {
            font-size: calc(0.75rem * var(--card-font-scale)); color: #1baf7a; font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .keep-card-summary { font-size: calc(0.8rem * var(--card-font-scale)); opacity: 0.85; line-height: 1.4; }
        .keep-card-risk-item { margin-bottom: 0.6rem; }
        .keep-card-risk-item:last-child { margin-bottom: 0; }

        /* Action-button row: flush against the card above (no gap) and against each other (no
           gap), only the outer bottom corners rounded -- mirrors .keep-card's own bottom-flat/
           top-rounded shape so the two form one seamless box, no visible seam anywhere. Scoped to
           just the "cardnative_"-keyed card containers (see _render_keep_card_grid_native) so
           nothing else in the app that happens to use st.container(key=...)/st.columns is affected. */
        [class*="st-key-cardnative_"][data-testid="stVerticalBlock"],
        [class*="st-key-cardnative_"] [data-testid="stVerticalBlock"] {
            gap: 0 !important; row-gap: 0 !important;
        }
        [class*="st-key-cardnative_"] [data-testid="stElementContainer"] { margin: 0 !important; }
        [class*="st-key-cardnative_"] [data-testid="stLayoutWrapper"] { margin: 0 !important; }
        [class*="st-key-cardnative_"] [data-testid="stHorizontalBlock"] {
            gap: 0 !important; column-gap: 0 !important; row-gap: 0 !important;
            /* Streamlit stacks stHorizontalBlock columns vertically below its own mobile breakpoint
            -- fine for the app's normal columns, but the Read/Fav pair must stay side-by-side at
            any width (it's a single flush action bar, not two independent stacked controls). */
            flex-direction: row !important; flex-wrap: nowrap !important;
        }
        [class*="st-key-cardnative_"] [data-testid="stColumn"] {
            padding: 0 !important; margin: 0 !important; gap: 0 !important;
            width: auto !important; flex: 1 1 0 !important; min-width: 0 !important;
        }
        [class*="st-key-cardnative_"] [data-testid="stButton"] { margin: 0 !important; padding: 0 !important; }
        [class*="st-key-cardnative_"] [data-testid="stButton"] > button {
            border-radius: 0 !important;
            margin: 0 !important;
            padding: 0.15rem 0.4rem !important;
            min-height: 0 !important;
            height: auto !important;
            line-height: 1.2 !important;
            font-size: calc(0.7rem * var(--card-font-scale)) !important;
        }
        [class*="st-key-cardnative_"] [data-testid="stColumn"]:first-of-type [data-testid="stButton"] > button {
            border-bottom-left-radius: 0.6rem !important;
        }
        [class*="st-key-cardnative_"] [data-testid="stColumn"]:last-of-type [data-testid="stButton"] > button {
            border-bottom-right-radius: 0.6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Switch here to flip every card grid on the whole app back to the iframe/swipe version --
# "native" drops the swipe/flip-animation UX for plain in-page Streamlit buttons (no iframe
# boundary at all); "iframe" is the original custom-component behavior, left fully intact below so
# switching back is a one-line change, not a revert. See _render_keep_card_grid_native/_iframe's
# own docstrings for what each actually does.
_CARD_GRID_MODE = "native"  # "native" or "iframe"
# Default column count for the native grid (see _render_keep_card_grid_native) -- the iframe
# version's real CSS grid adapted column count to available width automatically; this doesn't, so
# it's user-adjustable instead via _render_card_display_settings' popover (st.session_state
# "card_columns", falling back to this constant when unset).
_NATIVE_GRID_COLUMNS = 2
# How many cards _render_keep_card_grid_native renders up front before requiring a "Show more"
# click -- see that function's own docstring for why a large page benefits from this (each card is
# several real Streamlit widgets, and Streamlit streams them to the browser as the script runs).
_NATIVE_PAGE_SIZE = 20


def _show_more_cards(shown_key: str, current_shown: int) -> None:
    st.session_state[shown_key] = current_shown + _NATIVE_PAGE_SIZE


def _render_card_display_settings() -> None:
    """Small "⚙️ Display" popover, right-aligned at the top of whichever card-listing page calls
    this (Recent/Read/Favorites/Discovery/ticker pages -- see the one call site right before their
    shared dispatch) -- lets the user pick the native grid's column count and a font-size scale for
    card text, both read back out of st.session_state by _render_keep_card_grid_native/
    _inject_native_card_css. Only meaningful in native mode (_CARD_GRID_MODE == "native") -- the
    iframe version's real CSS grid already adapts column count on its own and has no equivalent
    font-scale knob, so this renders nothing there rather than offering controls that do nothing.
    """
    if _CARD_GRID_MODE != "native":
        return
    _, popover_col = st.columns([6, 1])
    with popover_col:
        with st.popover("⚙️ Display", width="content"):
            st.segmented_control(
                "Columns", options=[2, 3], default=st.session_state.get("card_columns", _NATIVE_GRID_COLUMNS),
                key="card_columns", required=True,
            )
            st.segmented_control(
                "Text size", options=[0.85, 1.0, 1.15, 1.3],
                format_func=lambda s: {0.85: "Small", 1.0: "Normal", 1.15: "Large", 1.3: "X-Large"}[s],
                default=st.session_state.get("card_font_scale", 1.0), key="card_font_scale", required=True,
            )


def _visible_cards_and_action(
    cards: list[tuple[str, str]], show_read: bool, show_favorites: bool, primary_action: str | None,
) -> tuple[list[tuple[str, str]], str, int, set[str]]:
    """Shared filtering logic both grid implementations need: which cards are visible right now,
    what their primary action is, how many were hidden (read cards not currently being shown), and
    the full favorite-id set (both grid renderers need it too, to badge/star already-favorited
    cards -- returned from here rather than fetched a second time by each caller, since this
    function already needs favorite_ids itself on the Favorites page; each read_state call is a
    real network round-trip to Upstash, and duplicating it was adding a second ~seconds-scale delay
    on top of the favorite/unfavorite button's own write on every click).
    See _render_keep_card_grid_native's docstring for what show_read/show_favorites/primary_action
    each mean -- unchanged from the iframe version, just factored out so both share one definition.
    """
    read_ids = read_state.read_ids(_CURRENT_USER)
    favorite_ids = read_state.favorite_ids(_CURRENT_USER)
    if primary_action is not None:
        visible = cards
    elif show_favorites:
        visible = [(cid, body) for cid, body in cards if cid in favorite_ids]
        primary_action = "unfavorite"
    elif show_read:
        visible = [(cid, body) for cid, body in cards if cid in read_ids]
        primary_action = "unread"
    else:
        visible = [(cid, body) for cid, body in cards if cid not in read_ids]
        primary_action = "read"
    hidden_count = len(cards) - len(visible) if not show_read and not show_favorites else 0
    return visible, primary_action, hidden_count, favorite_ids


_ACTION_BUTTON_LABEL = {
    "read": "✓ Read",  # plain check mark -- U+1F5F8 (light check) looked nicer on paper but doesn't
    # actually have glyph coverage in the fonts this renders with (showed as a tofu box), confirmed
    # by screenshotting it -- reverted rather than ship a broken-looking icon.
    "unread": "↩ Unread",
    "favorite": "☆ Fav",  # not yet favorited -- plain outline star
    "unfavorite": "⭐ Unfav",  # already favorited -- filled star emoji, renders yellow
    "discard": "\U0001f5d1 Discard",
}


def _render_keep_card_grid_native(
    cards: list[tuple[str, str]], show_read: bool, show_favorites: bool,
    primary_action: str | None, key: str,
) -> None:
    """Plain-Streamlit alternative to _render_keep_card_grid_iframe -- no custom component, no
    iframe, every card rendered directly into the page via st.container + st.markdown(unsafe_allow_
    html) for the card's own HTML (the flip-card front/back still works with zero JS, since it's a
    pure CSS <details>/<summary> disclosure), with ordinary st.button widgets underneath instead of
    swipe gestures. Costs a full rerun of this whole grid per click (a real Python round-trip either
    way, same as the iframe version's action handling -- just without that version's client-side-
    only hide/reveal optimizations). Trades the animated swipe/dismiss UX for being unambiguously
    "part of the page" -- no visible seam, no separate scroll area, no iframe-specific CSS overrides
    needed.

    Buttons use on_click (not "if button(...): action(); st.rerun()") -- Streamlit already reruns
    the script automatically on a button click, and on_click's handler runs BEFORE that automatic
    rerun's script body executes, so the fresh _visible_cards_and_action call below already reflects
    the click's effect on that one rerun; a second explicit st.rerun() would just be redundant.

    Same show_read/show_favorites/primary_action contract as the iframe version (see that
    function's own docstring for the full semantics) -- both call _visible_cards_and_action so the
    two stay in sync. A favorite toggle button is also shown alongside the primary action, except on
    the Favorites page itself (where the primary action already IS unfavorite -- a second redundant
    button there would just be visual clutter) or when primary_action is a bespoke action with no
    read/favorite concept at all (e.g. Discovery's "discard"). Its key stays "..._fav_toggle"
    regardless of which of favorite/unfavorite is currently active, rather than baking the action
    name into the key -- this ONE button flips state in place on the same visible card (unlike the
    primary button, whose card disappears once clicked in every view that offers this toggle, so its
    key never needs to change in place), and giving it a stable key avoids Streamlit treating it as
    a brand-new widget on every toggle.
    """
    if not cards:
        return
    visible, primary_action, hidden_count, favorite_ids = _visible_cards_and_action(
        cards, show_read, show_favorites, primary_action,
    )
    if hidden_count:
        st.caption(f"{hidden_count} read card(s) hidden -- see the Read page in the sidebar.")
    if not visible:
        return
    _inject_native_card_css()
    show_favorite_toggle = primary_action in ("read", "unread")
    # Paginated, not all of `visible` at once -- Streamlit streams UI updates to the browser AS the
    # script executes (not batched at the end), and every card here is 3-4 separate real Streamlit
    # widgets (a container, the markdown, a button row), so a page with a couple hundred cards was
    # visibly "filling in" one card at a time over several seconds (confirmed: the old iframe-based
    # version sent its whole card list as one bulk JSON payload the component then painted client-
    # side in one shot -- this trades that instant bulk paint for genuinely being part of the page,
    # at the cost of a large page taking a while to stream in). Capping the initial render to
    # _NATIVE_PAGE_SIZE cards keeps that stream short; "Show more" reveals the next batch on demand.
    # Keyed by `key` (unique per grid/page) so paging state doesn't leak between different pages'
    # grids, and resets naturally to the default when `key` itself changes (e.g. switching tickers).
    shown_key = f"{key}_shown_count"
    shown_count = st.session_state.get(shown_key, _NATIVE_PAGE_SIZE)
    page = visible[:shown_count]
    remaining = len(visible) - len(page)
    # The original was a responsive CSS grid (auto-fill, minmax(260px, 1fr)) -- narrow Google-Keep-
    # style MASONRY tiles: a short card is immediately followed by the next one in the same column,
    # never waiting for its row-mates to catch up. st.columns() alone can't do that -- it lays out
    # in strict rows (a whole row of N cards renders together, so row 2 always starts only after
    # every card in row 1, at the tallest one's height, regardless of how short its neighbors were).
    # Distributing cards round-robin into N columns UP FRONT and rendering each column as its own
    # independent vertical stack approximates real masonry without needing to measure any card's
    # actual rendered height (which Python can't do) -- not perfectly height-balanced across columns
    # since it doesn't know heights, but each column still flows continuously on its own, which is
    # the actual complaint being fixed here.
    num_cols = st.session_state.get("card_columns", _NATIVE_GRID_COLUMNS)
    columns = st.columns(num_cols)
    for i, (cid, card_html) in enumerate(page):
        with columns[i % num_cols]:
            # "cardnative_" prefix so _inject_native_card_css's CSS can target just these
            # containers via [class*="st-key-cardnative_"] without touching any other keyed
            # container elsewhere in the app.
            with st.container(key=f"cardnative_{key}_{cid}", gap="xxsmall"):
                st.markdown(card_html, unsafe_allow_html=True)
                # Always st.columns (even the 1-button case) so the DOM shape is identical either
                # way -- _inject_native_card_css's first-of-type/last-of-type corner-rounding
                # selectors then apply correctly regardless of whether there's 1 or 2 buttons.
                btn_cols = st.columns(2 if show_favorite_toggle else 1, gap="xxsmall")
                btn_cols[0].button(
                    _ACTION_BUTTON_LABEL[primary_action], key=f"{key}_{cid}_{primary_action}", width="stretch",
                    on_click=_CARD_ACTIONS[primary_action], args=(cid,),
                )
                if show_favorite_toggle:
                    is_favorite = cid in favorite_ids
                    fav_action = "unfavorite" if is_favorite else "favorite"
                    btn_cols[1].button(
                        _ACTION_BUTTON_LABEL[fav_action], key=f"{key}_{cid}_fav_toggle", width="stretch",
                        on_click=_CARD_ACTIONS[fav_action], args=(cid,),
                    )
    if remaining > 0:
        st.button(
            f"Show {min(remaining, _NATIVE_PAGE_SIZE)} more ({remaining} left)",
            key=f"{shown_key}_btn", width="stretch",
            on_click=_show_more_cards, args=(shown_key, shown_count),
        )


def _render_keep_card_grid_iframe(
    cards: list[tuple[str, str]], show_read: bool = False, show_favorites: bool = False,
    primary_action: str | None = None, key: str = "feed",
) -> None:
    """Renders (card_id, card_html) pairs, newest first, as a card grid -- via the card_feed custom
    component (components/card_feed/index.html) instead of st.columns + st.button. That split
    exists because a real "mark as read"/"favorite" persist needs a genuine Python round-trip
    (finance.read_state), but hiding the card, animating a swipe, and paginating a long list should
    NOT force a full Streamlit rerun each time -- so the component owns all of that client-side (its
    own DOM, its own revealed/hidden bookkeeping) and only calls back into Python (via
    st.components' value-changed mechanism) to persist the mark in the background. See that file's
    own comments for the wire protocol and reconcile() for why the round-trip it does trigger
    doesn't cause a visible flicker: this function keeps passing the *same* filtered card list
    across reruns (the component ignores ids it already hid itself), so nothing above the fold
    visibly changes and there's no scroll jump.

    `key` must be unique per simultaneous grid on a page (Streamlit requires unique component
    instance keys) -- every call site below passes its own.

    At most one of `show_read`/`show_favorites` should be True. Default (both False, every caller
    except the dedicated Read/Favorites pages): cards already marked read are hidden -- swiping/
    tapping "Mark read" hides a card here; swiping right instead favorites it (a completely
    independent tag -- see finance.read_state's own docstring -- so favoriting does NOT hide a card
    from this default view, only read status does). Already-favorited cards still show here too,
    just with a star badge, so favoriting something doesn't cost you easy access to it while it's
    still unread.

    `show_read=True` (the Read page) shows exactly the already-read set instead, with an "Unread"
    action. `show_favorites=True` (the Favorites page) shows exactly the favorited set instead
    (regardless of read status), with an "Unfavorite" action -- both dedicated pages get only their
    own single swipe/button action, not the opposite-direction gesture too, since there's nothing
    else useful to swipe toward there.

    `primary_action`, if given, overrides all of the above: every card is shown (no read_state
    filtering at all) with this action name instead, and no swipe-right gesture -- for a card type
    with no read/favorite concept at all, e.g. the Discovery page's "discard" (removes the
    underlying candidates permanently, via finance.newsloop.save_discovery_candidates -- there's no
    read_state entry to filter on, and nothing to un-discard).
    """
    if not cards:
        return
    visible, primary_action, hidden_count, favorite_ids = _visible_cards_and_action(
        cards, show_read, show_favorites, primary_action,
    )
    if hidden_count:
        st.caption(f"{hidden_count} read card(s) hidden -- see the Read page in the sidebar.")
    if not visible:
        return
    _inject_card_feed_iframe_css()
    payload = [
        {
            "id": cid, "html": card_html, "action": primary_action,
            "starred": cid in favorite_ids,
            # Only the default view offers the opposite-direction (right-swipe) gesture, and only
            # toward favoriting -- see this function's own docstring.
            "swipe_right_action": "favorite" if primary_action == "read" else None,
        }
        for cid, card_html in visible
    ]
    result = _CARD_FEED_COMPONENT(cards=payload, key=key, default=None)
    if result and result.get("acted_id"):
        handler = _CARD_ACTIONS.get(result["action"])
        if handler:
            handler(result["acted_id"])


def _render_keep_card_grid(
    cards: list[tuple[str, str]], show_read: bool = False, show_favorites: bool = False,
    primary_action: str | None = None, key: str = "feed",
) -> None:
    """Dispatches to _render_keep_card_grid_native or _render_keep_card_grid_iframe per
    _CARD_GRID_MODE -- every existing call site keeps calling this one function; only the module-
    level mode switch changes which implementation actually runs. See both implementations' own
    docstrings for what each does differently.
    """
    fn = _render_keep_card_grid_native if _CARD_GRID_MODE == "native" else _render_keep_card_grid_iframe
    fn(cards, show_read, show_favorites, primary_action, key)


_FUNDAMENTAL_DIRECTION_ARROW = {
    "long": _DIRECTION_ARROW["long"],
    "short": _DIRECTION_ARROW["short"],
    "neutral": "➖",
}


# Quality's headline factor (Operating Margin) shown last on the fundamental card, after the other
# four, rather than in _FUNDAMENTAL_CATEGORIES' own insertion order.
_FACTOR_DISPLAY_ORDER: tuple[str, ...] = ("Growth", "Valuation", "Sentiment", "Ownership", "Quality")


def _factor_headline_bits(factors: dict) -> list[str]:
    """One formatted headline metric per category present in `factors` (Growth's revenue_growth,
    Valuation's forward_pe, etc. -- see _CATEGORY_HEADLINE_FACTOR), for the fundamental card's front
    face -- labeled by the actual factor (e.g. "Forward P/E"), not the category it belongs to, since
    the category name alone doesn't say which metric drove the number. Same headline-per-category
    the old Fundamentals dialog showed, just as inline text here instead of st.metric widgets (a
    raw-HTML card can't embed those). Shown in _FACTOR_DISPLAY_ORDER, not dict order.
    """
    bits = []
    for category in _FACTOR_DISPLAY_ORDER:
        values = factors.get(category)
        if not values:
            continue
        headline_factor = _CATEGORY_HEADLINE_FACTOR.get(category)
        if headline_factor not in values:
            headline_factor = next(iter(values))  # fallback: whatever's there
        label = _FACTOR_LABEL.get(headline_factor, headline_factor)
        bits.append(f"{label} {_format_factor(headline_factor, values[headline_factor])}")
    return bits


def _fundamental_card_html(ev: dict, ticker: str) -> str:
    direction = ev.get("fundamental_direction") or "neutral"
    arrow = _FUNDAMENTAL_DIRECTION_ARROW.get(direction, "➖")
    confidence = ev.get("fundamental_confidence")
    confidence_text = f"{confidence:.0%}" if confidence is not None else "n/a"
    summary_html = (
        f'<div class="keep-card-context">{html.escape(ev["summary"])}</div>' if ev.get("summary") else ""
    )
    changes = ev.get("key_changes") or []
    if changes:
        changes_html = "".join(
            f'<div class="keep-card-context">\U0001f504 {html.escape(c)}</div>' for c in changes
        )
    else:
        changes_html = '<div class="keep-card-context">No notable changes since the last check.</div>'
    factor_bits = _factor_headline_bits(ev.get("factors") or {})
    factors_html = (
        f'<div class="keep-card-context">\U0001f4ca {html.escape(" · ".join(factor_bits))}</div>'
        if factor_bits else ""
    )
    fv, cp, implied = ev.get("fair_value_estimate"), ev.get("current_price"), ev.get("implied_return_pct")
    # Price/fair-value/implied-return goes right under the direction line, bold -- the single most
    # decision-relevant number on this card (is the current price cheap or rich vs. the analyst
    # anchor), so it shouldn't be buried at the bottom in the same low-emphasis line as the date.
    price_bits = []
    if cp:
        price_bits.append(f"price ${cp:,.2f}")
    if fv:
        price_bits.append(f"fair value ${fv:,.2f}")
    if implied is not None:
        price_bits.append(f"implied {implied:+.1f}%")
    price_html = (
        f'<div class="keep-card-context">{html.escape(" · ".join(price_bits))}</div>' if price_bits else ""
    )
    meta_html = html.escape(f"yfinance - {ev['date']}")
    logo_html = _ticker_logo_html(ticker, size_em=1.4)
    # Summary/key_changes/factors all live on the front now -- risks are the only thing behind the
    # flip, since they're the one part worth a deliberate second look rather than at-a-glance.
    card_body = (
        f'<div class="keep-card-source">{logo_html}#Fundamentals #{html.escape(ticker)}</div>'
        f'<div class="keep-card-claim">{arrow} {direction.title()} · {confidence_text}</div>'
        f"{price_html}"
        f"{summary_html}"
        f"{changes_html}"
        f"{factors_html}"
        f'<div class="keep-card-meta">{meta_html}</div>'
    )
    risks = ev.get("risks") or []
    back_bits = ['<div class="keep-card-summary-title">⚠️ Risks</div>']
    if risks:
        back_bits += [
            f'<div class="keep-card-summary keep-card-risk-item">{html.escape(risk)}</div>' for risk in risks
        ]
    else:
        back_bits.append('<div class="keep-card-summary">No risks flagged.</div>')
    return _flip_card_html(card_body, "".join(back_bits))


_EARNINGS_CALL_DIRECTION_ARROW = _FUNDAMENTAL_DIRECTION_ARROW
# finance.macro only ever attaches direction to DIRECTIONAL_SERIES (oil/gold/silver/copper/
# bitcoin -- literal tradeable instrument prices), using "bullish"/"bearish" rather than "long"/
# "short" since there's no actual position/thesis behind it, just a display-only read -- see
# finance.macro.macro_narrative_snapshot's own docstring for why no expected_return/horizon either.
_MACRO_DIRECTION_ARROW = {
    "bullish": _DIRECTION_ARROW["long"],
    "bearish": _DIRECTION_ARROW["short"],
    "neutral": "➖",
}
_TONE_EMOJI = {"confident": "\U0001f4aa", "cautious": "\U0001f914", "defensive": "\U0001f6e1️", "evasive": "\U0001f440"}


_EARNINGS_CARD_FRONT_RISKS = 3  # rest overflow to the back face, alongside Key Q&A moments


def _earnings_call_card_html(ev: dict, ticker: str) -> str:
    """Opposite split from _fundamental_card_html: risks front-and-center on the front (summary,
    guidance, management tone, risks), key Q&A highlights behind the flip. Risks themselves are
    capped at _EARNINGS_CARD_FRONT_RISKS on the front (most important first, per extract_event's
    own prompt) -- a call with many hedged/flagged risks was measuring 1000px+ tall from risk lines
    alone, well past every other card type on the same page; the overflow still isn't lost, just
    moved to the back face next to the Q&A moments instead of uncapped on front.
    """
    direction = ev.get("earnings_direction") or "neutral"
    arrow = _EARNINGS_CALL_DIRECTION_ARROW.get(direction, "➖")
    confidence = ev.get("earnings_confidence")
    confidence_text = f"{confidence:.0%}" if confidence is not None else "n/a"
    summary_html = (
        f'<div class="keep-card-context">{html.escape(ev["summary"])}</div>' if ev.get("summary") else ""
    )
    guidance_bits = []
    if ev.get("guidance_summary"):
        guidance_bits.append(f'<div class="keep-card-context">\U0001f4c8 {html.escape(ev["guidance_summary"])}</div>')
    if ev.get("guidance_change"):
        guidance_bits.append(f'<div class="keep-card-context">\U0001f504 {html.escape(ev["guidance_change"])}</div>')
    guidance_html = "".join(guidance_bits)
    risks = ev.get("risks") or []
    front_risks, overflow_risks = risks[:_EARNINGS_CARD_FRONT_RISKS], risks[_EARNINGS_CARD_FRONT_RISKS:]
    if front_risks:
        risks_html = "".join(
            f'<div class="keep-card-context">⚠️ {html.escape(r)}</div>' for r in front_risks
        )
        if overflow_risks:
            risks_html += f'<div class="keep-card-context">+{len(overflow_risks)} more risk(s) on the back</div>'
    else:
        risks_html = '<div class="keep-card-context">No risks flagged.</div>'
    tone = ev.get("management_tone")
    tone_html = ""
    if tone:
        tone_emoji = _TONE_EMOJI.get(tone, "")
        tone_html = f'<div class="keep-card-context">{tone_emoji} Management tone: {html.escape(tone)}</div>'
    # Date goes last, same convention as claim/fundamental cards -- ticker identifies the card up
    # top instead (see the source tag below). Prefers the actual call date (transcript_date) over
    # this snapshot's own generation date, same as before.
    date_html = f'<div class="keep-card-meta">Motley Fool - {html.escape(ev.get("transcript_date", ev["date"]))}</div>'
    logo_html = _ticker_logo_html(ticker, size_em=1.4)
    card_body = (
        f'<div class="keep-card-source">{logo_html}#Earnings Call Transcript #{html.escape(ticker)}</div>'
        f'<div class="keep-card-claim">{arrow} {direction.title()} · {confidence_text}</div>'
        f"{tone_html}"
        f"{summary_html}"
        f"{guidance_html}"
        f"{risks_html}"
        f"{date_html}"
    )
    qa_moments = ev.get("key_qa_moments") or []
    back_bits = ['<div class="keep-card-summary-title">\U0001f4ac Key Q&A moments</div>']
    if qa_moments:
        back_bits += [
            f'<div class="keep-card-summary keep-card-risk-item">{html.escape(m)}</div>' for m in qa_moments
        ]
    else:
        back_bits.append('<div class="keep-card-summary">No usable Q&A section for this call.</div>')
    if overflow_risks:
        back_bits.append('<div class="keep-card-summary-title" style="margin-top:0.6rem;">⚠️ More risks</div>')
        back_bits += [
            f'<div class="keep-card-summary keep-card-risk-item">{html.escape(r)}</div>' for r in overflow_risks
        ]
    return _flip_card_html(card_body, "".join(back_bits))


_EARNINGS_REMINDER_WINDOW_DAYS = 3


@st.cache_data(ttl=3600)
def _next_earnings_info(ticker: str) -> dict | None:
    """{"when": tz-aware datetime, "eps_estimate": float | None} for `ticker`'s next scheduled (not
    yet reported) earnings call, or None if yfinance has nothing queued. yfinance reports these in
    the exchange's own local time (typically America/New_York) -- kept tz-aware here (unlike
    finance.data.get_earnings_history's own cached "earnings_date" column, which strips tz on
    purpose for its own callers) specifically so _earnings_reminder_card_html can convert it to
    Amsterdam time correctly. A live yfinance call, not a batch-pipeline artifact -- cheap/
    deterministic, no LLM involved -- so cached for an hour here (same convention as
    _cached_macro_snapshot) rather than persisted to disk like finance.data's own caches.
    """
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=6)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    now = pd.Timestamp.now(tz=raw.index.tz)
    upcoming = raw[raw.index > now].sort_index()
    if upcoming.empty:
        return None
    row = upcoming.iloc[0]
    estimate = row.get("EPS Estimate")
    return {
        "when": upcoming.index[0].to_pydatetime(),
        "eps_estimate": None if pd.isna(estimate) else float(estimate),
    }


def _earnings_reminder_card_id(ticker: str, next_dt: dt.datetime) -> str:
    # Keyed on the report's own date, not the exact datetime -- an intraday time correction from
    # yfinance shouldn't spawn a second reminder for the same call. This naturally becomes a fresh
    # unread card again once next quarter's call enters the reminder window.
    return _card_id(ticker, "earnings_reminder", next_dt.date().isoformat())


def _earnings_reminder_card_html(ticker: str, next_dt: dt.datetime, eps_estimate: float | None, as_of: dt.date) -> str:
    """Styled identically to every other keep-card. Front face is all real numbers, zero LLM cost:
    the next call's date/time converted to Amsterdam (yfinance's own timezone, typically
    America/New_York, is also shown alongside since that's the "official" market-hours reference),
    the consensus EPS estimate if yfinance has one, and last quarter's surprise plus a short
    beat/miss streak over its last 4 *reported* quarters -- the same finance.data.get_earnings_history
    every other earnings display in this app already reads from.

    Back face is finance.earnings_calls' short, Finnhub-grounded "what to watch" read (see that
    module's own docstring for why it's grounded rather than a bare LLM guess -- an ungrounded
    version asked for "specific numbers" fabricated a confident table of numbers from a stale,
    completely wrong training-data snapshot when tested live). Numbers never come from the LLM side
    of this card at all -- only the front face's real data does.
    """
    amsterdam = next_dt.astimezone(ZoneInfo("Europe/Amsterdam"))
    et = next_dt.astimezone(ZoneInfo("America/New_York"))

    estimate_html = ""
    if eps_estimate is not None:
        estimate_html = f'<div class="keep-card-context">Consensus estimate: ${eps_estimate:.2f} EPS</div>'

    reported = get_earnings_history(ticker).dropna(subset=["reported_eps", "surprise_pct"])
    reported = reported.sort_values("earnings_date").tail(4)
    streak_html = ""
    if not reported.empty:
        last = reported.iloc[-1]
        beat = last["surprise_pct"] >= 0
        arrow_html = '<span style="color:#1baf7a">▲ beat</span>' if beat else '<span style="color:#e34948">▼ missed</span>'
        beats = int((reported["surprise_pct"] >= 0).sum())
        avg_surprise = reported["surprise_pct"].mean()
        streak_html = (
            f'<div class="keep-card-meta">Last quarter: {arrow_html} by {abs(last["surprise_pct"]):.1f}% '
            f'(${last["reported_eps"]:.2f} vs ${last["eps_estimate"]:.2f} est.) · '
            f'Beat in {beats}/{len(reported)} of last {len(reported)} quarters, avg {avg_surprise:+.1f}%</div>'
        )

    logo_html = _ticker_logo_html(ticker, size_em=1.4)
    card_body = (
        f'<div class="keep-card-source">{logo_html}#Earnings Reminder #{html.escape(ticker)}</div>'
        f'<div class="keep-card-claim">\U0001f4c5 Earnings call</div>'
        f'<div class="keep-card-context">'
        f'{amsterdam.strftime("%a, %b %-d")} · {amsterdam.strftime("%H:%M")} Amsterdam time '
        f'({et.strftime("%H:%M")} ET)</div>'
        f"{estimate_html}"
        f"{streak_html}"
        f'<div class="keep-card-meta">{as_of.isoformat()}</div>'
    )

    preview = latest_earnings_preview(ticker)
    if preview and preview.get("call_date") == next_dt.date().isoformat():
        back_html = (
            f'<div class="keep-card-summary-title">\U0001f4ac What to watch</div>'
            f'<div class="keep-card-summary">{html.escape(preview["narrative"])}</div>'
            f'<div class="keep-card-meta" style="margin-top:0.4rem;">Finnhub - {preview["date"]}</div>'
        )
    else:
        back_html = (
            f'<div class="keep-card-summary-title">\U0001f4ac What to watch</div>'
            f'<div class="keep-card-summary">No preview generated yet for this call.</div>'
        )
    return _flip_card_html(card_body, back_html)


def _upcoming_earnings_cards(as_of: dt.date) -> list[tuple[str, str]]:
    """(card_id, card_html) for every tracked ticker whose next earnings call falls within
    _EARNINGS_REMINDER_WINDOW_DAYS of `as_of` -- pinned at the top of the Recent page's own
    "Upcoming" section (see _render_recent_page), independent of that page's own Dates/Cards
    filters (a reminder you're about to miss shouldn't be one pill-click away from disappearing).
    """
    # _next_earnings_info is cached for up to an hour (see its own docstring), so its "next call"
    # can lag up to that long after the call actually happens -- checked again here, against the
    # real current moment (not the date-only comparison below), so a reminder never lingers past
    # its own event even mid-cache-window.
    now = dt.datetime.now(ZoneInfo("Europe/Amsterdam"))
    cards: list[tuple[str, str]] = []
    for ticker in tracked_universe():
        # Cheap pre-filter first, using finance.data's own disk-cached earnings history (kept warm
        # by run_loop_a's daily needs_transcript_check pass over every tracked ticker) -- avoids a
        # live, per-ticker yfinance round-trip via _next_earnings_info for the whole universe on
        # every cold page load. Only tickers whose cached next date already looks close to the
        # reminder window get the slower, tz-aware precise check below. +/-1 day of slack around
        # the window absorbs the exchange-local-vs-Amsterdam date-boundary shift that the naive
        # cached date doesn't account for.
        # reported_eps.isna() alone isn't "future" -- yfinance sometimes leaves it NaN on old
        # rows it never backfilled (seen for real: an IREN row from 2024 with no reported_eps),
        # which .min()'d straight past the genuine upcoming date and skipped the ticker entirely.
        # Require the date itself to actually be upcoming too.
        cached_history = get_earnings_history(ticker)
        future_rows = cached_history[
            cached_history["reported_eps"].isna() & (cached_history["earnings_date"].dt.date >= as_of)
        ]
        if future_rows.empty:
            continue
        cached_next_date = future_rows["earnings_date"].min().date()
        days_until_cached = (cached_next_date - as_of).days
        if not (-1 <= days_until_cached <= _EARNINGS_REMINDER_WINDOW_DAYS + 1):
            continue
        info = _next_earnings_info(ticker)
        if info is None or info["when"] <= now:
            continue
        amsterdam_date = info["when"].astimezone(ZoneInfo("Europe/Amsterdam")).date()
        days_until = (amsterdam_date - as_of).days
        if 0 <= days_until <= _EARNINGS_REMINDER_WINDOW_DAYS:
            cid = _earnings_reminder_card_id(ticker, info["when"])
            card_html = _earnings_reminder_card_html(ticker, info["when"], info["eps_estimate"], as_of)
            cards.append((days_until, cid, card_html))
    cards.sort(key=lambda item: item[0])  # soonest first, not newest-first like every other grid
    return [(cid, card_html) for _, cid, card_html in cards]


_EARNINGS_RESULT_WINDOW_DAYS = 3


def _latest_reported_earnings(ticker: str) -> pd.Series | None:
    """`ticker`'s most recently *reported* quarter (reported_eps/surprise_pct both non-null), or
    None if finance.data.get_earnings_history has nothing reported yet. Shared by
    _recent_earnings_result_cards (the Recent page's pinned window) and _render_mixed_keep_cards
    (the Ticker page's own windowed inclusion) so both agree on exactly which quarter counts as
    "the last one."
    """
    hist = get_earnings_history(ticker).dropna(subset=["reported_eps", "surprise_pct"])
    return hist.sort_values("earnings_date").iloc[-1] if not hist.empty else None


def _earnings_result_card_id(ticker: str, report_date: str) -> str:
    return _card_id(ticker, "earnings_result", report_date)


def _earnings_result_card_html(ticker: str, row: pd.Series, as_of: dt.date) -> str:
    """A fast, LLM-free "results are in" card -- built purely from finance.data.
    get_earnings_history's own reported_eps/surprise_pct (already cached, zero extra cost), meant
    to fire the moment those numbers land, well before finance.earnings_calls' own transcript-based
    card (_earnings_call_card_html) can possibly be ready -- that one needs the full call
    transcript fetched and summarized by an LLM, which lags the actual print by a while. Same
    "#Earnings Call" tag as that card -- both describe the same real-world event, just at very
    different latency/depth, so they're meant to be seen as a fast headline followed later by the
    fuller read, not two unrelated things.
    """
    beat = row["surprise_pct"] >= 0
    arrow_html = '<span style="color:#1baf7a">▲ beat</span>' if beat else '<span style="color:#e34948">▼ missed</span>'
    logo_html = _ticker_logo_html(ticker, size_em=1.4)
    report_date = row["earnings_date"]
    report_date = report_date.date() if hasattr(report_date, "date") else report_date
    card_body = (
        f'<div class="keep-card-source">{logo_html}#Earnings Call #{html.escape(ticker)}</div>'
        f'<div class="keep-card-claim">\U0001f4ca Earnings results are in</div>'
        f'<div class="keep-card-context">'
        f'{arrow_html} estimates by {abs(row["surprise_pct"]):.1f}% '
        f'(${row["reported_eps"]:.2f} reported vs ${row["eps_estimate"]:.2f} est.)</div>'
        f'<div class="keep-card-meta">{report_date.isoformat()}</div>'
    )
    return _flip_card_html(card_body, None)


def _recent_earnings_result_cards(as_of: dt.date) -> list[tuple[str, str]]:
    """(card_id, card_html) for every tracked ticker whose latest reported quarter landed within
    _EARNINGS_RESULT_WINDOW_DAYS of `as_of` -- the Recent page's own pinned "Just Reported" section
    (see _render_recent_page), the after-the-fact counterpart to _upcoming_earnings_cards.
    """
    cards: list[tuple[int, str, str]] = []
    for ticker in tracked_universe():
        row = _latest_reported_earnings(ticker)
        if row is None:
            continue
        report_date = row["earnings_date"].date()
        days_since = (as_of - report_date).days
        if 0 <= days_since <= _EARNINGS_RESULT_WINDOW_DAYS:
            cid = _earnings_result_card_id(ticker, report_date.isoformat())
            card_html = _earnings_result_card_html(ticker, row, as_of)
            cards.append((days_since, cid, card_html))
    cards.sort(key=lambda item: item[0])  # most recent first
    return [(cid, card_html) for _, cid, card_html in cards]


def _render_mixed_keep_cards(
    claims: list, fundamental_events: list[dict], earnings_call_events: list[dict] | None = None,
    ticker: str = "",
) -> None:
    """Renders claim, fundamental, and earnings-call cards together in one grid, interleaved by
    date (newest first) -- see page_ticker's "Cards" filter for choosing which type(s) show, and
    Sources/Dates/Importance for the rest (Sources/Importance only apply to claims -- fundamental
    and earnings-call snapshots have neither). All three card builders share the same flip
    mechanism and column grid (_render_keep_card_grid), so any mix lays out identically to a
    single type alone. `ticker` is only used by the fundamental/earnings-call cards (claims already
    carry their own #source tag) -- it identifies which ticker a card belongs to, since neither
    snapshot type stores its own ticker in its JSON.
    """
    summaries = _article_summaries()
    dated: list[tuple[dt.date, str, str]] = [
        (c.created, c.id, _claim_card_html(c, summaries.get(c.source_link))) for c in claims
    ]
    dated += [
        (
            dt.date.fromisoformat(ev["date"]), _card_id(ticker, "fundamental", ev["date"]),
            _fundamental_card_html(ev, ticker),
        )
        for ev in fundamental_events
    ]
    dated += [
        (
            dt.date.fromisoformat(ev["date"]), _card_id(ticker, "earnings_call", ev["date"]),
            _earnings_call_card_html(ev, ticker),
        )
        for ev in (earnings_call_events or [])
    ]
    # LLM-free "results are in" card -- unlike the earnings-reminder card (which only ever means
    # anything within its own countdown window), this is a permanent addition to this ticker's card
    # history, same as fundamentals/earnings-call cards: no time window here, always the latest
    # reported quarter. Only _recent_earnings_result_cards (the Recent page's pinned "Just
    # Reported" section) applies a freshness window -- that's a temporary attention-grabbing
    # banner, a separate concern from this permanent per-ticker history entry.
    report_row = _latest_reported_earnings(ticker) if ticker else None
    if report_row is not None:
        report_date = report_row["earnings_date"].date()
        dated.append((
            report_date, _earnings_result_card_id(ticker, report_date.isoformat()),
            _earnings_result_card_html(ticker, report_row, dt.date.today()),
        ))
    if not dated:
        st.caption("No cards to show.")
        return
    dated.sort(key=lambda item: item[0], reverse=True)
    _render_keep_card_grid([(cid, card_html) for _, cid, card_html in dated], key=f"feed_ticker_{ticker}")


@st.dialog("Claims", width="medium")
def _claims_dialog(ticker: str, claims: list) -> None:
    """Experimental alternative to nested expanders: every claim for `ticker`
    shown as its own card in a scrollable modal, closeable without losing
    your place in the Theses list underneath. Shows the source article's own
    Stage A summary if it has one (see _article_summaries) -- never
    generates one on demand, that'd be a fresh LLM call just to populate a
    claim card.
    """
    summaries = _article_summaries()
    n_long = sum(1 for c in claims if c.direction == "long")
    n_short = sum(1 for c in claims if c.direction == "short")
    st.caption(
        f"{ticker}  ·  {len(claims)} claim(s), newest first  ·  \U0001f53c {n_long} long  ·  "
        f"\U0001f53d {n_short} short"
    )
    for c in claims:
        with st.container(border=True, key=f"claim_card_{c.id}"):
            badge = {"long": "🔼", "short": "🔽"}.get(c.direction, "➖")
            st.markdown(f"**{badge} {_md(c.claim)}**")
            st.markdown(f":green[#{c.source or 'unknown'}]")
            st.caption(
                f"{c.created}  ·  importance {c.importance}/10  ·  "
                f"[{c.source_title}]({c.source_link})  ·  event_type={c.event_type}"
            )
            # Trade-worthy/confidence are meaningful for every claim, trade-worthy or not -- direction
            # is already shown via the badge above, no need to repeat it as a metric too. Only the
            # sizing fields (return/horizon) genuinely don't exist for a claim too vague/indirect to
            # size a trade on (the extractor leaves them at 0 rather than guessing), so those stay
            # gated behind trade_worthy. Shown above the context paragraph, not below, so the
            # at-a-glance numbers don't require scrolling past a paragraph of prose first.
            with st.container(key=f"claim_metrics_{c.id}"):
                if c.trade_worthy:
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("Trade-worthy", "Yes")
                    cm2.metric("Confidence", f"{c.confidence:.0%}")
                    cm3.metric("Expected return", f"{c.expected_return_pct:+.1f}%")
                    cm4.metric("Horizon", f"{c.expected_horizon_days}d")
                else:
                    cm1, cm2 = st.columns(2)
                    cm1.metric("Trade-worthy", "No")
                    cm2.metric("Confidence", f"{c.confidence:.0%}")
            if c.context:
                st.write(_md(c.context))
            if not c.trade_worthy:
                st.caption(
                    "Not specific/actionable enough to size a trade on -- recorded as evidence "
                    "for the ticker's overall thesis only."
                )
            article_summary = summaries.get(c.source_link)
            if article_summary:
                st.caption(f"\U0001f4f0 Summary of article: {_md(article_summary)}")


@st.dialog("Aggregation history", width="medium")
def _aggregation_history_dialog(ticker: str, aggregated_events: list) -> None:
    """Same card-in-a-scrollable-modal treatment as the Claims dialog, for
    Stage C's synthesized-thesis snapshots over time.
    """
    st.caption(f"{ticker}  ·  {len(aggregated_events)} aggregation(s), newest first")
    for n, ev in reversed(list(enumerate(aggregated_events, 1))):
        with st.container(border=True, key=f"agg_card_{ticker}_{n}"):
            st.markdown(f"**#{n}  \U0001f9e9  {ev['date']}  ·  {ev.get('claims_considered', '?')} claim(s) considered**")
            a1, a2, a3 = st.columns(3)
            a1.metric("Direction", ev["direction"])
            a2.metric("Confidence", f"{ev['confidence']:.0%}")
            a3.metric(
                "Expected return", f"{ev['expected_return_pct']:+.1f}%",
                f"{ev['expected_horizon_days']}d horizon", delta_color="off",
            )
            fundamental_direction = ev.get("fundamental_direction")
            fundamental_confidence = ev.get("fundamental_confidence")
            if fundamental_direction is not None:
                conviction = f" ({fundamental_confidence:.0%} conviction)" if fundamental_confidence is not None else ""
                st.caption(f"Fundamental picture Stage C weighed in: {fundamental_direction}{conviction}")
            st.write(f"Thesis: {_md(ev['thesis'])}")
            if ev.get("catalysts"):
                st.write("Catalysts: " + _md(", ".join(ev["catalysts"])))
            if ev.get("invalidation"):
                st.write("Invalidation: " + _md(", ".join(ev["invalidation"])))
            st.caption(_md(ev["reasoning"]))


@st.dialog("Fundamentals", width="medium")
def _fundamentals_dialog(ticker: str, fundamental_events: list) -> None:
    """Same card-in-a-scrollable-modal treatment as the Claims dialog, for the independent
    fundamental snapshot history (finance.fundamentals.fundamental_snapshot) -- a standalone read
    of the business/valuation picture, NOT scored against any particular thesis (see page_ticker's
    own inline foldable card grid, _render_fundamental_keep_cards, for the primary way to browse
    these; this dialog is the compact version used from Research's per-ticker Theses expander).
    """
    st.caption(f"{ticker}  ·  {len(fundamental_events)} check(s), newest first")
    for n, ev in reversed(list(enumerate(fundamental_events, 1))):
        with st.container(border=True, key=f"fund_card_{ticker}_{n}"):
            icon, headline = _FUNDAMENTAL_STYLE.get(
                ev.get("fundamental_direction"), ("", ev.get("fundamental_direction", "unknown"))
            )
            st.markdown(f"**#{n}  {icon}  {headline}  ·  {ev['date']}**")
            confidence = ev.get("fundamental_confidence")
            st.metric(
                "Conviction", f"{confidence:.0%}" if confidence is not None else "--",
                help="0% = no conviction, 100% = strong conviction in the direction above.",
            )
            fv, cp = ev.get("fair_value_estimate"), ev.get("current_price")
            if fv and cp:
                implied = ev.get("implied_return_pct")
                implied_text = f"  ·  implied return {implied:+.1f}%" if implied is not None else ""
                st.caption(f"Analyst fair value estimate: \\${fv:.2f}  ·  Current price: \\${cp:.2f}{implied_text}")
            if ev.get("summary"):
                st.write(_md(ev["summary"]))
            for change in ev.get("key_changes") or []:
                st.write(f"\U0001f504 {_md(change)}")
            factors = ev.get("factors")
            if factors:
                present = {c: v for c, v in factors.items() if v}
                if present:
                    with st.container(key=f"fundamental_factors_{ticker}_{n}"):
                        cat_cols = st.columns(len(present))
                        for col, (category, values) in zip(cat_cols, present.items()):
                            headline_factor = _CATEGORY_HEADLINE_FACTOR.get(category)
                            if headline_factor not in values:
                                headline_factor = next(iter(values))  # fallback: whatever's there
                            value = values[headline_factor]
                            higher_better = HIGHER_IS_BETTER.get(headline_factor, True)
                            arrow = "▲" if higher_better else "▼"
                            description = FACTOR_DESCRIPTIONS.get(category, {}).get(headline_factor, "")
                            col.metric(
                                f"{category} {arrow}", _format_factor(headline_factor, value),
                                help=description or None,
                            )
            for risk in ev.get("risks") or []:
                st.write(f"⚠️ risk: {_md(risk)}")
            st.caption(_md(ev.get("reasoning", "")))


@st.dialog("Critic", width="medium")
def _critic_dialog(ticker: str, critic_events: list) -> None:
    """Same card-in-a-scrollable-modal treatment as the Claims dialog, for
    the critic pass's history: deterministic guardrails (source
    concentration, evidence thinness, staleness) plus one LLM red-team call
    -- both dampen confidence, never raise it (see finance.critic and
    finance.thesis._deterministic_critic_flags).
    """
    st.caption(f"{ticker}  ·  {len(critic_events)} check(s), newest first")
    for n, ev in reversed(list(enumerate(critic_events, 1))):
        with st.container(border=True, key=f"critic_card_{ticker}_{n}"):
            multiplier = ev.get("final_multiplier", 1.0)
            if multiplier >= 0.99:
                icon, headline = "✅", "No material concerns"
            else:
                icon, headline = "🟡", f"Confidence dampened {1 - multiplier:.0%}"
            st.markdown(f"**#{n}  {icon}  {headline}  ·  {ev['date']}**")
            c1, c2 = st.columns(2)
            c1.metric("Confidence before critic", f"{ev['confidence_before']:.0%}")
            c2.metric("Confidence after critic", f"{ev['confidence_after']:.0%}")
            for flag in ev.get("deterministic_flags") or []:
                st.write(f"\U0001f6a9 {flag}")
            for concern in ev.get("llm_concerns") or []:
                st.write(f"\U0001f9d0 {_md(concern)}")
            if ev.get("llm_reasoning"):
                st.caption(_md(ev["llm_reasoning"]))


def render_research_tab() -> None:
    """Read-only knowledge/recommendation view of Loop A's output -- global,
    not scoped to any portfolio, and with no trade-execution affordance of
    any kind. Deliberately separate from the Portfolio tab (which owns
    actual buy/sell/undo actions) so this can be browsed -- e.g. on a public
    hosted deploy, see app.py's HOSTED flag -- without exposing anything
    that moves money. The optional portfolio picker below is only for the
    "already holding this" cross-reference; it never enables a trade.
    """
    st.caption(
        "Global, shared across every portfolio -- finance.claims (every article-level claim ever "
        "extracted) synthesized by finance.thesis into one current view per ticker. Read-only: "
        "no trades happen here, see the Portfolio tab for that."
    )
    existing_portfolios = list_portfolios()
    context_portfolio = st.selectbox(
        "Cross-reference against a portfolio's holdings (optional)", options=existing_portfolios,
        index=None, placeholder="None -- just browsing", key="research_context_portfolio",
    )

    theses_tickers = list_tickers_with_thesis()
    if not theses_tickers:
        st.caption("No theses yet -- run Loop A against any portfolio to populate this.")
        return

    all_claims_by_ticker = {ticker: load_claims(ticker) for ticker in theses_tickers}
    all_sources = sorted({c.source or "unknown" for claims in all_claims_by_ticker.values() for c in claims})
    st.caption(
        "Sources -- deselect a source to hide its claims from this view only, nothing is "
        "deleted. Doesn't change a thesis's own confidence/direction (Stage C already synthesized "
        "those from every claim, selected or not)."
    )
    selected_sources = st.pills(
        "Sources", options=all_sources, default=all_sources, selection_mode="multi",
        key="research_sources_selected", label_visibility="collapsed",
    )
    masked_sources = set(all_sources) - set(selected_sources)

    # Live price-action context (momentum, relative strength vs. S&P 500, relative volume) --
    # deterministic, no LLM, never cached/blended into the thesis itself (see finance.ranking's
    # own Momentum category, deliberately excluded from finance.fundamentals). Purely informational
    # for a human deciding entry/exit timing -- Loop A's own trade logic never looks at this.
    # Fetched once for every ticker shown here, not per-ticker, same batching the Rank tab uses.
    _THESES_MOMENTUM_WEEKS = 12
    technical_lookback_days = _THESES_MOMENTUM_WEEKS * 7 + 15
    technical_start = (dt.date.today() - dt.timedelta(days=technical_lookback_days)).isoformat()
    # Purely best-effort: yfinance rate-limits are common on shared cloud IPs (e.g. Streamlit
    # Community Cloud) and this context is display-only, so a fetch failure here should never
    # take down the whole Theses list -- just show it without the price-action line.
    try:
        with st.spinner("Fetching price action..."):
            technical_prices = get_prices(theses_tickers + [SP500_BENCHMARK], start=technical_start)
        if SP500_BENCHMARK in technical_prices.columns:
            technical_factors = build_factor_table(
                theses_tickers, technical_prices, technical_prices[SP500_BENCHMARK],
                momentum_weeks=_THESES_MOMENTUM_WEEKS,
            )
        else:
            technical_factors = pd.DataFrame()
    except Exception:
        st.caption("⚠️ Price action unavailable right now (rate-limited) -- theses shown without it.")
        technical_factors = pd.DataFrame()
    theses_rows = []
    for ticker in theses_tickers:
        tt = load_ticker_thesis(ticker)
        if tt is None:
            continue
        visible = [c for c in all_claims_by_ticker[ticker] if (c.source or "unknown") not in masked_sources]
        claims = sorted(visible, key=lambda c: c.created, reverse=True)
        theses_rows.append((ticker, tt, claims))
    # Most recently (re)aggregated first -- tt.updated is set to as_of every time Stage C
    # (update_ticker_thesis) actually re-synthesizes this ticker's thesis, so this surfaces
    # whatever a run just acted on, unlike sorting by a claim's own article-publish date (which a
    # backfilled old article would keep buried regardless of how recently it was added). An
    # earnings-window fundamentals-only refresh (finance.thesis.refresh_fundamentals) does
    # NOT touch this -- it only appends a "fundamental" event, no new "aggregated" one.
    theses_rows.sort(key=lambda row: row[1].updated, reverse=True)
    for ticker, tt, claims in theses_rows:
        aggregated_events = [ev for ev in tt.history if ev["event"] == "aggregated"]
        # New-claims indicator: compares the latest aggregation's claims_considered against the
        # one before it (0 if this is the ticker's first-ever aggregation -- every claim behind a
        # brand-new thesis is new by definition). Every "aggregated" event is now always a genuine
        # new-claims-driven Stage C run (refresh_fundamentals no longer appends one), so a positive
        # delta always means real new evidence fed this update. A real emoji (not markdown color
        # syntax) so it renders green right in the expander title -- st.expander labels only
        # support a narrow markdown subset that excludes color spans.
        new_claims_badge = ""
        if aggregated_events:
            previous_considered = aggregated_events[-2].get("claims_considered", 0) if len(aggregated_events) >= 2 else 0
            delta = aggregated_events[-1].get("claims_considered", 0) - previous_considered
            if delta > 0:
                new_claims_badge = f"  \U0001f7e2 +{delta}"
        n_aggregations = len(aggregated_events)
        latest_claim_date = f", latest {claims[0].created.isoformat()}" if claims else ""
        direction_arrow = {"long": "\U0001f53c", "short": "\U0001f53d"}.get(tt.direction, "➖")
        title = (
            f"{ticker}  ·  {direction_arrow}  ·  conf={tt.confidence:.0%}  ·  "
            f"{n_aggregations} aggregation(s)  ·  {len(claims)} claim(s){latest_claim_date}{new_claims_badge}"
        )
        with st.expander(title, expanded=True):
            if context_portfolio:
                held = open_positions(context_portfolio, ticker=ticker)
                if held:
                    p = held[0]
                    st.info(
                        f"**{context_portfolio}** is currently holding {ticker} -- opened {p.created} at "
                        f"entry confidence {p.entry_confidence:.0%}, {p.expected_horizon_days}d horizon."
                    )
            st.write(f"**Thesis:** {_md(tt.thesis)}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Direction", tt.direction)
            m2.metric("Confidence", f"{tt.confidence:.0%}")
            m3.metric("Expected return", f"{tt.expected_return_pct:+.1f}%")
            m4.metric("Horizon", f"{tt.expected_horizon_days}d")
            if ticker in technical_factors.index:
                tech = technical_factors.loc[ticker]
                parts = []
                if pd.notna(tech.get("momentum")):
                    parts.append(f"Momentum ({_THESES_MOMENTUM_WEEKS}w): {tech['momentum']:+.1f}%")
                if pd.notna(tech.get("relative_strength")):
                    parts.append(f"vs S&P 500: {tech['relative_strength']:+.1f}pp")
                if pd.notna(tech.get("relative_volume")):
                    parts.append(f"Volume: {tech['relative_volume']:.1f}x avg")
                if parts:
                    st.caption("\U0001f4c8 " + "  ·  ".join(parts) + "  (context only, not used in the thesis)")

            fundamental_events = [ev for ev in tt.history if ev["event"] == "fundamental"]
            critic_events = [ev for ev in tt.history if ev["event"] == "critic"]

            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            with btn_col1:
                if st.button(f"Claims ({len(claims)})", key=f"claims_dialog_btn_{ticker}"):
                    _claims_dialog(ticker, claims)
            with btn_col2:
                if critic_events and st.button(
                    f"Critic ({len(critic_events)})", key=f"critic_dialog_btn_{ticker}"
                ):
                    _critic_dialog(ticker, critic_events)
            with btn_col3:
                if fundamental_events and st.button(
                    f"Fundamentals ({len(fundamental_events)})", key=f"fund_dialog_btn_{ticker}"
                ):
                    _fundamentals_dialog(ticker, fundamental_events)
            with btn_col4:
                if aggregated_events and st.button(
                    f"Theses ({len(aggregated_events)})", key=f"agg_dialog_btn_{ticker}"
                ):
                    _aggregation_history_dialog(ticker, aggregated_events)


@st.dialog("New portfolio")
def _new_portfolio_dialog() -> None:
    st.text_input("Name", key="portfolio_new_name", placeholder="e.g. momentum_top5")
    st.number_input(
        "Starting cash ($)", min_value=100.0, value=10000.0, step=500.0, key="portfolio_new_cash"
    )
    st.caption("Loop A strategy -- how this portfolio reacts to Loop A's theses. Fixed at creation.")
    st.selectbox(
        "Risk profile", options=list(RISK_PROFILES), index=1, key="portfolio_new_risk_profile",
        help=(
            "How high a ticker-thesis's confidence must be before this portfolio trades it. "
            "conservative=75%/85%, balanced=60%/75% (the long-standing default), aggressive=50%/65% "
            "(min confidence to trade at all / to size a full position)."
        ),
    )
    st.selectbox(
        "Concentration", options=list(CONCENTRATION_PROFILES), index=1, key="portfolio_new_concentration",
        help=(
            "Position sizing and how many positions can be open at once. diversified=3%/1.5% of "
            "portfolio value per position, up to 12 open; balanced=5%/2.5%, up to 8 (the "
            "long-standing default); concentrated=10%/5%, up to 4."
        ),
    )
    st.selectbox(
        "Horizon", options=list(HORIZON_PROFILES), index=2, key="portfolio_new_horizon",
        help=(
            "Only opens a position if the ticker-thesis's own expected_horizon_days falls in "
            "range. short_term=7-90 days, long_term=90-730 days, any=no restriction (the "
            "long-standing default). Doesn't affect closing -- positions still close on their "
            "own thesis's horizon regardless of this setting."
        ),
    )
    st.button("Create", key="portfolio_new_create", on_click=_create_portfolio_clicked)
    if st.session_state.get("_portfolio_create_error"):
        st.error(st.session_state["_portfolio_create_error"])


def render_portfolio_tab() -> None:
    st.caption(
        "Paper-trading portfolios. Each one is its own trade log saved to disk under "
        "`output/portfolios/`, so it survives closing the app. Trades come from Loop A "
        "(see run_loop_a.py) -- for rule-driven, automated trades and backtesting, see the "
        "Simulation tab."
    )

    existing = list_portfolios()
    name_col, create_col, delete_col = st.columns([2, 1, 1])
    with name_col:
        selected = st.selectbox(
            "Portfolio", options=existing, index=None, placeholder="Choose a portfolio",
            label_visibility="collapsed", key="portfolio_selected",
        )
    with create_col:
        if st.button("New portfolio", key="portfolio_new_open"):
            _new_portfolio_dialog()
    with delete_col:
        if selected:
            with st.popover("Delete portfolio"):
                st.write(f"Permanently delete **{selected}** and its trade history?")
                st.button(
                    "Confirm delete", key="portfolio_delete_confirm",
                    on_click=_delete_portfolio_clicked, args=(selected,),
                )

    if not selected:
        st.info("Create a portfolio to get started." if not existing else "Choose a portfolio above.")
        return

    meta = load_meta(selected)
    cash, positions = current_state(selected)
    prices_now: dict[str, float] = {}
    if not positions.empty:
        # Best-effort -- holdings_value below already falls back to avg_cost per-ticker when a
        # price is missing, so a total fetch failure (e.g. yfinance rate-limited) degrades to
        # showing cost-basis values instead of crashing the page.
        try:
            with st.spinner("Fetching current prices..."):
                recent = get_prices(sorted(positions.index), start=(dt.date.today() - dt.timedelta(days=7)).isoformat())
            if not recent.empty:
                prices_now = recent.ffill().iloc[-1].to_dict()
        except Exception:
            st.caption("⚠️ Live prices unavailable right now (rate-limited) -- showing cost-basis values.")

    holdings_value = sum(qty * prices_now.get(t, row["avg_cost"]) for t, row in positions.iterrows() for qty in [row["shares"]])
    total_value = cash + holdings_value
    total_return = total_value / meta["initial_cash"] - 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cash", f"${cash:,.2f}")
    m2.metric("Holdings value", f"${holdings_value:,.2f}")
    m3.metric("Total value", f"${total_value:,.2f}")
    m4.metric("Return since inception", f"{total_return:+.2%}")

    trades = load_trades(selected)
    if not trades.empty and st.button("Undo last trade", key="portfolio_undo"):
        undo_last_trade(selected)
        st.rerun()

    st.divider()
    st.subheader("Current positions")
    if positions.empty:
        st.caption("No open positions.")
    else:
        display = positions.copy()
        display["current_price"] = [prices_now.get(t, display.loc[t, "avg_cost"]) for t in display.index]
        display["market_value"] = display["shares"] * display["current_price"]
        display["unrealized_pl"] = (display["current_price"] - display["avg_cost"]) * display["shares"]
        display["unrealized_pl_pct"] = display["current_price"] / display["avg_cost"] - 1
        display.index.name = "Ticker"
        st.dataframe(
            display.rename(
                columns={
                    "shares": "Shares",
                    "avg_cost": "Avg cost",
                    "current_price": "Current price",
                    "market_value": "Market value",
                    "unrealized_pl": "Unrealized P/L ($)",
                    "unrealized_pl_pct": "Unrealized P/L (%)",
                }
            ).style.format(
                {
                    "Shares": "{:g}",
                    "Avg cost": "${:,.2f}",
                    "Current price": "${:,.2f}",
                    "Market value": "${:,.2f}",
                    "Unrealized P/L ($)": "${:+,.2f}",
                    "Unrealized P/L (%)": "{:+.2%}",
                }
            ),
            width="stretch",
            key="table_portfolio_positions",
        )

    st.subheader("Value over time")
    history = valuation_history(selected)
    if len(history) > 1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=history["date"], y=history["total_value"], name="Total value", line=dict(width=2))
        )
        fig.add_hline(y=meta["initial_cash"], line=dict(dash="dash", color="#9a9da4"), opacity=0.6)
        fig.update_layout(
            yaxis_title="Portfolio value ($)",
            xaxis_title="Date",
            hovermode="x unified",
            height=400,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch", key="chart_portfolio_value")

    st.subheader("Trade history")
    if trades.empty:
        st.caption("No trades logged yet.")
    else:
        trades_display = trades.sort_values("date", ascending=False).copy()
        trades_display["date"] = trades_display["date"].dt.date
        st.dataframe(
            trades_display.rename(
                columns={
                    "date": "Date",
                    "ticker": "Ticker",
                    "action": "Action",
                    "shares": "Shares",
                    "price": "Price",
                    "note": "Note",
                }
            ).style.format({"Shares": "{:g}", "Price": "${:,.2f}"}),
            width="stretch",
            key="table_portfolio_trades",
        )


def render_simulation_tab() -> None:
    st.caption(
        "Automated, rule-driven trading and historical backtesting. Shares the same underlying "
        "storage as the Portfolio tab -- a simulation created here shows up there too, and vice versa."
    )

    existing = list_portfolios()
    name_col, create_col, delete_col = st.columns([2, 1, 1])
    with name_col:
        selected = st.selectbox(
            "Simulation", options=existing, index=None, placeholder="Choose a simulation",
            label_visibility="collapsed", key="simulation_selected",
        )
    with create_col:
        with st.popover("+ New simulation"):
            new_name = st.text_input("Name", key="simulation_new_name", placeholder="e.g. momentum_top5")
            new_cash = st.number_input(
                "Starting cash ($)", min_value=100.0, value=10000.0, step=500.0, key="simulation_new_cash"
            )
            if st.button("Create", key="simulation_new_create"):
                try:
                    create_portfolio(new_name, new_cash)
                    st.session_state["simulation_selected"] = new_name.strip()
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    with delete_col:
        if selected:
            with st.popover("Delete simulation"):
                st.write(f"Permanently delete **{selected}** and its trade history?")
                if st.button("Confirm delete", key="simulation_delete_confirm"):
                    delete_portfolio(selected)
                    del st.session_state["simulation_selected"]
                    st.rerun()

    if not selected:
        st.info("Create a simulation to get started." if not existing else "Choose a simulation above.")
        return

    meta = load_meta(selected)

    st.subheader("Automated rules")
    st.caption(
        "Runs immediately, once, against today's data. Exits first (positions past their hold-days "
        "limit, or -- for rules where that applies -- whose signal is gone), then entries: candidates "
        "from every *enabled* rule are ranked by percentile within their own rule and filled from one "
        "shared cash pool, fixed $ per position, until either the ranked list or the cash runs out. "
        "Re-running with the same signal does nothing new -- a ticker already held under a rule is "
        "skipped from that rule's entries."
    )
    saved_rules = meta.get("rules", {})
    universe_for_rules = sorted(set(TICKER_TO_NAME))
    active_rules: dict[str, tuple] = {}

    diverged_defaults = saved_rules.get("diverged_pairs", {})
    with st.expander("Diverged pairs (correlation >= 0.8, |Z| >= 1.5, buy the laggard)", expanded=True):
        dp_enabled = st.checkbox("Enable", value=False, key="portfolio_dp_enabled")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            dp_top_n = st.number_input(
                "Max concurrent positions", min_value=1, value=int(diverged_defaults.get("top_n", 3)),
                step=1, key="portfolio_dp_top_n",
            )
        with rc2:
            dp_hold_days = st.number_input(
                "Force-exit after (days)", min_value=1, value=int(diverged_defaults.get("hold_days", 15)),
                step=1, key="portfolio_dp_hold_days",
            )
        with rc3:
            dp_unit_size = st.number_input(
                "$ per position", min_value=10.0, value=float(diverged_defaults.get("unit_size", 500.0)),
                step=50.0, key="portfolio_dp_unit_size",
            )
        if dp_enabled:
            settings = {"top_n": dp_top_n, "hold_days": dp_hold_days, "unit_size": dp_unit_size}
            active_rules["diverged_pairs"] = (
                (lambda as_of: diverged_pairs_candidates(as_of, universe_for_rules)),
                RuleConfig(name="diverged_pairs", **settings),
            )

    dip_defaults = saved_rules.get("buy_the_dip", {})
    with st.expander("Buy the dip (fell 5%+ in a day, sell after N days)", expanded=True):
        dip_enabled = st.checkbox("Enable", value=False, key="portfolio_dip_enabled")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            dip_drop_pct = st.number_input(
                "Drop threshold (%)", min_value=1.0, value=float(dip_defaults.get("drop_pct", 5.0)),
                step=0.5, key="portfolio_dip_drop_pct",
            )
        with rc2:
            dip_top_n = st.number_input(
                "Max concurrent positions", min_value=1, value=int(dip_defaults.get("top_n", 5)),
                step=1, key="portfolio_dip_top_n",
            )
        with rc3:
            dip_hold_days = st.number_input(
                "Sell after (days)", min_value=1, value=int(dip_defaults.get("hold_days", 7)),
                step=1, key="portfolio_dip_hold_days",
            )
        with rc4:
            dip_unit_size = st.number_input(
                "$ per position", min_value=10.0, value=float(dip_defaults.get("unit_size", 500.0)),
                step=50.0, key="portfolio_dip_unit_size",
            )
        if dip_enabled:
            settings = {
                "drop_pct": dip_drop_pct, "top_n": dip_top_n, "hold_days": dip_hold_days,
                "unit_size": dip_unit_size,
            }
            active_rules["buy_the_dip"] = (
                (lambda as_of: dip_candidates(as_of, universe_for_rules, drop_pct=dip_drop_pct / 100)),
                RuleConfig(
                    name="buy_the_dip", top_n=dip_top_n, hold_days=dip_hold_days, unit_size=dip_unit_size,
                    exit_on_signal_loss=False,  # a daily-drop event, not an ongoing condition to keep re-checking
                ),
            )

    streak_defaults = saved_rules.get("earnings_streak", {})
    with st.expander("Earnings-beat streak, mild (last 2 quarters surprised 0-2%, buy day before next report)", expanded=True):
        streak_enabled = st.checkbox("Enable", value=False, key="portfolio_streak_enabled")
        st.caption(
            "Only ever qualifies on the single trading day before a ticker's next earnings report -- "
            "run this rule that day (or later that same day, before the report) to catch it."
        )
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1:
            streak_low = st.number_input(
                "Min prior surprise (%)", value=float(streak_defaults.get("low_pct", 0.0)),
                step=0.5, key="portfolio_streak_low",
            )
        with rc2:
            streak_high = st.number_input(
                "Max prior surprise (%)", value=float(streak_defaults.get("high_pct", 2.0)),
                step=0.5, key="portfolio_streak_high",
            )
        with rc3:
            streak_top_n = st.number_input(
                "Max concurrent positions", min_value=1, value=int(streak_defaults.get("top_n", 5)),
                step=1, key="portfolio_streak_top_n",
            )
        with rc4:
            streak_hold_days = st.number_input(
                "Sell after (days)", min_value=1, value=int(streak_defaults.get("hold_days", 14)),
                step=1, key="portfolio_streak_hold_days",
            )
        with rc5:
            streak_unit_size = st.number_input(
                "$ per position", min_value=10.0, value=float(streak_defaults.get("unit_size", 500.0)),
                step=50.0, key="portfolio_streak_unit_size",
            )
        if streak_enabled:
            active_rules["earnings_streak"] = (
                (lambda as_of: earnings_streak_candidates(as_of, universe_for_rules, streak_low, streak_high)),
                RuleConfig(
                    name="earnings_streak", top_n=streak_top_n, hold_days=streak_hold_days,
                    unit_size=streak_unit_size,
                    exit_on_signal_loss=False,  # a one-time pre-earnings entry, not an ongoing condition
                ),
            )

    analyst_defaults = saved_rules.get("analyst_momentum", {})
    with st.expander("Analyst momentum (last 7 days: up+maintain >= down, avg target >= 20% above price)", expanded=True):
        analyst_enabled = st.checkbox("Enable", value=False, key="portfolio_analyst_enabled")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            analyst_min_upside = st.number_input(
                "Min target upside (%)", value=float(analyst_defaults.get("min_upside_pct", 20.0)),
                step=1.0, key="portfolio_analyst_min_upside",
            )
        with rc2:
            analyst_top_n = st.number_input(
                "Max concurrent positions", min_value=1, value=int(analyst_defaults.get("top_n", 5)),
                step=1, key="portfolio_analyst_top_n",
            )
        with rc3:
            analyst_hold_days = st.number_input(
                "Sell after (days)", min_value=1, value=int(analyst_defaults.get("hold_days", 14)),
                step=1, key="portfolio_analyst_hold_days",
            )
        with rc4:
            analyst_unit_size = st.number_input(
                "$ per position", min_value=10.0, value=float(analyst_defaults.get("unit_size", 500.0)),
                step=50.0, key="portfolio_analyst_unit_size",
            )
        if analyst_enabled:
            active_rules["analyst_momentum"] = (
                (lambda as_of: analyst_momentum_candidates(as_of, universe_for_rules, min_upside_pct=analyst_min_upside)),
                RuleConfig(
                    name="analyst_momentum", top_n=analyst_top_n, hold_days=analyst_hold_days,
                    unit_size=analyst_unit_size,
                    exit_on_signal_loss=False,  # fixed hold-then-sell, per the rule as specified
                ),
            )

    if st.button("Run enabled rules now", key="portfolio_run_rules", disabled=not active_rules):
        if "diverged_pairs" in active_rules:
            save_rule_settings(
                selected, "diverged_pairs",
                {"top_n": dp_top_n, "hold_days": dp_hold_days, "unit_size": dp_unit_size},
            )
        if "buy_the_dip" in active_rules:
            save_rule_settings(
                selected, "buy_the_dip",
                {"drop_pct": dip_drop_pct, "top_n": dip_top_n, "hold_days": dip_hold_days, "unit_size": dip_unit_size},
            )
        if "earnings_streak" in active_rules:
            save_rule_settings(
                selected, "earnings_streak",
                {
                    "low_pct": streak_low, "high_pct": streak_high, "top_n": streak_top_n,
                    "hold_days": streak_hold_days, "unit_size": streak_unit_size,
                },
            )
        if "analyst_momentum" in active_rules:
            save_rule_settings(
                selected, "analyst_momentum",
                {
                    "min_upside_pct": analyst_min_upside, "top_n": analyst_top_n,
                    "hold_days": analyst_hold_days, "unit_size": analyst_unit_size,
                },
            )
        with st.spinner("Evaluating rules and running trades..."):
            trade_log = run_rules(selected, active_rules)
        st.session_state["portfolio_last_rule_log"] = trade_log
        st.rerun()
    if not active_rules:
        st.caption("Enable at least one rule above to run it.")

    last_log = st.session_state.pop("portfolio_last_rule_log", None)
    if last_log is not None:
        if last_log:
            st.success(f"Rule run: {len(last_log)} trade(s) executed (see Trade history below).")
        else:
            st.info("Rule run: no trades needed -- nothing new qualified, and no exits were due.")

    st.divider()
    st.subheader("Backtest (simulate in the past)")
    st.caption(
        "Runs whichever rules are enabled above, once per business day, over a chosen historical "
        "window -- in a separate, temporary simulation, so it never touches this portfolio's real "
        "trade log. Can take a while for a long window (one set of network calls per rule per day, "
        "cached after the first run)."
    )
    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        backtest_start = st.date_input(
            "Start", value=dt.date.today() - dt.timedelta(days=30), max_value=dt.date.today(),
            key="portfolio_bt_start",
        )
    with bt2:
        backtest_end = st.date_input(
            "End", value=dt.date.today(), max_value=dt.date.today(), key="portfolio_bt_end"
        )
    with bt3:
        backtest_cash = st.number_input(
            "Starting cash ($)", min_value=100.0, value=10000.0, step=500.0, key="portfolio_bt_cash"
        )

    if st.button(
        "Run backtest", key="portfolio_run_backtest",
        disabled=not active_rules or backtest_start > backtest_end,
    ):
        scratch_name = "_backtest_scratch"
        if scratch_name in list_portfolios():
            delete_portfolio(scratch_name)
        create_portfolio(scratch_name, backtest_cash, created=backtest_start)
        business_days = pd.bdate_range(backtest_start, backtest_end)
        progress = st.progress(0.0, text="Running backtest...")
        try:
            for i, d in enumerate(business_days):
                run_rules(scratch_name, active_rules, as_of=d.date())
                progress.progress((i + 1) / len(business_days), text=f"Running backtest... {d.date()}")

            trades_bt = closed_trades(scratch_name)
            cash_bt, positions_bt = current_state(scratch_name)

            end_ts = pd.Timestamp(backtest_end)
            total_value = cash_bt
            if not positions_bt.empty:
                end_prices = get_prices(
                    sorted(positions_bt.index), start=(backtest_end - dt.timedelta(days=10)).isoformat(),
                    end=(backtest_end + dt.timedelta(days=1)).isoformat(),
                )
                end_prices = end_prices.loc[end_prices.index <= end_ts]
                for t, row in positions_bt.iterrows():
                    mark_price = (
                        end_prices[t].dropna().iloc[-1] if t in end_prices.columns and not end_prices[t].dropna().empty
                        else row["avg_cost"]
                    )
                    total_value += row["shares"] * mark_price
                    open_mask = (trades_bt["ticker"] == t) & (trades_bt["status"] == "open")
                    trades_bt.loc[open_mask, "sell_price"] = mark_price
                    trades_bt.loc[open_mask, "return_pct"] = mark_price / trades_bt.loc[open_mask, "buy_price"] - 1
                    trades_bt.loc[open_mask, "pnl"] = (
                        (mark_price - trades_bt.loc[open_mask, "buy_price"]) * trades_bt.loc[open_mask, "shares"]
                    )

            spy_prices = get_prices(
                [SP500_BENCHMARK], start=backtest_start.isoformat(),
                end=(backtest_end + dt.timedelta(days=1)).isoformat(),
            )[SP500_BENCHMARK].dropna()
            spy_return = spy_prices.iloc[-1] / spy_prices.iloc[0] - 1 if len(spy_prices) > 1 else float("nan")

            st.session_state["portfolio_backtest_result"] = {
                "trades": trades_bt,
                "total_return": total_value / backtest_cash - 1,
                "spy_return": spy_return,
                "start": backtest_start,
                "end": backtest_end,
            }
        finally:
            progress.empty()
            if scratch_name in list_portfolios():
                delete_portfolio(scratch_name)
        st.rerun()

    backtest_result = st.session_state.get("portfolio_backtest_result")
    if backtest_result:
        trades_bt = backtest_result["trades"]
        st.caption(f"Backtest: {backtest_result['start']} to {backtest_result['end']}, {len(trades_bt)} trade(s).")
        bm1, bm2, bm3 = st.columns(3)
        bm1.metric("Total return", f"{backtest_result['total_return']:+.2%}")
        bm2.metric(f"{SP500_BENCHMARK} return (same period)", f"{backtest_result['spy_return']:+.2%}")
        bm3.metric("Excess vs S&P 500", f"{backtest_result['total_return'] - backtest_result['spy_return']:+.2%}")

        if trades_bt.empty:
            st.info("No trades were made in this window.")
        else:
            closed_bt = trades_bt[trades_bt["status"] == "closed"]
            st.markdown("**Return by rule**")
            if closed_bt.empty:
                st.caption("No trades closed within the window yet (everything bought is still open).")
            else:
                by_rule = closed_bt.groupby("rule").agg(
                    Trades=("return_pct", "count"),
                    **{"Avg return": ("return_pct", "mean"), "Total P/L": ("pnl", "sum")},
                )
                st.dataframe(
                    by_rule.style.format({"Avg return": "{:+.2%}", "Total P/L": "${:+,.2f}"}),
                    width="stretch", key="table_backtest_by_rule",
                )

            st.markdown("**All trades**")
            display_bt = trades_bt.copy()
            display_bt["buy_date"] = display_bt["buy_date"].dt.date
            display_bt["sell_date"] = display_bt["sell_date"].dt.date
            display_bt = display_bt.rename(
                columns={
                    "rule": "Rule", "ticker": "Ticker", "buy_date": "Buy date", "buy_price": "Buy price",
                    "shares": "Shares", "sell_date": "Sell date", "sell_price": "Sell price",
                    "return_pct": "Return", "pnl": "P/L ($)", "hold_days": "Held (days)", "status": "Status",
                }
            )
            st.dataframe(
                display_bt.sort_values("Buy date", ascending=False).style.format(
                    {
                        "Buy price": "${:,.2f}", "Shares": "{:g}", "Sell price": "${:,.2f}",
                        "Return": "{:+.2%}", "P/L ($)": "${:+,.2f}", "Held (days)": "{:g}",
                    },
                    na_rep="--",
                ),
                width="stretch", key="table_backtest_trades", hide_index=True,
            )



def render_correlations_tab() -> None:
    st.caption(
        "Across every tracked ticker: which pairs usually move together (return correlation "
        "over the lookback window), and among those, which have drifted apart from their usual "
        "relationship right now."
    )
    col1, col2 = st.columns(2)
    with col1:
        lookback_months = st.slider("Lookback (months)", min_value=1, max_value=24, value=6, key="corr_lookback")
    with col2:
        min_correlation = st.slider(
            "Minimum correlation to count as 'usually moves together'",
            min_value=0.0,
            max_value=0.95,
            value=0.75,
            step=0.05,
            key="corr_min",
        )

    universe = sorted(TICKER_TO_NAME)
    lookback_start = dt.date.today() - dt.timedelta(days=int(lookback_months * 30.44) + 5)
    with st.spinner("Fetching prices for correlation analysis..."):
        corr_prices = get_prices(universe + [SP500_BENCHMARK], start=lookback_start.isoformat())
    corr_prices = corr_prices.dropna(axis=1, how="all")

    daily_returns = corr_prices[[t for t in universe if t in corr_prices.columns]].pct_change(
        fill_method=None
    ).dropna(how="all")
    pairs = pairwise_correlation(daily_returns)
    if pairs.empty:
        st.info("Not enough overlapping data to compute correlations yet.")
        return

    def label(ticker: str) -> str:
        return f"{TICKER_TO_NAME.get(ticker, ticker)} ({ticker})"

    corr_matrix = daily_returns.corr()
    display_labels = [label(t) for t in corr_matrix.columns]
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=display_labels,
            y=display_labels,
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING_COLORSCALE,
            colorbar=dict(title="corr"),
            hovertemplate="%{y}<br>%{x}<br>correlation: %{z:.2f}<extra></extra>",
        )
    )
    heatmap.update_layout(height=650, margin=dict(t=20, b=20))
    st.plotly_chart(heatmap, width="stretch", key="chart_corr_heatmap")

    st.subheader("Most correlated pairs")
    top_correlated = pairs.sort_values("correlation", ascending=False).head(15).copy()
    top_correlated["A"] = top_correlated["ticker_a"].map(label)
    top_correlated["B"] = top_correlated["ticker_b"].map(label)
    st.dataframe(
        top_correlated[["A", "B", "correlation"]]
        .rename(columns={"correlation": "Correlation"})
        .style.format({"Correlation": "{:.2f}"}),
        width=600,
        key="table_top_correlated",
        hide_index=True,
    )

    st.subheader("Beta vs S&P 500")
    st.latex(r"\beta = \dfrac{\text{Cov}(R_{\text{stock}},\, R_{\text{market}})}{\text{Var}(R_{\text{market}})}")
    st.caption(
        "How much a stock tends to move for a given move in the S&P 500, over the same lookback "
        "window as above. β = 1 moves in line with the market; β > 1 amplifies market moves (more "
        "volatile than the index); β < 1 dampens them; β < 0 tends to move opposite the market."
    )
    if SP500_BENCHMARK in corr_prices.columns:
        market_returns = corr_prices[SP500_BENCHMARK].pct_change(fill_method=None)
        beta_rows = []
        for t in universe:
            if t not in corr_prices.columns:
                continue
            paired = pd.concat(
                [corr_prices[t].pct_change(fill_method=None), market_returns], axis=1, keys=["stock", "market"]
            ).dropna()
            if len(paired) < 10 or paired["market"].var() == 0:
                continue
            beta = paired["stock"].cov(paired["market"]) / paired["market"].var()
            beta_rows.append({"Ticker": label(t), "Beta": beta})
        if beta_rows:
            beta_df = pd.DataFrame(beta_rows).sort_values("Beta", ascending=False)
            st.dataframe(
                beta_df.style.format({"Beta": "{:.2f}"}), width=450, key="table_beta", hide_index=True
            )
        else:
            st.info("Not enough data to compute beta.")
    else:
        st.info("S&P 500 data unavailable for beta.")

    st.subheader("Most diverged right now (among correlated pairs)")
    st.latex(
        r"\text{spread}_t = \text{cumret}_A(t) - \text{cumret}_B(t)"
        r"\qquad\qquad"
        r"z = \dfrac{\text{spread}_{\text{today}} - \overline{\text{spread}}}{\sigma_{\text{spread}}}"
    )
    st.caption(
        "Spread = the gap between A's and B's cumulative return since the lookback start. "
        "Z-score = how many standard deviations today's spread sits from its own average over "
        "the window (not from zero) — it measures whether *this pair* looks unusual *for this pair*, "
        "not whether either stock's move is unusual on its own. "
        "Z strongly **positive** (A outperformed B by more than usual) → the reversion bet is "
        "*short A / long B*. Z strongly **negative** (A underperformed B by more "
        "than usual) → the reversion bet is *long A / short B*."
    )
    cumulative_return = corr_prices / corr_prices.bfill().iloc[0] - 1
    candidates = pairs[pairs["correlation"] >= min_correlation]
    diverged = divergence_now(cumulative_return, candidates)
    if diverged.empty:
        st.info("No pairs meet the minimum correlation threshold.")
        return

    diverged = diverged.reindex(diverged["z_score"].abs().sort_values(ascending=False).index).head(15).copy()
    diverged["A"] = diverged["ticker_a"].map(label)
    diverged["B"] = diverged["ticker_b"].map(label)
    st.dataframe(
        diverged[["A", "B", "correlation", "mean_spread", "current_spread", "z_score"]].rename(
            columns={
                "correlation": "Correlation",
                "mean_spread": "Mean spread",
                "current_spread": "Current spread",
                "z_score": "Z-score",
            }
        ).style.format(
            {"Correlation": "{:.2f}", "Mean spread": "{:+.2%}", "Current spread": "{:+.2%}", "Z-score": "{:+.2f}"}
        ),
        width=750,
        key="table_top_diverged",
        hide_index=True,
    )


def render_momentum_tab() -> None:
    hypothesis_placeholder = st.empty()
    universe_tickers = sorted(STOCK_TICKER_TO_NAME)
    col0, col1, col2 = st.columns(3)
    with col0:
        momentum_start_date = st.date_input(
            "Since (D)", value=dt.date(2026, 6, 1), max_value=dt.date.today(), key="mom_since"
        )
    with col1:
        rebalance_weeks = st.number_input(
            "Rebalance every T weeks", min_value=1, max_value=26, value=1, step=1, key="mom_T"
        )
    with col2:
        lookback_weeks = st.number_input(
            "Lookback H weeks", min_value=1, max_value=104, value=4, step=1, key="mom_H"
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        top_n = st.number_input(
            "Number of stocks (N)",
            min_value=1,
            max_value=len(universe_tickers),
            value=min(2, len(universe_tickers)),
            step=1,
            key="mom_N",
        )
    with col4:
        cost_bps = st.number_input("Cost (bps, round-trip)", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="mom_cost")
    with col5:
        worst_performers = st.toggle("Worst performers", value=False, key="mom_direction")

    direction: Direction = "worst" if worst_performers else "best"
    hypothesis_placeholder.caption(
        f"From date D, every T weeks, buy the N stocks with the {direction} trailing H-weeks gains, "
        "equal-weighted. Does that beat just holding the S&P 500?"
    )

    rebalance_days = int(rebalance_weeks) * 7
    lookback_days = lookback_weeks * 7
    momentum_start = pd.Timestamp(momentum_start_date)
    buffer_start = momentum_start - dt.timedelta(days=lookback_days + 5)
    with st.spinner("Running backtest..."):
        prices = get_prices(universe_tickers + [SP500_BENCHMARK], start=buffer_start.date().isoformat())
        stock_prices = prices[universe_tickers].dropna(axis=1, how="all")

        weight_func = top_n_momentum_weight_func(lookback_days=lookback_days, n=int(top_n), direction=direction)
        strategy_equity = run_backtest(
            stock_prices, weight_func, freq=rebalance_days, start_date=momentum_start, cost_bps=cost_bps
        )
        equal_weight_func = equal_weight_universe_weight_func(lookback_days=lookback_days)
        equal_weight_equity = run_backtest(
            stock_prices, equal_weight_func, freq=rebalance_days, start_date=momentum_start, cost_bps=cost_bps
        )
        random_equity = random_n_average_equity(
            stock_prices,
            lookback_days=lookback_days,
            n=int(top_n),
            freq=rebalance_days,
            start_date=momentum_start,
            trials=1000,
            cost_bps=cost_bps,
        )
        benchmark_prices = prices[SP500_BENCHMARK].loc[prices.index >= momentum_start]
        benchmark_equity = buy_and_hold(benchmark_prices)

    strategy_label = f"{'Worst' if direction == 'worst' else 'Top'}-{int(top_n)} momentum"
    equal_weight_label = "Equal-weight universe"
    random_label = f"Random-{int(top_n)} control (avg of 1000)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=strategy_equity.index, y=strategy_equity.values, name=strategy_label, line=dict(width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=equal_weight_equity.index,
            y=equal_weight_equity.values,
            name=equal_weight_label,
            line=dict(width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=random_equity.index,
            y=random_equity.values,
            name=random_label,
            line=dict(width=2, dash="dashdot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.values,
            name="S&P 500 (SPY)",
            line=dict(width=3, color=SP500_LINE_COLOR),
        )
    )
    fig.update_layout(
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        hovermode="x unified",
        legend_title_text="",
        height=450,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch", key="chart_momentum")

    table = summary_table(
        {
            strategy_label: strategy_equity,
            equal_weight_label: equal_weight_equity,
            random_label: random_equity,
            "S&P 500 (SPY)": benchmark_equity,
        }
    )
    st.dataframe(
        table.style.format(
            {
                "total_return": "{:+.2%}",
                "cagr": "{:+.2%}",
                "volatility": "{:.2%}",
                "sharpe": "{:.2f}",
                "max_drawdown": "{:.2%}",
            }
        ),
        width=750,
        key="table_momentum",
    )

    all_dates = rebalance_dates(stock_prices, freq=int(rebalance_days), anchor=momentum_start)
    dates_to_show = all_dates[all_dates >= momentum_start]
    picks = picks_by_rebalance(stock_prices, dates_to_show, lookback_days, int(top_n), direction=direction)

    with st.expander(f"Picks at each rebalance ({len(dates_to_show)} rebalances)", expanded=True):
        if picks.empty:
            st.info("Not enough history yet for a pick.")
        else:
            picks = picks.copy()
            picks["label"] = (
                picks["ticker"].map(lambda t: STOCK_TICKER_TO_NAME.get(t, t))
                + " ("
                + picks["ticker"]
                + ") "
                + picks["trailing_return"].map(lambda r: f"{r:+.1%}")
            )
            wide = picks.pivot(index="date", columns="rank", values="label")
            wide.columns = [f"#{c}" for c in wide.columns]
            wide.index = wide.index.date
            wide = wide.sort_index(ascending=False)
            st.dataframe(wide, width=1100, key="table_momentum_picks")


def render_dip_tab() -> None:
    st.caption(
        "Hypothesis: whenever a stock falls more than T% in a single day, buy it and sell it "
        "D days later. How does that compare to just holding the S&P 500 over the same window?"
    )
    universe_tickers = sorted(STOCK_TICKER_TO_NAME)
    col0, col1, col2, col3 = st.columns(4)
    with col0:
        dip_start_date = st.date_input(
            "Since", value=dt.date(2026, 1, 1), max_value=dt.date.today(), key="dip_since"
        )
    with col1:
        drop_pct = st.number_input(
            "Drop more than T%", min_value=1.0, max_value=50.0, value=5.0, step=0.5, key="dip_T"
        )
    with col2:
        hold_days = st.number_input(
            "Sell D days later", min_value=1, max_value=180, value=10, step=1, key="dip_D"
        )
    with col3:
        dip_cost_bps = st.number_input(
            "Cost per trade (round-trip, bps)", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="dip_cost"
        )

    dip_start = pd.Timestamp(dip_start_date)
    # A few days of buffer before "Since" so the very first fetched trading day
    # still has a prior close to compute its own return against -- otherwise a
    # dip that happened to fall on that first day is unmeasurable (NaN return)
    # and silently disappears from the scan.
    fetch_start = dip_start - pd.Timedelta(days=10)
    with st.spinner("Scanning for dips..."):
        prices = get_prices(universe_tickers + [SP500_BENCHMARK], start=fetch_start.date().isoformat())
        stock_prices = prices[universe_tickers].dropna(axis=1, how="all")
        benchmark = prices[SP500_BENCHMARK]

        trades = find_dip_trades(
            stock_prices, benchmark, drop_pct=drop_pct / 100, hold_days=int(hold_days), cost_bps=dip_cost_bps
        )
        trades = trades[trades["buy_date"] >= dip_start]

    if trades.empty:
        pending = find_pending_dips(stock_prices, drop_pct=drop_pct / 100, hold_days=int(hold_days))
        pending = pending[pending["buy_date"] >= dip_start]
        if pending.empty:
            st.info("No dips of that size found in this period.")
        else:
            st.info(
                f"{len(pending)} dip(s) of that size happened in this period, but none have reached "
                f"their {int(hold_days)}-day hold yet -- still in progress, not yet a scoreable trade."
            )
            pending_display = pending.copy()
            pending_display["Ticker"] = pending_display["ticker"].map(lambda t: f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})")
            pending_display["Buy date"] = pending_display["buy_date"].dt.date
            pending_display = pending_display.rename(
                columns={
                    "buy_price": "Buy price", "drop_that_day": "Drop that day",
                    "days_held_so_far": "Days held so far",
                }
            )
            st.dataframe(
                pending_display[["Ticker", "Buy date", "Buy price", "Drop that day", "Days held so far"]]
                .style.format({"Buy price": "${:,.2f}", "Drop that day": "{:+.2%}"}),
                width=800, key="table_dip_pending", hide_index=True,
            )
        return

    compounded_stock = (1 + trades["stock_return"]).prod() - 1
    compounded_benchmark = (1 + trades["benchmark_return"]).prod() - 1

    m1, m2, m3 = st.columns(3)
    m1.metric("Trades", len(trades))
    m2.metric("Avg return per trade", f"{trades['stock_return'].mean():+.2%}")
    m3.metric("Avg excess vs S&P 500", f"{trades['excess_return'].mean():+.2%}")

    st.caption(
        f"If every trade were taken one after another (capital fully rotated, no overlap): "
        f"strategy compounds to **{compounded_stock:+.2%}** vs the S&P 500 compounding to "
        f"**{compounded_benchmark:+.2%}** across those same {len(trades)} windows."
    )

    with st.expander(f"Trades ({len(trades)})", expanded=True):
        display = trades.copy()
        display["Ticker"] = display["ticker"].map(lambda t: f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})")
        display["Buy date"] = display["buy_date"].dt.date
        display["Sell date"] = display["sell_date"].dt.date
        display = display.rename(
            columns={
                "drop_that_day": "Drop that day",
                "stock_return": "Stock return",
                "benchmark_return": "S&P 500 return",
                "excess_return": "Excess return",
            }
        )
        st.dataframe(
            display[
                ["Ticker", "Buy date", "Drop that day", "Sell date", "Stock return", "S&P 500 return", "Excess return"]
            ]
            .sort_values("Buy date", ascending=False)
            .style.format(
                {
                    "Drop that day": "{:+.2%}",
                    "Stock return": "{:+.2%}",
                    "S&P 500 return": "{:+.2%}",
                    "Excess return": "{:+.2%}",
                }
            ),
            width=900,
            key="table_dip_trades",
            hide_index=True,
        )


def render_calendar_tab() -> None:
    st.subheader("Average intraday shape")
    st.caption(
        "Average shape of the trading day: for each hourly bar, the return since that day's open, "
        "averaged across every trading day in the period. Yahoo only serves hourly bars this far "
        "back (finer intervals are limited to roughly the last 60 days), so this is ~7 points per "
        "day connected by a line, not a smooth continuous curve. Pick companies from the sidebar."
    )
    shape_period_months = st.pills(
        "Period",
        options=[3, 6, 12, 24],
        format_func=lambda m: f"{m} months",
        default=12,
        key="intraday_period",
    )
    if not shape_period_months:
        st.info("Pick a period above.")
        return
    shape_period = {3: "3mo", 6: "6mo", 12: "1y", 24: "2y"}[shape_period_months]

    shape_typed = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    shape_tickers = sorted(picked_tickers | shape_typed)
    if not shape_tickers:
        st.info("Pick tickers from the sidebar (or type some in) to see their average intraday shape.")
    else:
        with st.spinner("Fetching intraday bars..."):
            shape_closes = get_intraday_closes(shape_tickers, period=shape_period, interval="60m")

        missing = set(shape_tickers) - set(shape_closes.columns)
        if missing:
            st.warning(f"No intraday data for: {', '.join(sorted(missing))}")

        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
        color_i = 0
        fig3 = go.Figure()
        plotted_any = False
        for t in shape_tickers:
            if t not in shape_closes.columns:
                continue
            series = shape_closes[t].dropna()
            if series.empty:
                continue
            path = average_intraday_path(series)
            n_days = series.groupby(series.index.date).ngroups

            if t == SP500_BENCHMARK:
                color = SP500_LINE_COLOR
            else:
                color = palette[color_i % len(palette)]
                color_i += 1

            fig3.add_trace(
                go.Scatter(
                    x=path.index,
                    y=path.values,
                    mode="lines+markers",
                    name=f"{t} (n={n_days} days)",
                    line=dict(width=2, color=color),
                    marker=dict(size=7),
                )
            )
            plotted_any = True

        if not plotted_any:
            st.error("No intraday data available for this selection/period.")
        else:
            fig3.update_layout(
                yaxis_tickformat="+.2%",
                yaxis_title="Avg return since day's open",
                xaxis_title="Time of day (exchange local)",
                hovermode="x unified",
                legend_title_text="",
                height=450,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig3, width="stretch", key="chart_intraday_shape")

    st.divider()
    st.subheader("Overnight vs Intraday")
    st.caption(
        "Hypothesis: does a stock's return come mostly from overnight (previous close -> today's "
        "open) or intraday (today's open -> today's close)? Averaged across the period below."
    )
    period_label = st.radio("Period", options=["1 year", "2 years"], horizontal=True, key="on_period")
    years = 1 if period_label == "1 year" else 2

    universe_tickers = sorted(STOCK_TICKER_TO_NAME)
    all_tickers = universe_tickers + [SP500_BENCHMARK]
    start = dt.date.today() - dt.timedelta(days=365 * years + 5)

    with st.spinner("Fetching open/close prices..."):
        opens, closes = get_open_close(all_tickers, start=start.isoformat())
    opens = opens.loc[opens.index >= pd.Timestamp(start)]
    closes = closes.loc[closes.index >= pd.Timestamp(start)]

    overnight, intraday = decompose_returns(opens, closes)
    summary = summarize_overnight(overnight, intraday)
    if summary.empty:
        st.info("Not enough data for this period.")
        return

    labels = {t: (SP500_BENCHMARK if t == SP500_BENCHMARK else f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})") for t in summary.index}
    summary = summary.rename(index=labels).sort_values("overnight_return", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary.index, y=summary["overnight_return"], name="Overnight", marker_color=OVERNIGHT_COLOR))
    fig.add_trace(go.Bar(x=summary.index, y=summary["intraday_return"], name="Intraday", marker_color=INTRADAY_COLOR))
    fig.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title=f"Compounded return over {period_label}",
        hovermode="x unified",
        legend_title_text="",
        height=500,
        margin=dict(t=20, b=20),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, width="stretch", key="chart_overnight_bars")

    st.divider()
    st.subheader("Day of week")
    st.caption(
        "Average daily (close-to-close) return by weekday, over the same period as above, for S&P "
        "500 plus whichever tickers are picked in the sidebar."
    )
    dow_typed = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    dow_tickers = [t for t in sorted(picked_tickers | dow_typed | {SP500_BENCHMARK}) if t in closes.columns]

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_returns = closes[dow_tickers].pct_change(fill_method=None)
    dow_returns["_weekday"] = closes.index.day_name()
    avg_by_weekday = dow_returns.groupby("_weekday")[dow_tickers].mean().reindex(weekday_order)

    dow_palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    fig_dow = go.Figure()
    color_i = 0
    for t in dow_tickers:
        color = SP500_LINE_COLOR if t == SP500_BENCHMARK else dow_palette[color_i % len(dow_palette)]
        if t != SP500_BENCHMARK:
            color_i += 1
        fig_dow.add_trace(go.Bar(x=avg_by_weekday.index, y=avg_by_weekday[t], name=t, marker_color=color))
    fig_dow.update_layout(
        barmode="group",
        yaxis_tickformat="+.3%",
        yaxis_title="Avg daily return",
        hovermode="x unified",
        legend_title_text="",
        height=420,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_dow, width="stretch", key="chart_day_of_week")


EARNINGS_LOOKBACK_QUARTERS = 12


def render_pead_tab() -> None:
    hypothesis_placeholder = st.empty()
    universe_tickers = sorted(STOCK_TICKER_TO_NAME)

    with st.spinner("Fetching earnings history (one request per stock, cached after first run)..."):
        earnings_by_ticker = {t: get_earnings_history(t) for t in universe_tickers}

    col0, col1, col2, col3 = st.columns(4)
    with col0:
        pead_beat = st.toggle(
            "Beat estimate", value=True, key="pead_direction", help="On = beat the EPS estimate, off = miss it"
        )
    with col1:
        surprise_threshold = st.number_input(
            "Surprise magnitude X%", min_value=0.0, max_value=200.0, value=5.0, step=1.0, key="pead_X"
        )
    with col2:
        hold_days = st.number_input("Hold T days", min_value=1, max_value=90, value=10, step=1, key="pead_T")
    with col3:
        pead_since = st.date_input(
            "Since", value=dt.date(2026, 1, 1), max_value=dt.date.today(), key="pead_since",
            help=f"Only trades whose buy date is on/after this. Earnings history itself still only "
                 f"covers the last {EARNINGS_LOOKBACK_QUARTERS} reported quarters.",
        )

    pead_direction: PeadDirection = "beat" if pead_beat else "miss"
    verb = "beat" if pead_direction == "beat" else "missed"
    sign = "+" if pead_direction == "beat" else "-"
    hypothesis_placeholder.caption(
        f"Hypothesis (classic PEAD): whenever a stock's earnings {verb} its EPS estimate by more than "
        f"X% (surprise {sign}X%), buy it the next trading day and sell it T days later. How does that "
        "compare to the S&P 500 over the same window? Surprise% = (Reported EPS − EPS Estimate) / "
        "|EPS Estimate| × 100 — taken directly from the data provider (computed from full-precision "
        "estimate/actual EPS, not the rounded figures shown in the trade table)."
    )

    all_earnings_dates = pd.concat(
        [e["earnings_date"] for e in earnings_by_ticker.values() if not e.empty], ignore_index=True
    )
    if all_earnings_dates.empty:
        st.info("No earnings history available for this universe.")
        return
    price_start = (all_earnings_dates.min() - pd.Timedelta(days=5)).date().isoformat()

    with st.spinner("Fetching prices..."):
        prices = get_prices(universe_tickers + [SP500_BENCHMARK], start=price_start)

    all_trades = []
    for t in universe_tickers:
        earnings = earnings_by_ticker[t]
        if earnings.empty or t not in prices.columns:
            continue
        trades = find_pead_trades(
            t,
            prices[t],
            prices[SP500_BENCHMARK],
            earnings,
            surprise_threshold_pct=surprise_threshold,
            hold_days=int(hold_days),
            direction=pead_direction,
        )
        if not trades.empty:
            all_trades.append(trades)

    if not all_trades:
        st.info("No qualifying earnings beats found for these settings.")
        return

    trades = pd.concat(all_trades, ignore_index=True).sort_values("buy_date")
    trades = trades[trades["buy_date"] >= pd.Timestamp(pead_since)]
    if trades.empty:
        st.info(f"No qualifying earnings beats found on/after {pead_since} for these settings.")
        return

    compounded_stock = (1 + trades["stock_return"]).prod() - 1
    compounded_benchmark = (1 + trades["benchmark_return"]).prod() - 1

    m1, m2, m3 = st.columns(3)
    m1.metric("Trades", len(trades))
    m2.metric("Avg return per trade", f"{trades['stock_return'].mean():+.2%}")
    m3.metric("Avg excess vs S&P 500", f"{trades['excess_return'].mean():+.2%}")

    st.caption(
        f"If every trade were taken one after another (capital fully rotated, no overlap): "
        f"strategy compounds to **{compounded_stock:+.2%}** vs the S&P 500 compounding to "
        f"**{compounded_benchmark:+.2%}** across those same {len(trades)} windows."
    )

    with st.expander(f"Trades ({len(trades)})", expanded=True):
        display = trades.copy()
        display["Ticker"] = display["ticker"].map(lambda t: f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})")
        display["Earnings date"] = display["earnings_date"].dt.date
        display["Buy date"] = display["buy_date"].dt.date
        display["Sell date"] = display["sell_date"].dt.date
        display = display.rename(
            columns={
                "surprise_pct": "EPS surprise",
                "stock_return": "Stock return",
                "benchmark_return": "S&P 500 return",
                "excess_return": "Excess return",
            }
        )
        st.dataframe(
            display[
                ["Ticker", "Earnings date", "EPS surprise", "Buy date", "Sell date", "Stock return", "S&P 500 return", "Excess return"]
            ]
            .sort_values("Buy date", ascending=False)
            .style.format(
                {
                    "EPS surprise": "{:+.2f}%",
                    "Stock return": "{:+.2%}",
                    "S&P 500 return": "{:+.2%}",
                    "Excess return": "{:+.2%}",
                }
            ),
            width=1000,
            key="table_pead_trades",
            hide_index=True,
        )

    st.divider()
    st.subheader("Alternative hypothesis: earnings-beat streak")
    st.caption(
        "Anticipatory, not reactive: if a stock's last **two** reported quarters both beat estimates "
        "by more than Y%, buy it at the close the day before its *next* earnings report, and sell it "
        "T2 days after that report -- a bet the streak continues into the next print, taken before "
        "that print happens (not a reaction to how it actually goes)."
    )
    streak_col0, streak_col1 = st.columns(2)
    with streak_col0:
        streak_threshold = st.number_input(
            "Prior surprise streak Y%", min_value=0.0, max_value=200.0, value=2.0, step=0.5, key="pead_streak_Y"
        )
    with streak_col1:
        streak_hold_days = st.number_input(
            "Hold T2 days (after the next report)", min_value=1, max_value=90, value=10, step=1, key="pead_streak_T"
        )

    streak_trades_list = []
    for t in universe_tickers:
        earnings = earnings_by_ticker[t]
        if earnings.empty or t not in prices.columns:
            continue
        trades_t = find_earnings_streak_trades(
            t, prices[t], prices[SP500_BENCHMARK], earnings,
            streak_threshold_pct=streak_threshold, hold_days=int(streak_hold_days),
        )
        if not trades_t.empty:
            streak_trades_list.append(trades_t)

    if not streak_trades_list:
        st.info("No qualifying two-quarter beat streaks found for these settings.")
    else:
        streak_trades = pd.concat(streak_trades_list, ignore_index=True).sort_values("buy_date")
        streak_trades = streak_trades[streak_trades["buy_date"] >= pd.Timestamp(pead_since)]
        if streak_trades.empty:
            st.info(f"No qualifying streak trades on/after {pead_since} for these settings.")
        else:
            compounded_streak_stock = (1 + streak_trades["stock_return"]).prod() - 1
            compounded_streak_bench = (1 + streak_trades["benchmark_return"]).prod() - 1

            s1, s2, s3 = st.columns(3)
            s1.metric("Trades", len(streak_trades))
            s2.metric("Avg return per trade", f"{streak_trades['stock_return'].mean():+.2%}")
            s3.metric("Avg excess vs S&P 500", f"{streak_trades['excess_return'].mean():+.2%}")
            st.caption(
                f"If every trade were taken one after another (capital fully rotated, no overlap): "
                f"strategy compounds to **{compounded_streak_stock:+.2%}** vs the S&P 500 compounding to "
                f"**{compounded_streak_bench:+.2%}** across those same {len(streak_trades)} windows."
            )

            with st.expander(f"Trades ({len(streak_trades)})", expanded=True):
                display = streak_trades.copy()
                display["Ticker"] = display["ticker"].map(lambda tk: f"{STOCK_TICKER_TO_NAME.get(tk, tk)} ({tk})")
                display["Earnings date"] = display["earnings_date"].dt.date
                display["Buy date"] = display["buy_date"].dt.date
                display["Sell date"] = display["sell_date"].dt.date
                display = display.rename(
                    columns={
                        "prior_surprise_1": "Prior surprise (Q-1)",
                        "prior_surprise_2": "Prior surprise (Q-2)",
                        "stock_return": "Stock return",
                        "benchmark_return": "S&P 500 return",
                        "excess_return": "Excess return",
                    }
                )
                st.dataframe(
                    display[
                        [
                            "Ticker", "Earnings date", "Prior surprise (Q-2)", "Prior surprise (Q-1)",
                            "Buy date", "Sell date", "Stock return", "S&P 500 return", "Excess return",
                        ]
                    ]
                    .sort_values("Buy date", ascending=False)
                    .style.format(
                        {
                            "Prior surprise (Q-2)": "{:+.2f}%",
                            "Prior surprise (Q-1)": "{:+.2f}%",
                            "Stock return": "{:+.2%}",
                            "S&P 500 return": "{:+.2%}",
                            "Excess return": "{:+.2%}",
                        }
                    ),
                    width=1000,
                    key="table_pead_streak_trades",
                    hide_index=True,
                )

    st.divider()
    st.subheader("Historical surprise — sidebar picks")
    st.caption(
        f"Last {EARNINGS_LOOKBACK_QUARTERS} reported quarters of EPS surprise for whichever tickers are "
        "picked in the sidebar (any category, or typed in). 0% = met estimate exactly."
    )
    sidebar_typed = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    sidebar_tickers = sorted(picked_tickers | sidebar_typed)
    if not sidebar_tickers:
        st.info("Pick tickers from the sidebar (or type some in) to see their surprise history.")
    else:
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
        fig_surprise = go.Figure()
        plotted = False
        for i, t in enumerate(sidebar_tickers):
            hist = earnings_by_ticker.get(t)
            if hist is None:  # a sidebar pick outside the fixed universe_tickers set (e.g. typed in)
                hist = get_earnings_history(t)
            hist = hist.dropna(subset=["reported_eps", "surprise_pct"])
            if hist.empty:
                continue
            fig_surprise.add_trace(
                go.Scatter(
                    x=hist["earnings_date"],
                    y=hist["surprise_pct"],
                    mode="lines+markers",
                    name=t,
                    line=dict(width=2, color=palette[i % len(palette)]),
                    marker=dict(size=6),
                )
            )
            plotted = True
        if plotted:
            fig_surprise.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_surprise.update_layout(
                yaxis_title="EPS surprise %",
                xaxis_title="Earnings date",
                hovermode="x unified",
                legend_title_text="",
                height=420,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_surprise, width="stretch", key="chart_pead_surprise_history")
        else:
            st.info("No earnings history for the tickers picked in the sidebar (e.g. crypto/metals/ETFs don't report EPS).")

    st.divider()
    st.subheader("All stocks: last 4 quarters + next report")
    summary_rows = []
    for t in universe_tickers:
        hist_full = earnings_by_ticker[t]
        if hist_full.empty:
            continue
        reported = hist_full.dropna(subset=["reported_eps", "surprise_pct"]).sort_values("earnings_date")
        last4 = reported.tail(4)["surprise_pct"].tolist()
        while len(last4) < 4:
            last4.insert(0, None)

        upcoming = hist_full[hist_full["reported_eps"].isna()]
        upcoming_date = upcoming["earnings_date"].min() if not upcoming.empty else None

        summary_rows.append(
            {
                "Ticker": f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})",
                "4 reports ago": last4[0],
                "3 reports ago": last4[1],
                "2 reports ago": last4[2],
                "Last report": last4[3],
                "Next earnings date": upcoming_date.strftime("%Y-%m-%d") if pd.notna(upcoming_date) else "—",
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        pct_cols = ["4 reports ago", "3 reports ago", "2 reports ago", "Last report"]
        st.dataframe(
            summary_df.style.format({c: (lambda v: "" if pd.isna(v) else f"{v:+.2f}%") for c in pct_cols}),
            width=1100,
            key="table_pead_all_stocks",
            hide_index=True,
        )
    else:
        st.info("No earnings history available.")


def _format_compact_dollars(value: float) -> str:
    """$52,300,000 -> "$52.3M" -- insider/institutional $ flows span from a
    few thousand to billions, and a raw comma-formatted number is hard to
    scan at a glance across that range.
    """
    if pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{sign}${magnitude / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:.1f}M"
    if magnitude >= 1e3:
        return f"{sign}${magnitude / 1e3:.1f}K"
    return f"{sign}${magnitude:,.0f}"


def render_insider_tab() -> None:
    st.caption(
        "Officer/director insider transactions (SEC Form 4 filings, via yfinance). "
        "Not a backtest — just raw activity: who's buying or selling, and how much."
    )
    universe_tickers = sorted(STOCK_TICKER_TO_NAME)
    lookback_months = st.pills(
        "Lookback", options=[3, 6, 12, 24], format_func=lambda m: f"{m} months", default=3, key="insider_months"
    )
    if not lookback_months:
        st.info("Pick a lookback above.")
        return
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(lookback_months * 30.44))

    with st.spinner("Fetching insider transactions (one request per stock, cached after first run)..."):
        tx_by_ticker = {t: get_insider_transactions(t) for t in universe_tickers}

    all_tx = []
    for t, tx in tx_by_ticker.items():
        if tx.empty:
            continue
        recent = tx[tx["start_date"] >= cutoff].copy()
        if recent.empty:
            continue
        recent["ticker"] = t
        all_tx.append(recent)

    if not all_tx:
        st.info("No insider transactions found for this universe in that window.")
        return

    combined = pd.concat(all_tx, ignore_index=True)
    combined["is_sale"] = combined["transaction"].str.contains("Sale", case=False, na=False)
    combined["is_buy"] = combined["transaction"].str.contains("Buy|Purchase", case=False, na=False, regex=True)

    sidebar_typed = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    sidebar_tickers = sorted(picked_tickers | sidebar_typed)
    if not sidebar_tickers:
        st.info("Pick tickers from the sidebar (or type some in) to see their per-company insider activity.")
    else:
        for t in sidebar_tickers:
            company_tx = combined[combined["ticker"] == t]
            company_name = STOCK_TICKER_TO_NAME.get(t, t)
            major = get_major_holders(t)
            insiders_pct = major.get("insidersPercentHeld", float("nan")) * 100

            st.subheader(f"{company_name} ({t}) — last {int(lookback_months)} months")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Filings", len(company_tx))
            m2.metric("Buy filings", int(company_tx["is_buy"].sum()))
            m3.metric("Sale filings", int(company_tx["is_sale"].sum()))
            net_value = (
                company_tx.loc[company_tx["is_buy"], "value"].sum() - company_tx.loc[company_tx["is_sale"], "value"].sum()
            )
            m4.metric("Net $ (buys - sales)", _format_compact_dollars(net_value))
            m5.metric("Insiders % held", f"{insiders_pct:.2f}%" if pd.notna(insiders_pct) else "—")

            with st.expander(f"All filings ({len(company_tx)})", expanded=False):
                if company_tx.empty:
                    st.caption("No insider transactions on record for this window.")
                else:
                    tx_display = company_tx.copy()
                    tx_display["Date"] = tx_display["start_date"].dt.date
                    tx_display = tx_display.rename(
                        columns={
                            "insider": "Insider",
                            "position": "Position",
                            "transaction": "Transaction",
                            "shares": "Shares",
                            "value": "Value",
                        }
                    )
                    st.dataframe(
                        tx_display[["Date", "Insider", "Position", "Transaction", "Shares", "Value"]]
                        .sort_values("Date", ascending=False)
                        .style.format({"Shares": "{:,.0f}", "Value": "${:,.0f}"}, na_rep="—"),
                        width=900,
                        key=f"table_insider_company_filings_{t}",
                        hide_index=True,
                    )
            st.divider()

    st.subheader("All stocks: filing counts")
    st.caption("Buy/sale filing counts and net $ (buys - sales), across the whole equity universe, in the lookback window.")
    universe_summary = (
        combined.groupby("ticker")
        .apply(
            lambda g: pd.Series(
                {
                    "Filings": len(g),
                    "Buy filings": int(g["is_buy"].sum()),
                    "Sale filings": int(g["is_sale"].sum()),
                    "Net $": g.loc[g["is_buy"], "value"].sum() - g.loc[g["is_sale"], "value"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    universe_summary["Ticker"] = universe_summary["ticker"].map(lambda t: f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})")
    universe_summary = universe_summary.sort_values("Net $", ascending=False)
    universe_summary["Net $"] = universe_summary["Net $"].map(_format_compact_dollars)
    st.dataframe(
        universe_summary[["Ticker", "Filings", "Buy filings", "Sale filings", "Net $"]],
        width=800,
        key="table_insider_universe_summary",
        hide_index=True,
    )

    st.divider()
    st.subheader("Recent activity across universe")
    st.caption("Filings of at least $100k, across the whole equity universe, in the lookback window.")
    sizable = combined[combined["value"] >= 100_000]
    if sizable.empty:
        st.info("No filings of at least $100k in this window.")
    else:
        display = sizable.copy()
        display["Ticker"] = display["ticker"].map(lambda t: f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})")
        display["Date"] = display["start_date"].dt.date
        display = display.rename(
            columns={"insider": "Insider", "position": "Position", "transaction": "Transaction", "shares": "Shares", "value": "Value"}
        )
        st.dataframe(
            display[["Ticker", "Date", "Insider", "Position", "Transaction", "Shares", "Value"]]
            .sort_values("Date", ascending=False)
            .style.format({"Shares": "{:,.0f}", "Value": "${:,.0f}"}, na_rep="—"),
            width=1100,
            key="table_insider_recent",
            hide_index=True,
        )


def render_ownership_tab() -> None:
    st.caption(
        "Institutional + mutual fund ownership (13F-derived, via yfinance) — the ~10 largest "
        "institutional holders and ~10 largest mutual fund holders reported each quarter. "
        "'% held' is % of shares outstanding held by that holder; '% change' is the change in "
        "that holder's position since the prior quarter's filing."
    )
    universe_tickers = sorted(STOCK_TICKER_TO_NAME)

    sidebar_typed = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    sidebar_tickers = sorted(picked_tickers | sidebar_typed)
    if not sidebar_tickers:
        st.info("Pick tickers from the sidebar (or type some in) to see their per-company ownership breakdown.")

    for t in sidebar_tickers:
        ownership = get_institutional_ownership(t)
        major = get_major_holders(t)
        company_name = STOCK_TICKER_TO_NAME.get(t, t)

        as_of = ownership["date_reported"].max().date() if not ownership.empty else None
        header = f"{company_name} ({t})"
        if as_of is not None:
            header += f" — as of {as_of}"
        st.subheader(header)

        if ownership.empty:
            st.caption("No institutional/fund ownership data available.")
            continue

        avg_pct_held = ownership["pct_held"].mean() * 100
        avg_pct_change = ownership["pct_change"].mean() * 100
        institutions_pct = major.get("institutionsPercentHeld", float("nan")) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg % held (inst. + funds)", f"{avg_pct_held:.2f}%")
        m2.metric("Avg % change vs prior quarter", f"{avg_pct_change:+.2f}%")
        m3.metric("Total institutional ownership", f"{institutions_pct:.2f}%" if pd.notna(institutions_pct) else "—")

        with st.expander(f"Holders ({len(ownership)})", expanded=False):
            display = ownership.copy()
            display["Date reported"] = display["date_reported"].dt.date
            display = display.rename(
                columns={
                    "holder": "Holder",
                    "holder_type": "Type",
                    "pct_held": "% held",
                    "shares": "Shares",
                    "value": "Value",
                    "pct_change": "% change",
                }
            )
            st.dataframe(
                display[["Holder", "Type", "Date reported", "% held", "Shares", "Value", "% change"]]
                .sort_values("Value", ascending=False)
                .style.format({"% held": "{:.2%}", "Shares": "{:,.0f}", "Value": "${:,.0f}", "% change": "{:+.2%}"}),
                width=1000,
                key=f"table_ownership_{t}",
                hide_index=True,
            )

        st.divider()

    st.subheader("Stocks by institutional/fund position change")
    st.caption(
        "Across the equity universe, ranked by the largest average % change (positive or negative) "
        "in institutional + mutual fund holders' positions since the prior quarter's filing."
    )
    with st.spinner("Fetching ownership data (one request per stock, cached after first run)..."):
        rows = []
        for t in universe_tickers:
            ownership = get_institutional_ownership(t)
            if ownership.empty:
                continue
            rows.append(
                {
                    "ticker": t,
                    "avg_pct_held": ownership["pct_held"].mean() * 100,
                    "avg_pct_change": ownership["pct_change"].mean() * 100,
                    "as_of": ownership["date_reported"].max().date(),
                }
            )

    if not rows:
        st.info("No ownership data available for this universe.")
        return

    ranking = pd.DataFrame(rows)
    ranking["abs_change"] = ranking["avg_pct_change"].abs()
    ranking_display = ranking.sort_values("abs_change", ascending=False).copy()
    ranking_display["Ticker"] = ranking_display["ticker"].map(lambda t: f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})")
    ranking_display = ranking_display.rename(
        columns={"avg_pct_held": "Avg % held", "avg_pct_change": "Avg % change", "as_of": "As of"}
    )
    st.dataframe(
        ranking_display[["Ticker", "Avg % held", "Avg % change", "As of"]].style.format(
            {"Avg % held": "{:.2f}%", "Avg % change": "{:+.2f}%"}
        ),
        width=800,
        key="table_ownership_ranking",
        hide_index=True,
    )


def render_ranking_tab() -> None:
    st.caption(
        "Relative rank of stocks across six factor categories: Growth, Valuation, Quality, Momentum, "
        "Sentiment, and Ownership/Flow. Each raw factor is converted to a percentile rank within this "
        "universe (100 = best), factors within a category are averaged equally, and categories are "
        "combined into one composite score using the weights below (default: equal). All data is "
        "today's live snapshot from yfinance -- for point-in-time historical values, see the Metrics tab."
    )
    universe_tickers = sorted(STOCK_TICKER_TO_NAME)

    momentum_weeks_col, _ = st.columns([1, 4])
    with momentum_weeks_col:
        momentum_weeks = st.number_input(
            "Momentum lookback (weeks)", min_value=1, max_value=52, value=12, step=1, key="rank_momentum_weeks"
        )

    st.caption("Category weights (relative -- don't need to sum to anything in particular):")
    weight_cols = st.columns(len(FACTOR_CATEGORIES))
    weights = {}
    for col, category in zip(weight_cols, FACTOR_CATEGORIES):
        with col:
            weights[category] = st.slider(category, min_value=0.0, max_value=3.0, value=1.0, step=0.25, key=f"rank_w_{category}")

    price_days_needed = momentum_weeks * 7 + 15
    price_start = (dt.date.today() - dt.timedelta(days=price_days_needed)).isoformat()

    with st.spinner("Fetching prices..."):
        prices = get_prices(universe_tickers + [SP500_BENCHMARK], start=price_start)

    with st.spinner("Fetching fundamentals/analyst data (one request per stock, cached after first run)..."):
        factors = build_factor_table(universe_tickers, prices, prices[SP500_BENCHMARK], momentum_weeks=int(momentum_weeks))

    if factors.empty:
        st.info("No factor data available for this universe.")
        return

    ranked = percentile_rank_table(factors)
    scores = category_scores(ranked)
    composite = composite_score(scores, weights)

    table = scores.copy()
    table.insert(0, "Composite", composite)
    table = table.sort_values("Composite", ascending=False)
    table.insert(0, "Ticker", [f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})" for t in table.index])

    st.subheader("Composite rank")
    st.dataframe(
        table.reset_index(drop=True).style.format({c: "{:.1f}" for c in table.columns if c != "Ticker"}),
        width=1100,
        key="table_ranking_composite",
        hide_index=True,
    )

    with st.expander("Raw factor values", expanded=False):
        raw_display = factors.copy()
        raw_display.insert(0, "Ticker", [f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})" for t in raw_display.index])
        st.dataframe(
            raw_display.reset_index(drop=True),
            width=1400,
            key="table_ranking_raw",
            hide_index=True,
        )

    with st.expander("Percentile ranks (per factor)", expanded=False):
        rank_display = ranked.copy()
        rank_display.insert(0, "Ticker", [f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})" for t in rank_display.index])
        st.dataframe(
            rank_display.reset_index(drop=True).style.format({c: "{:.0f}" for c in ranked.columns}),
            width=1400,
            key="table_ranking_percentiles",
            hide_index=True,
        )

    with st.expander("What's in each factor?", expanded=False):
        for category, descriptions in FACTOR_DESCRIPTIONS.items():
            st.markdown(f"**{category}**")
            for factor, description in descriptions.items():
                direction = "higher is better" if HIGHER_IS_BETTER.get(factor, True) else "lower is better"
                st.markdown(f"- `{factor}` -- {description} ({direction})")


def render_weekly_tab() -> None:
    caption_col, refresh_col = st.columns([5, 1])
    with caption_col:
        st.caption(
            "Items actually worth checking on a weekly cadence -- filtered down to what changes that "
            "often. Quarterly-cadence stuff (fundamentals, institutional 13F flow, the Metrics tab) is "
            "left out on purpose since it won't have moved since last week."
        )
    with refresh_col:
        weekly_refresh = st.button(
            "Refresh data", key="weekly_refresh_btn",
            help="Bypass the cache and re-fetch everything on this tab fresh from yfinance.",
        )

    universe_tickers = sorted(STOCK_TICKER_TO_NAME)
    today = pd.Timestamp.today().normalize()
    window_start = today - pd.Timedelta(days=7)
    window_end = today + pd.Timedelta(days=7)

    momentum_weeks = 12
    ma_days = 50
    price_days_needed = max(momentum_weeks * 7, 95) + 15
    price_start = (dt.date.today() - dt.timedelta(days=price_days_needed)).isoformat()
    with st.spinner("Fetching prices/volume..."):
        weekly_prices = get_prices(universe_tickers + [SP500_BENCHMARK], start=price_start, refresh=weekly_refresh)
        weekly_volumes = get_volume(universe_tickers, start=price_start, refresh=weekly_refresh)

    st.subheader("Upcoming earnings (next 7 days)")
    with st.spinner("Fetching earnings calendar..."):
        earnings_by_ticker = {t: get_earnings_history(t, refresh=weekly_refresh) for t in universe_tickers}
    upcoming_rows = []
    for t, earnings in earnings_by_ticker.items():
        if earnings.empty:
            continue
        future = earnings[earnings["reported_eps"].isna()]
        if future.empty:
            continue
        next_date = future["earnings_date"].min()
        if today <= next_date <= window_end:
            upcoming_rows.append(
                {
                    "Ticker": f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})",
                    "Earnings date": next_date.date(),
                    "EPS estimate": future.loc[future["earnings_date"] == next_date, "eps_estimate"].iloc[0],
                }
            )
    if upcoming_rows:
        st.dataframe(
            pd.DataFrame(upcoming_rows).sort_values("Earnings date"),
            width=700,
            key="table_weekly_earnings",
            hide_index=True,
        )
    else:
        st.caption("No earnings scheduled in this universe over the next 7 days.")

    st.divider()
    st.subheader("Earning reports")
    st.caption("Earnings actually reported in the last 7 days -- catches results that just landed.")
    reported_rows = []
    for t, earnings in earnings_by_ticker.items():
        if earnings.empty:
            continue
        reported = earnings.dropna(subset=["reported_eps"])
        recent = reported[(reported["earnings_date"] >= window_start) & (reported["earnings_date"] <= today)]
        for _, r in recent.iterrows():
            reported_rows.append(
                {
                    "Ticker": t,
                    "Earnings date": r["earnings_date"].date(),
                    "Surprise %": r["surprise_pct"],
                }
            )
    if reported_rows:
        reported_df = pd.DataFrame(reported_rows).dropna(subset=["Surprise %"]).sort_values("Surprise %")
        bar_colors = ["#1baf7a" if v >= 0 else "#e34948" for v in reported_df["Surprise %"]]
        fig_reported = go.Figure(
            go.Bar(
                x=reported_df["Surprise %"],
                y=reported_df["Ticker"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.1f}%" for v in reported_df["Surprise %"]],
                textposition="outside",
                hovertemplate="%{y}<br>Surprise: %{x:+.1f}%<extra></extra>",
            )
        )
        fig_reported.add_vline(x=0, line_color="gray", line_width=1)
        fig_reported.update_layout(
            xaxis_title="EPS surprise %",
            yaxis_title="",
            height=max(220, 40 * len(reported_df) + 60),
            margin=dict(t=20, b=20, l=10, r=40),
        )
        st.plotly_chart(fig_reported, width="stretch", key="chart_weekly_reported")
    else:
        st.caption("No earnings reported in this universe over the last 7 days.")

    st.divider()
    st.subheader("Recent analyst actions (last 7 days)")
    with st.spinner("Fetching analyst upgrades/downgrades..."):
        analyst_rows = []
        for t in universe_tickers:
            ud = get_upgrades_downgrades(t, refresh=weekly_refresh)
            if ud.empty:
                continue
            recent = ud[ud["grade_date"] >= window_start]
            if recent.empty:
                continue
            recent = recent.copy()
            recent["TickerRaw"] = t
            recent["Ticker"] = f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})"
            analyst_rows.append(recent)
    if analyst_rows:
        combined = pd.concat(analyst_rows, ignore_index=True)

        summary = combined.groupby("TickerRaw")["action"].agg(
            Actions="count",
            Up=lambda a: int((a == "up").sum()),
            Down=lambda a: int((a == "down").sum()),
        )
        summary["Maintain"] = summary["Actions"] - summary["Up"] - summary["Down"]
        # 0 means "no target given with that action" (see get_upgrades_downgrades),
        # not an actual $0 target -- excluded so it doesn't drag the average down.
        priced = combined[combined["current_price_target"] > 0]
        summary["Avg target"] = priced.groupby("TickerRaw")["current_price_target"].mean()
        summary["Current price"] = pd.Series(
            {t: weekly_prices[t].dropna().iloc[-1] for t in summary.index if t in weekly_prices.columns and not weekly_prices[t].dropna().empty}
        )
        summary["Upside %"] = (summary["Avg target"] / summary["Current price"] - 1) * 100
        summary["Last action date"] = combined.groupby("TickerRaw")["grade_date"].max().dt.date
        summary["Label"] = [
            f"{t} [U={u}, D={d}, M={m}] {date}"
            for t, u, d, m, date in zip(
                summary.index, summary["Up"], summary["Down"], summary["Maintain"], summary["Last action date"]
            )
        ]

        priced_summary = summary.dropna(subset=["Avg target", "Current price"])
        if not priced_summary.empty:
            st.caption(
                "Analyst average price target vs. today's price -- line length = the implied "
                "upside/downside. [U/D/M] = up/down/maintain action counts this week; date = most "
                "recent action."
            )
            target_sorted = priced_summary.sort_values("Upside %")
            fig_target = go.Figure()
            for _, r in target_sorted.iterrows():
                line_color = "#1baf7a" if r["Upside %"] >= 0 else "#e34948"
                fig_target.add_trace(
                    go.Scatter(
                        x=[r["Current price"], r["Avg target"]],
                        y=[r["Label"], r["Label"]],
                        mode="lines",
                        line=dict(color=line_color, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig_target.add_trace(
                go.Scatter(
                    x=target_sorted["Current price"],
                    y=target_sorted["Label"],
                    mode="markers",
                    name="Current price",
                    marker=dict(color="#2a78d6", size=9),
                    hovertemplate="%{y}<br>Current: $%{x:.2f}<extra></extra>",
                )
            )
            fig_target.add_trace(
                go.Scatter(
                    x=target_sorted["Avg target"],
                    y=target_sorted["Label"],
                    mode="markers",
                    name="Avg target",
                    marker=dict(color="#eda100", size=9),
                    customdata=target_sorted["Upside %"],
                    hovertemplate="%{y}<br>Avg target: $%{x:.2f}<br>Upside: %{customdata:+.1f}%<extra></extra>",
                )
            )
            fig_target.update_layout(
                xaxis_title="Price ($)",
                yaxis_title="",
                legend_title_text="",
                height=max(220, 40 * len(target_sorted) + 60),
                margin=dict(t=20, b=20, l=10, r=40),
            )
            st.plotly_chart(fig_target, width="stretch", key="chart_weekly_analyst_target")

        combined["Date"] = combined["grade_date"].dt.date
        combined = combined.rename(
            columns={
                "firm": "Firm",
                "to_grade": "To grade",
                "from_grade": "From grade",
                "action": "Action",
                "current_price_target": "Price target",
            }
        )
        with st.expander(f"All actions ({len(combined)})", expanded=False):
            st.dataframe(
                combined[["Ticker", "Date", "Firm", "Action", "From grade", "To grade", "Price target"]]
                .sort_values("Date", ascending=False)
                .style.format({"Price target": "${:,.2f}"}, na_rep="—"),
                width=1100,
                key="table_weekly_analyst",
                hide_index=True,
            )
    else:
        st.caption("No analyst actions in this universe over the last 7 days.")

    st.divider()
    st.subheader("Momentum & technical snapshot")
    st.caption(
        "Price-based factors, current as of today's close -- these are the ones that actually move "
        "week to week, unlike the fundamentals-based factors in Ranking/Metrics."
    )
    with st.expander("How these are computed", expanded=True):
        st.markdown(
            "- **7-day return** -- raw price return over the trailing 7 days.\n"
            "- **Momentum (2w / 12w)** -- price return over the trailing 2 or 12 weeks.\n"
            "- **Relative strength (2w / 12w)** -- that same window's return minus the S&P 500's "
            "return over the identical window.\n"
            "- **Price vs MA** -- % above/below the 50-day moving average of closing price.\n"
            "- **Relative volume** -- 10-day average volume divided by 60-day average volume, "
            "both ending today (a smoothed ratio, not a single day's volume spike)."
        )
    bench_series = weekly_prices[SP500_BENCHMARK].dropna() if SP500_BENCHMARK in weekly_prices.columns else pd.Series(dtype=float)
    technical_rows = []
    for t in universe_tickers:
        if t not in weekly_prices.columns:
            continue
        series = weekly_prices[t].dropna()
        if len(series) < 2:
            continue
        row: dict[str, object] = {"Ticker": t}

        seven_day_start = series.index[-1] - pd.Timedelta(days=7)
        window7 = series[series.index >= seven_day_start]
        if len(window7) > 1:
            row["7-day return"] = (window7.iloc[-1] / window7.iloc[0] - 1) * 100

        for label, lookback_days in [("2w", 14), ("12w", momentum_weeks * 7)]:
            lookback_start = series.index[-1] - pd.Timedelta(days=lookback_days)
            window = series[series.index >= lookback_start]
            if len(window) > 1:
                stock_ret = window.iloc[-1] / window.iloc[0] - 1
                row[f"Momentum ({label})"] = stock_ret * 100
                bench_window = bench_series[(bench_series.index >= window.index[0]) & (bench_series.index <= window.index[-1])]
                if len(bench_window) > 1:
                    bench_ret = bench_window.iloc[-1] / bench_window.iloc[0] - 1
                    row[f"Relative strength ({label})"] = (stock_ret - bench_ret) * 100

        if len(series) >= ma_days:
            ma = series.tail(ma_days).mean()
            row["Price vs MA"] = (series.iloc[-1] / ma - 1) * 100

        if t in weekly_volumes.columns:
            vol_series = weekly_volumes[t].dropna()
            if len(vol_series) >= 60:
                short_avg, long_avg = vol_series.tail(10).mean(), vol_series.tail(60).mean()
                if long_avg:
                    row["Relative volume"] = (short_avg / long_avg - 1) * 100

        technical_rows.append(row)

    if technical_rows:
        technical_cols = [
            "7-day return", "Momentum (2w)", "Momentum (12w)", "Relative strength (2w)",
            "Relative strength (12w)", "Price vs MA", "Relative volume",
        ]
        technical_table = pd.DataFrame(technical_rows).sort_values("Momentum (12w)", ascending=False)
        display_table = technical_table[["Ticker", *technical_cols]]
        styled = display_table.style.format("{:+.1f}%", subset=technical_cols, na_rep="—")
        # Diverging background per column, centered at 0 (not a shared scale across
        # columns -- Momentum swings much wider than Price vs MA, so a symmetric
        # per-column range keeps each column's own colors meaningful) -- lets you spot
        # a broadly green (strong) vs. broadly red (weak) row without reading every number.
        for col in technical_cols:
            col_max = display_table[col].abs().max()
            if pd.notna(col_max) and col_max > 0:
                styled = styled.background_gradient(cmap="RdYlGn", subset=[col], vmin=-col_max, vmax=col_max)
        st.dataframe(
            styled,
            width="stretch",
            column_config={
                "Ticker": st.column_config.Column(width="small"),
                **{col: st.column_config.Column(width="small") for col in technical_cols},
            },
            key="table_weekly_technical",
            hide_index=True,
        )
    else:
        st.caption("No price data available for this universe.")

    st.divider()
    st.subheader("Diverged pairs (correlation ≥ 0.8, |Z-score| ≥ 1.5)")
    st.caption(
        "Pairs that usually move together (return correlation over a 6-month lookback) but whose "
        "cumulative-return spread is currently unusual for that pair -- a mean-reversion candidate. "
        "Same methodology as the Correlations tab."
    )
    corr_universe = sorted(TICKER_TO_NAME)
    with st.spinner("Fetching prices for correlation analysis..."):
        corr_lookback_prices = get_prices(
            corr_universe, start=(today - pd.Timedelta(days=185)).date().isoformat(), refresh=weekly_refresh
        )
    corr_lookback_prices = corr_lookback_prices.dropna(axis=1, how="all")
    corr_daily_returns = corr_lookback_prices.pct_change(fill_method=None).dropna(how="all")
    corr_pairs = pairwise_correlation(corr_daily_returns)
    if corr_pairs.empty:
        st.caption("Not enough overlapping data to compute correlations.")
    else:
        corr_candidates = corr_pairs[corr_pairs["correlation"] >= 0.8]
        corr_cumulative = corr_lookback_prices / corr_lookback_prices.bfill().iloc[0] - 1
        corr_diverged = divergence_now(corr_cumulative, corr_candidates)
        corr_diverged = corr_diverged[corr_diverged["z_score"].abs() >= 1.5]
        if corr_diverged.empty:
            st.caption("No pairs currently meet both thresholds.")
        else:
            corr_diverged = corr_diverged.reindex(corr_diverged["z_score"].abs().sort_values(ascending=False).index)
            corr_diverged["A"] = corr_diverged["ticker_a"].map(lambda t: f"{TICKER_TO_NAME.get(t, t)} ({t})")
            corr_diverged["B"] = corr_diverged["ticker_b"].map(lambda t: f"{TICKER_TO_NAME.get(t, t)} ({t})")
            st.dataframe(
                corr_diverged[["A", "B", "correlation", "z_score"]]
                .rename(columns={"correlation": "Correlation", "z_score": "Z-score"})
                .style.format({"Correlation": "{:.2f}", "Z-score": "{:+.2f}"}),
                width=600,
                key="table_weekly_diverged",
                hide_index=True,
            )

    st.divider()
    st.subheader("Dip candidates (fell 5%+ in a single day, last 7 days)")
    with st.spinner("Fetching prices..."):
        dip_prices = get_prices(
            universe_tickers, start=(today - pd.Timedelta(days=21)).date().isoformat(), refresh=weekly_refresh
        )
    dip_returns = dip_prices.pct_change(fill_method=None)
    dip_rows = []
    for t in universe_tickers:
        if t not in dip_returns.columns:
            continue
        recent = dip_returns[t][(dip_returns.index >= window_start) & (dip_returns[t] <= -0.05)]
        for d, r in recent.items():
            dip_rows.append({"Ticker": t, "Date": d.date(), "Drop": r * 100})
    if dip_rows:
        dip_df = pd.DataFrame(dip_rows).sort_values(["Date", "Drop"], ascending=[False, True])
        dip_df["Label"] = dip_df["Ticker"] + " (" + dip_df["Date"].astype(str) + ")"
        # reversed so the chart reads top-to-bottom in the same order as dip_df
        # (Plotly's horizontal bars otherwise plot the first row at the bottom)
        plot_df = dip_df.iloc[::-1]
        fig_dip = go.Figure(
            go.Bar(
                x=plot_df["Drop"],
                y=plot_df["Label"],
                orientation="h",
                marker_color="#e34948",
                text=[f"{v:+.1f}%" for v in plot_df["Drop"]],
                textposition="outside",
                hovertemplate="%{y}<br>Drop: %{x:+.1f}%<extra></extra>",
            )
        )
        fig_dip.add_vline(x=0, line_color="gray", line_width=1)
        # bars extend negative (left), so the "outside" %-drop text sits further left of
        # the bar end -- pad the x-axis range so that text has room and doesn't get clipped.
        min_drop = plot_df["Drop"].min()
        fig_dip.update_layout(
            xaxis_title="Single-day drop %",
            xaxis_range=[min_drop * 1.35, -min_drop * 0.15],
            yaxis_title="",
            height=max(220, 40 * len(dip_df) + 60),
            margin=dict(t=20, b=20, l=10, r=40),
        )
        st.plotly_chart(fig_dip, width="stretch", key="chart_weekly_dip")
    else:
        st.caption("No single-day drops of 5%+ in this universe over the last 7 days.")

    st.divider()
    st.subheader("Volume spikes (2x+ the 60-day average, last 7 days)")
    st.caption(
        "A single day's volume vs. the trailing 60-day average as of that day -- catches a sudden "
        "spike (news, filing, index flow) that a smoothed 10-day/60-day ratio can dilute away."
    )
    spike_rows = []
    for t in universe_tickers:
        if t not in weekly_volumes.columns:
            continue
        vol_series = weekly_volumes[t].dropna()
        recent_days = vol_series.index[vol_series.index >= window_start]
        for d in recent_days:
            trailing = vol_series[vol_series.index < d].tail(60)
            if len(trailing) < 60:
                continue
            avg60 = trailing.mean()
            if avg60 and vol_series[d] >= 2 * avg60:
                spike_rows.append(
                    {
                        "Ticker": t,
                        "Date": d.date(),
                        "Volume": vol_series[d],
                        "vs 60-day avg": (vol_series[d] / avg60 - 1) * 100,
                    }
                )
    if spike_rows:
        spike_df = pd.DataFrame(spike_rows).sort_values(["Date", "vs 60-day avg"], ascending=[False, False])
        spike_df["Label"] = spike_df["Ticker"] + " (" + spike_df["Date"].astype(str) + ")"
        # reversed so the chart reads top-to-bottom in the same order as spike_df
        plot_df = spike_df.iloc[::-1]
        fig_spike = go.Figure(
            go.Bar(
                x=plot_df["vs 60-day avg"],
                y=plot_df["Label"],
                orientation="h",
                marker_color="#2a78d6",
                text=[f"+{v:.0f}%" for v in plot_df["vs 60-day avg"]],
                textposition="outside",
                customdata=plot_df["Volume"],
                hovertemplate="%{y}<br>vs 60-day avg: +%{x:.0f}%<br>Volume: %{customdata:,.0f}<extra></extra>",
            )
        )
        fig_spike.update_layout(
            xaxis_title="Volume vs. 60-day average %",
            yaxis_title="",
            height=max(220, 40 * len(spike_df) + 60),
            margin=dict(t=20, b=20, l=10, r=50),
        )
        st.plotly_chart(fig_spike, width="stretch", key="chart_weekly_volume_spikes")
    else:
        st.caption("No volume spikes of 2x+ in this universe over the last 7 days.")

    st.divider()
    st.subheader("Recent insider filings (last 7 days, at least $100k)")
    with st.spinner("Fetching insider transactions..."):
        insider_rows = []
        for t in universe_tickers:
            tx = get_insider_transactions(t, refresh=weekly_refresh)
            if tx.empty:
                continue
            recent = tx[(tx["start_date"] >= window_start) & (tx["value"] >= 100_000)]
            if recent.empty:
                continue
            recent = recent.copy()
            recent["Ticker"] = f"{STOCK_TICKER_TO_NAME.get(t, t)} ({t})"
            insider_rows.append(recent)
    if insider_rows:
        combined_insider = pd.concat(insider_rows, ignore_index=True)
        combined_insider["Date"] = combined_insider["start_date"].dt.date
        combined_insider = combined_insider.rename(
            columns={"insider": "Insider", "position": "Position", "transaction": "Transaction", "value": "Value"}
        )
        st.dataframe(
            combined_insider[["Ticker", "Date", "Insider", "Position", "Transaction", "Value"]]
            .sort_values("Date", ascending=False)
            .style.format({"Value": "${:,.0f}"}),
            width=1000,
            key="table_weekly_insider",
            hide_index=True,
        )
    else:
        st.caption("No insider filings of at least $100k in this universe over the last 7 days.")



PANEL_CSV_PATH = "output/panel.csv"


PANEL_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# snake_case column name -> capitalized acronym, for `_metric_label`.
_METRIC_LABEL_ACRONYMS = {"pe": "PE", "ma": "MA", "fcf": "FCF"}


def _metric_label(key: str) -> str:
    """"trailing_pe" -> "Trailing PE", "price_vs_ma" -> "Price vs MA", etc."""
    words = key.split("_")
    labeled = [
        _METRIC_LABEL_ACRONYMS.get(w.lower(), w.capitalize() if i == 0 else w.lower()) for i, w in enumerate(words)
    ]
    return " ".join(labeled)


# metric -> unit kind, for the chart's y-axis label. "$" gets a dynamic
# K/millions/billions scale chosen from the plotted data (see `_dollar_scale`)
# since insider $ flow spans from a few thousand to billions across tickers.
_METRIC_UNITS: dict[str, str] = {
    "revenue_growth": "%",
    "earnings_growth": "%",
    "trailing_pe": "x",
    "price_to_sales": "x",
    "operating_margin": "fraction",
    "fcf_margin": "fraction",
    "momentum": "%",
    "relative_strength": "%",
    "price_vs_ma": "%",
    "relative_volume": "%",
    "analyst_upside": "%",
    "net_upgrades": "count",
    "revisions_trend": "%",
    "institutional_flow": "%",
    "insider_flow": "$",
    "forward_return": "%",
}


def _dollar_scale(max_abs_value: float) -> tuple[float, str]:
    """(divisor, label) picked from the data's own magnitude -- e.g. (1e6,
    "$ millions") -- so a chart of $50,000,000 shows "50" on an axis labeled
    "$ millions" instead of an unreadable string of zeros.
    """
    if max_abs_value >= 1e9:
        return 1e9, "$ billions"
    if max_abs_value >= 1e6:
        return 1e6, "$ millions"
    if max_abs_value >= 1e3:
        return 1e3, "$ thousands"
    return 1.0, "$"


# metric -> definition, for the Metrics tab. Written independently of
# finance.ranking.FACTOR_DESCRIPTIONS on purpose -- some metrics share a name
# with a Ranking-tab factor but are computed completely differently here (see
# PANEL_METRICS_DIFFER_FROM_RANKING below), so inheriting text from ranking.py
# risks silently describing the wrong thing whenever one module changes and
# the other doesn't. Keeping these fully separate trades a little duplication
# for that safety.
PANEL_METRIC_DESCRIPTIONS: dict[str, str] = {
    "revenue_growth": "Trailing-twelve-months revenue vs. trailing-twelve-months revenue a year earlier.",
    "earnings_growth": "Trailing-twelve-months earnings vs. trailing-twelve-months earnings a year earlier.",
    "trailing_pe": "Price divided by trailing-twelve-months EPS. Lower is better.",
    "price_to_sales": "Market cap divided by trailing-twelve-months revenue. Lower is better.",
    "operating_margin": "Trailing-twelve-months operating income divided by trailing-twelve-months revenue.",
    "fcf_margin": "Trailing-twelve-months operating cash flow divided by trailing-twelve-months revenue.",
    "momentum": "Raw price return over the momentum lookback window, ending at this date.",
    "relative_strength": "That same return minus the S&P 500's return over the identical window.",
    "price_vs_ma": "% above/below the 50-day moving average of closing price.",
    "relative_volume": "10-day average volume divided by 60-day average volume, both ending at this date.",
    "analyst_upside": (
        "% gap between the reconstructed analyst price-target consensus and the price on this date -- "
        "built from each firm's latest dated upgrade/downgrade action known by this date, excluding "
        "firms whose last action is over ~18 months old."
    ),
    "net_upgrades": "Upgrades minus downgrades in the trailing 90 days ending at this date.",
    "revisions_trend": (
        "Change in upgrade momentum: net upgrades in the trailing 90 days ending at this date, minus net "
        "upgrades in the 90 days before that."
    ),
    "institutional_flow": (
        "Share-count % change of the top-10 SEC 13F institutional holders (by shares), for the most "
        "recent reporting quarter fully knowable as of this date."
    ),
    "insider_flow": "Net $ of insider buys minus sells (Form 4 filings) in the trailing 180 days ending at this date.",
    "forward_return": (
        "The ticker's actual price return from this date to ~20 calendar days later -- "
        "the training target, not a point-in-time-safe feature (it looks into the future by design)."
    ),
}

# Metrics that share a column name with a finance.ranking factor but are
# computed with a genuinely different methodology there (not just "live
# snapshot vs. point-in-time reconstruction of the same formula") -- flagged
# in the UI so a user comparing the two tabs doesn't assume they're the same
# number computed on a different date. Checked directly against
# finance/ranking.py's build_factor_table and finance/pointintime.py:
#   - revenue_growth / earnings_growth: yfinance's single-most-recent-quarter
#     YoY (ranking.py) vs. a true trailing-twelve-months YoY here.
#   - relative_volume: today's volume / yfinance's own "averageVolume" field
#     (ranking.py) vs. a 10-day/60-day historical average ratio here.
#   - analyst_upside: yfinance's live targetMeanPrice, aggregated over
#     whichever analysts currently cover the stock (ranking.py), vs. a
#     reconstruction from the dated upgrade/downgrade log only, with a
#     staleness cutoff, here.
#   - revisions_trend: yfinance's buy/hold/sell recommendation-trend field
#     (ranking.py) vs. a momentum-of-net-upgrades proxy here -- these use
#     different underlying data entirely (see pointintime.py's own docstring).
#   - institutional_flow: yfinance's own ~10 institutional + ~10 fund holder
#     list (ranking.py) vs. the top-10-by-shares SEC 13F filers here --
#     confirmed to disagree substantially in practice (e.g. AMD).
#   - insider_flow: sums *all* available insider transactions with no time
#     window (ranking.py) vs. a trailing-180-day window here.
# momentum/relative_strength/net_upgrades/operating_margin/fcf_margin use the
# same formula in both places (only the anchor date differs -- "today" vs. a
# historical rebalance date), so they're intentionally left out of this set.
PANEL_METRICS_DIFFER_FROM_RANKING: set[str] = {
    "revenue_growth",
    "earnings_growth",
    "relative_volume",
    "analyst_upside",
    "revisions_trend",
    "institutional_flow",
    "insider_flow",
}

# metric -> where its value actually comes from -- panel.py blends four
# different data sources, and it's not obvious from the metric name alone
# which one backs a given number.
PANEL_METRIC_SOURCES: dict[str, str] = {
    "revenue_growth": "SEC XBRL filings",
    "earnings_growth": "SEC XBRL filings",
    "trailing_pe": "SEC XBRL (EPS) + yfinance (price)",
    "price_to_sales": "SEC XBRL (revenue, shares) + yfinance (price)",
    "operating_margin": "SEC XBRL filings",
    "fcf_margin": "SEC XBRL filings",
    "momentum": "yfinance (price history)",
    "relative_strength": "yfinance (price history)",
    "price_vs_ma": "yfinance (price history)",
    "relative_volume": "yfinance (volume history)",
    "analyst_upside": "yfinance (analyst upgrade/downgrade log)",
    "net_upgrades": "yfinance (analyst upgrade/downgrade log)",
    "revisions_trend": "yfinance (analyst upgrade/downgrade log)",
    "institutional_flow": "SEC 13F bulk filings",
    "insider_flow": "yfinance (Form 4 insider filings)",
    "forward_return": "yfinance (price history)",
}


def render_panel_tab() -> None:
    st.caption(
        "Point-in-time metric history from output/panel.csv (finance/panel.py) -- one value per month per "
        "metric, reconstructed from data that was actually knowable as of each date. Pick companies from "
        "the sidebar -- only those with data in panel.csv are shown."
    )

    try:
        panel = pd.read_csv(PANEL_CSV_PATH, parse_dates=["date"])
    except FileNotFoundError:
        st.info(f"{PANEL_CSV_PATH} not found -- generate it first (see finance/panel.py).")
        return

    sidebar_typed = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    sidebar_tickers = sorted((picked_tickers | sidebar_typed) & set(panel["ticker"]))
    if not sidebar_tickers:
        st.info("Pick tickers from the sidebar (or type some in) that have data in output/panel.csv.")
        return

    with st.expander("Data sources", expanded=False):
        st.caption(
            "panel.csv blends four data sources -- which one backs a metric isn't obvious from its "
            "name alone, so here's the breakdown. :red[Red] metrics share a name with a Ranking-tab "
            "factor but are computed with a genuinely different methodology there -- not the same "
            "number on a different date, an actually different definition."
        )
        for m in [*PANEL_FACTOR_COLUMNS, "forward_return"]:
            line = f"`{_metric_label(m)}` -- {PANEL_METRIC_SOURCES.get(m, 'unknown')}"
            if m in PANEL_METRICS_DIFFER_FROM_RANKING:
                st.markdown(f":red[- {line} (differs from Ranking tab)]")
            else:
                st.markdown(f"- {line}")

    metric_options = [*PANEL_FACTOR_COLUMNS, "forward_return"]

    def _metric_pill_label(m: str) -> str:
        label = _metric_label(m)
        return f"{label} 🔴" if m in PANEL_METRICS_DIFFER_FROM_RANKING else label

    metric_help = "\n\n".join(
        f"**{_metric_label(m)}** -- {PANEL_METRIC_DESCRIPTIONS.get(m, '')} (Source: {PANEL_METRIC_SOURCES.get(m, 'unknown')})"
        + (" **Differs from the Ranking tab's definition of the same name.**" if m in PANEL_METRICS_DIFFER_FROM_RANKING else "")
        for m in metric_options
    )
    metric = st.pills(
        "Metric (🔴 = differs from the Ranking tab's definition of the same name)",
        options=metric_options,
        format_func=_metric_pill_label,
        selection_mode="single",
        default=PANEL_FACTOR_COLUMNS[0],
        key="panel_factor",
        help=metric_help,
    )
    if not metric:
        st.info("Pick a metric above.")
        return

    metric_label = _metric_label(metric)
    description_line = f"{PANEL_METRIC_DESCRIPTIONS.get(metric, '')} (Source: {PANEL_METRIC_SOURCES.get(metric, 'unknown')})"
    if metric in PANEL_METRICS_DIFFER_FROM_RANKING:
        st.markdown(f":red[{description_line} -- differs from the Ranking tab's definition of the same name.]")
    else:
        st.caption(description_line)

    histories = {}
    for t in sidebar_tickers:
        history = panel[(panel["ticker"] == t) & (panel["date"] >= "2025-01-01")].sort_values("date")
        history = history.dropna(subset=[metric])
        if not history.empty:
            histories[t] = history

    if not histories:
        st.info(f"No {metric_label} data since 2025-01-01 for the selected companies.")
        return

    unit = _METRIC_UNITS.get(metric, "")
    divisor = 1.0
    axis_label = f"{metric_label} ({unit})" if unit else metric_label
    if unit == "$":
        max_abs_value = max(h[metric].abs().max() for h in histories.values())
        divisor, dollar_label = _dollar_scale(max_abs_value)
        axis_label = f"{metric_label} ({dollar_label})"

    fig = go.Figure()
    for i, t in enumerate(sidebar_tickers):
        if t not in histories:
            continue
        history = histories[t]
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history[metric] / divisor,
                mode="lines+markers",
                name=f"{TICKER_TO_NAME.get(t, t)} ({t})",
                line=dict(width=2, color=PANEL_PALETTE[i % len(PANEL_PALETTE)]),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        yaxis_title=axis_label,
        xaxis_title="Date",
        hovermode="x unified",
        legend_title_text="",
        height=420,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch", key="chart_panel_factor_history")

    with st.expander("Raw values", expanded=False):
        raw = panel[panel["ticker"].isin(sidebar_tickers) & (panel["date"] >= "2025-01-01")][
            ["date", "ticker", metric]
        ].sort_values(["ticker", "date"])
        raw = raw.rename(columns={"date": "Date", "ticker": "Ticker", metric: metric_label})
        st.dataframe(raw.reset_index(drop=True), width=600, key="table_panel_factor_history", hide_index=True)


_MACRO_UNIT_FORMAT = {
    "pct": lambda v: f"{v:.2f}%",
    "usd": lambda v: f"${v:,.2f}",
    "num": lambda v: f"{v:.2f}",
}
# For chart axis ticks specifically -- "usd" drops decimals (a commodity price's cents digit is
# noise at a quick-glance axis scale, unlike the headline value line above the chart, which keeps
# full precision). "pct" keeps 2 decimals even on the axis -- a rate/spread's whole weekly move is
# often just a few basis points, so rounding to whole percent would flatten it to a flat line.
_MACRO_AXIS_UNIT_FORMAT = {
    "pct": _MACRO_UNIT_FORMAT["pct"],
    "usd": lambda v: f"${v:,.0f}",
    "num": _MACRO_UNIT_FORMAT["num"],
}
_MACRO_DELTA_FORMAT = {
    "bps": lambda d: f"{d * 100:+.0f} bps",
    "pp": lambda d: f"{d:+.2f} pp",
    "pct_change": lambda d: f"{d:+.1f}%",
    "points": lambda d: f"{d:+.2f} pts",
}


def _render_macro_tiles(tiles: list[dict]) -> None:
    """A row of st.metric tiles -- delta_color="off" throughout since, unlike a portfolio P&L
    number, a macro series moving up isn't uniformly good or bad (rising oil helps XOM, hurts
    margin-sensitive names elsewhere) -- letting st.metric auto-color it green/red would silently
    editorialize a plain data point.
    """
    cols = st.columns(len(tiles)) if tiles else []
    for col, tile in zip(cols, tiles):
        value_str = _MACRO_UNIT_FORMAT[tile["unit"]](tile["value"])
        delta_str = _MACRO_DELTA_FORMAT[tile["delta_format"]](tile["delta"]) if tile["delta"] is not None else None
        col.metric(tile["label"], value_str, delta_str, delta_color="off", help=f"As of {tile['as_of']}")


def _macro_chart_html(history: list[list], unit: str, width: int = 300, height: int = 80) -> str:
    """A compact inline SVG line+area chart with a 3-tick value axis (max/mid/min, stacked to the
    chart's left, like a real y-axis) and a 3-tick date row below it (start/mid/end) --
    deliberately not st.line_chart, which renders at full column width and native chart height
    regardless of how little content it's showing; that's the right tool for a real
    data-exploration chart, but wrong for a card-sized trend glance. Still no gridlines/full tick
    marks -- just enough to read the chart's actual scale.

    `history` is finance.macro's compact [date_str, value] pair format (see its _history_pairs
    docstring), not {"date":..., "value":...} dicts.

    The value/date labels are plain HTML, not SVG <text> elements inside the chart -- the SVG
    itself uses preserveAspectRatio="none" (stretches non-uniformly to fill the card's actual
    width, which is unknown at render time), and text inside a non-uniformly-scaled SVG renders
    visibly warped (letters stretched or squished) whenever the real width differs from the 300px
    viewBox. Plain HTML text has no such issue -- it just reflows normally.
    """
    values = [h[1] for h in history]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pad = 4
    n = len(values)

    def x(i: int) -> float:
        return pad + (width - 2 * pad) * i / (n - 1)

    def y(v: float) -> float:
        return height - pad - (height - 2 * pad) * (v - lo) / span

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    area = f"{x(0):.1f},{height - pad} {points} {x(n - 1):.1f},{height - pad}"
    svg = (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<polyline points="{area}" fill="rgba(27,175,122,0.15)" stroke="none"/>'
        f'<polyline points="{points}" fill="none" stroke="#1baf7a" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )

    fmt = _MACRO_AXIS_UNIT_FORMAT[unit]
    # Vertical axis ticks are evenly spaced between hi/lo (a real axis midpoint), not the value at
    # whatever data point happens to sit at the middle index -- those two only coincide by chance.
    hi_label = html.escape(fmt(hi))
    mid_label = html.escape(fmt((hi + lo) / 2))
    lo_label = html.escape(fmt(lo))
    # Horizontal axis ticks, by contrast, use the *actual* middle data point's own date -- dates
    # aren't an average to compute, they're calendar days from real data, same as start/end below.
    mid_idx = n // 2
    start_date = dt.date.fromisoformat(history[0][0]).strftime("%m/%d")
    mid_date = dt.date.fromisoformat(history[mid_idx][0]).strftime("%m/%d")
    end_date = dt.date.fromisoformat(history[-1][0]).strftime("%m/%d")
    label_style = "font-size:0.68rem;opacity:0.55;line-height:1.1;white-space:nowrap;"
    return (
        f'<div style="display:flex;align-items:stretch;margin:0.4rem 0 0.1rem;">'
        f'<div style="display:flex;flex-direction:column;justify-content:space-between;'
        f'padding-right:0.3rem;text-align:right;{label_style}">'
        f"<span>{hi_label}</span><span>{mid_label}</span><span>{lo_label}</span>"
        f"</div>"
        f'<div style="flex:1;min-width:0;">'
        f"{svg}"
        f'<div style="display:flex;justify-content:space-between;{label_style}margin-top:0.15rem;">'
        f"<span>{start_date}</span><span>{mid_date}</span><span>{end_date}</span>"
        f"</div>"
        f"</div>"
        f"</div>"
    )


def _chart_window_label(days: int) -> str:
    """A short, human window label for a chart card's header -- derived from the actual configured
    chart_days/extra_chart_days rather than hardcoded per card type, so a future MACRO_SERIES
    tweak (or a new cadence) doesn't silently go stale against the label text.
    """
    if days <= 10:
        return "1w"
    if days <= 45:
        return "1m"
    return "1y"


def _macro_card_source_html(tile: dict, window: str) -> str:
    """The "{icon} #Macro #{tag} ({window})" header line shared by every macro card type -- no
    leading-space artifact when a series has no icon configured (MACRO_SERIES' "icon" is optional).
    """
    icon_prefix = f"{tile['icon']} " if tile["icon"] else ""
    return f'<div class="keep-card-source">{icon_prefix}#Macro #{html.escape(tile["tag"])} ({window})</div>'


def _macro_direction_badge_html(narrative: dict | None) -> str:
    """The "↑ Bullish · High confidence" badge, shared by the weekly and monthly narrative cards --
    only present for DIRECTIONAL_SERIES (see finance.macro.DIRECTIONAL_SERIES/_MACRO_DIRECTION_ARROW)
    -- a rate/spread/index series' narrative dict simply has no "direction" key, same optional-field
    convention as fundamental_direction/earnings_direction.
    """
    if not narrative or "direction" not in narrative:
        return ""
    direction = narrative["direction"]
    arrow = _MACRO_DIRECTION_ARROW.get(direction, "➖")
    confidence_label = str(narrative.get("confidence", "low")).title()
    signal_note = "" if narrative.get("trade_worthy") else " (low signal)"
    return (
        f'<div class="keep-card-claim" style="font-size:1.05rem;">'
        f"{arrow} {direction.title()} · {confidence_label} confidence{signal_note}</div>"
    )


def _macro_headlines_back_html(narrative: dict | None, source: str) -> str | None:
    """Flip side: the actual headlines a narrative was grounded in (finance.macro's stored
    "headlines_used" -- already fetched/persisted, so this is free, no new news-fetch/LLM cost). Capped
    at 8 -- enough to substantiate the narrative without turning the back of a card-sized tile into
    a full article-list dump. None (no flip at all) if there's no narrative yet, or it genuinely
    found no headlines for that period.
    """
    headlines = (narrative or {}).get("headlines_used") or []
    if not headlines:
        return None
    back_bits = ['<div class="keep-card-summary-title">\U0001f4f0 Headlines behind this narrative</div>']
    back_bits += [
        f'<div class="keep-card-summary keep-card-risk-item">'
        f'[{html.escape(h["date"])}] ({html.escape(h["domain"])}) {html.escape(h["title"])}</div>'
        for h in headlines[:8]
    ]
    back_bits.append(f'<div class="keep-card-meta">{html.escape(source)} - {html.escape(narrative["date"])}</div>')
    return "".join(back_bits)


def _macro_narrative_card_html(tile: dict) -> str:
    """Card-sized trend chart + narrative, same Keep-card visual language as claims/fundamentals/
    earnings-calls (_flip_card_html/keep-card CSS) rather than a full-width native chart -- for a
    series where finance.macro attached recent "history" (see MACRO_SERIES' chart_days). Front
    face is always the *weekly* read (finance.macro's weekly "why," grounded in real news
    headlines from the last 7 days -- see finance.news_sources).

    For a series in MONTHLY_NARRATIVE_SERIES (has its own separate monthly-period narrative too --
    see finance.macro.refresh_macro_narrative's period="month"), the back face is the *monthly*
    read (_macro_monthly_back_html) instead of a raw headlines list -- weekly is "what moved this
    week," monthly is "is that part of a bigger move," and pairing those two time horizons on one
    card is more useful than a headlines dump would be (previously its own separate card,
    _macro_mini_chart_card_html, now folded in here). A series without a monthly narrative
    (pce/unemployment) keeps the old raw-headlines-list back instead, since there's nothing else to
    put there.
    """
    value_str = _MACRO_UNIT_FORMAT[tile["unit"]](tile["value"])
    delta_str = _MACRO_DELTA_FORMAT[tile["delta_format"]](tile["delta"]) if tile["delta"] is not None else "n/a"
    chart_svg = _macro_chart_html(tile.get("history") or [], tile["unit"])
    window = _chart_window_label(MACRO_SERIES[tile["key"]]["chart_days"])
    narrative = latest_narrative(tile["key"])
    source = _NARRATIVE_SOURCE_LABEL.get(tile["key"], "News")
    if narrative:
        icon = "\U0001f4f0" if narrative.get("has_clear_driver") else "\U0001f937"
        narrative_html = f'<div class="keep-card-context">{icon} {html.escape(narrative["narrative"])}</div>'
        meta_html = f'<div class="keep-card-meta">{html.escape(source)} - {html.escape(narrative["date"])}</div>'
    else:
        narrative_html = '<div class="keep-card-context">No weekly narrative generated yet.</div>'
        meta_html = f'<div class="keep-card-meta">{html.escape(tile["as_of"])}</div>'
    direction_html = _macro_direction_badge_html(narrative)
    card_body = (
        f"{_macro_card_source_html(tile, window)}"
        f'<div class="keep-card-claim">{html.escape(value_str)} '
        f'<span style="opacity:0.7;font-weight:400;font-size:0.8rem;">({html.escape(delta_str)} this week)</span></div>'
        f"{direction_html}"
        f"{chart_svg}"
        f"{narrative_html}"
        f"{meta_html}"
    )
    back_html = (
        _macro_monthly_back_html(tile) if tile["key"] in MONTHLY_NARRATIVE_SERIES
        else _macro_headlines_back_html(narrative, source)
    )
    return _flip_card_html(card_body, back_html)


def _macro_monthly_back_html(tile: dict) -> str:
    """The back face for _macro_narrative_card_html's combined weekly/monthly card -- the bigger-
    picture 30-day chart + monthly-period narrative (finance.macro.refresh_macro_narrative's
    period="month"). Only ever called for a tile in MONTHLY_NARRATIVE_SERIES (see that function).
    """
    chart_svg = _macro_chart_html(tile.get("history_extra") or [], tile["unit"])
    window = _chart_window_label(MACRO_SERIES[tile["key"]]["extra_chart_days"])
    narrative = latest_narrative(tile["key"], period="month")
    if narrative:
        icon = "\U0001f4f0" if narrative.get("has_clear_driver") else "\U0001f937"
        # The monthly narrative carries its own value/delta (finance.macro's
        # _value_delta_from_window, computed over the same 30-day window as the chart below) --
        # distinct from the tile's own "value"/"delta", which are always the *weekly* comparison.
        monthly_value_str = _MACRO_UNIT_FORMAT[tile["unit"]](narrative["value"])
        monthly_delta_str = (
            _MACRO_DELTA_FORMAT[tile["delta_format"]](narrative["delta"])
            if narrative.get("delta") is not None else "n/a"
        )
        value_html = (
            f'<div class="keep-card-claim">{html.escape(monthly_value_str)} '
            f'<span style="opacity:0.7;font-weight:400;font-size:0.8rem;">'
            f'({html.escape(monthly_delta_str)} over last 30 days)</span></div>'
        )
        narrative_html = f'<div class="keep-card-summary">{icon} {html.escape(narrative["narrative"])}</div>'
        source = _NARRATIVE_SOURCE_LABEL.get(tile["key"], "News")
        meta_html = f'<div class="keep-card-meta">{html.escape(source)} - {narrative["date"]}</div>'
    else:
        value_html = ""
        narrative_html = '<div class="keep-card-summary">No monthly narrative generated yet.</div>'
        meta_html = ""
    direction_html = _macro_direction_badge_html(narrative)
    return (
        f'<div class="keep-card-summary-title">\U0001f4c8 Bigger picture ({window})</div>'
        f"{value_html}"
        f"{direction_html}"
        f"{chart_svg}"
        f"{narrative_html}"
        f"{meta_html}"
    )


def _macro_chart_card_html(tile: dict) -> str:
    """Same trend chart as _macro_narrative_card_html, minus the narrative -- for a series with
    "history" (MACRO_SERIES' chart_days) but no entry in finance.macro.NARRATIVE_SERIES. No "no
    narrative yet" filler text, unlike
    _macro_narrative_card_html's fallback -- these series were never going to have one, so that
    text would read as a bug report rather than an honest "not generated yet" state. Used for both
    weekly-cadence series (credit spread, VIX, commodities) and monthly-cadence ones (PCE,
    unemployment) -- the window label just reflects each one's own configured chart_days.
    """
    value_str = _MACRO_UNIT_FORMAT[tile["unit"]](tile["value"])
    delta_str = _MACRO_DELTA_FORMAT[tile["delta_format"]](tile["delta"]) if tile["delta"] is not None else "n/a"
    chart_svg = _macro_chart_html(tile.get("history") or [], tile["unit"])
    window = _chart_window_label(MACRO_SERIES[tile["key"]]["chart_days"])
    period_label = "this week" if tile["cadence"] == "weekly" else "this reading"
    card_body = (
        f"{_macro_card_source_html(tile, window)}"
        f'<div class="keep-card-claim">{html.escape(value_str)} '
        f'<span style="opacity:0.7;font-weight:400;font-size:0.8rem;">({html.escape(delta_str)} {period_label})</span></div>'
        f"{chart_svg}"
        f'<div class="keep-card-meta">{html.escape(tile["as_of"])}</div>'
    )
    return _flip_card_html(card_body, None)


def _macro_weekly_card_id(tile: dict) -> str:
    """finance.read_state id for a weekly macro narrative card -- keyed by the narrative's own
    "date" (when that read was actually generated/last refreshed), not the tile's daily price
    date, so the card only counts as "new" again once the narrative itself changes, not every
    time the underlying price ticks. Falls back to the tile's own as_of if there's no narrative yet
    (still gives every card a stable id, even a plain/pre-narrative one).
    """
    narrative = latest_narrative(tile["key"])
    date = narrative["date"] if narrative else tile["as_of"]
    return _card_id(tile["key"], "macro_week", date)


def _macro_chart_card_id(tile: dict) -> str:
    return _card_id(tile["key"], "macro_chart", tile["as_of"])


def _render_recent_page() -> None:
    """A cross-universe "what's new" feed -- claims/fundamentals/earnings-calls (every tracked
    ticker) and macro narratives (every configured series), filtered to a recent time window,
    reusing every existing card renderer verbatim (_claim_card_html/_fundamental_card_html/
    _earnings_call_card_html/_macro_narrative_card_html) -- no new
    rendering logic, just a wider net across the whole universe plus a date filter over what's
    already on disk/cached (macro tiles come from the same @st.cache_data _cached_macro_snapshot
    the Macro page uses, so this costs no extra live FRED/yfinance fetches beyond that page's own).

    Thesis and critic-review state are deliberately excluded -- both reflect current synthesized
    state rather than a discrete new event, so "updated recently" wouldn't reliably mean "something
    new happened" the way a fresh claim/fundamental/earnings-call/macro read does.
    """
    upcoming_cards = _upcoming_earnings_cards(dt.date.today())
    if upcoming_cards:
        st.markdown("#### \U0001f4c5 Upcoming")
        _render_keep_card_grid(upcoming_cards, key="feed_upcoming")

    just_reported_cards = _recent_earnings_result_cards(dt.date.today())
    if just_reported_cards:
        st.markdown("#### \U0001f4ca Just Reported")
        _render_keep_card_grid(just_reported_cards, key="feed_just_reported")

    st.markdown("### Recent")
    st.caption("All claims, fundamentals, earnings calls, and macro narratives in one place.")

    # Same st.pills picking interaction, and the same labels/option text, as the Ticker page's own
    # Dates/Cards filters (see page_ticker) -- Dates is single-select (a card is either within the
    # window or not), Cards is multi-select (any combination can show at once).
    dates_label_col, dates_pills_col, cards_label_col, cards_pills_col = st.columns(
        [1, 2, 1, 4], vertical_alignment="center",
    )
    with dates_label_col:
        st.write("**Dates**")
    with dates_pills_col:
        window_label = st.pills(
            "Dates", options=["1d", "1w"], default="1d", selection_mode="single",
            key="recent_page_window", label_visibility="collapsed",
        )
    with cards_label_col:
        st.write("**Cards**")
    with cards_pills_col:
        types = st.pills(
            "Cards", options=["Claims", "Fundamentals", "Earnings Calls", "Macro"],
            default=["Claims", "Fundamentals", "Earnings Calls", "Macro"], selection_mode="multi",
            key="recent_page_types", label_visibility="collapsed",
        )

    window_label = window_label or "1d"  # single-select pills can be clicked off, leaving None
    types = types or []
    cutoff = dt.date.today() - dt.timedelta(days=1 if window_label == "1d" else 7)

    dated: list[tuple[dt.date, str, str]] = []
    summaries = _article_summaries() if "Claims" in types else {}
    for ticker in tracked_universe():
        if "Claims" in types:
            dated += [
                (c.created, c.id, _claim_card_html(c, summaries.get(c.source_link)))
                for c in load_claims(ticker) if c.created >= cutoff
            ]
        if "Fundamentals" in types:
            dated += [
                (
                    dt.date.fromisoformat(ev["date"]), _card_id(ticker, "fundamental", ev["date"]),
                    _fundamental_card_html(ev, ticker),
                )
                for ev in load_fundamental_history(ticker) if dt.date.fromisoformat(ev["date"]) >= cutoff
            ]
        if "Earnings Calls" in types:
            dated += [
                (
                    dt.date.fromisoformat(ev["date"]), _card_id(ticker, "earnings_call", ev["date"]),
                    _earnings_call_card_html(ev, ticker),
                )
                for ev in load_earnings_call_history(ticker) if dt.date.fromisoformat(ev["date"]) >= cutoff
            ]

    if "Macro" in types:
        # The monthly-period narrative is now the back face of the same weekly card (see
        # _macro_narrative_card_html), not a separate card/id -- no more separate MONTHLY_NARRATIVE_
        # SERIES branch here.
        for tile in _cached_macro_snapshot():
            weekly = latest_narrative(tile["key"], period="week")
            if weekly and dt.date.fromisoformat(weekly["date"]) >= cutoff:
                dated.append((
                    dt.date.fromisoformat(weekly["date"]), _macro_weekly_card_id(tile),
                    _macro_narrative_card_html(tile),
                ))

    if not dated:
        st.info("Nothing generated in this window for the selected card types.")
        return
    dated.sort(key=lambda item: item[0], reverse=True)
    # No st.caption count here -- the card_feed component shows its own live "N card(s)" line that
    # decrements the instant a card is dismissed, which a caption computed here couldn't do without
    # a full rerun (see components/card_feed/index.html's updateCountLine).
    _render_keep_card_grid([(cid, card_html) for _, cid, card_html in dated], key="feed_recent")


def _render_read_page() -> None:
    """Every card, of any type, across the whole app that the current user has ever explicitly
    marked read (finance.read_state) -- the counterpart to every other page hiding them by
    default. Same whole-universe traversal as _render_recent_page, just inverted (only ids that
    ARE in read_ids) and with no date cutoff, since a card can sit in the read pile indefinitely.
    Reuses the exact same card renderers, and passes show_read=True into the shared grid so it
    shows this page's cards rather than hiding them (the grid's default is "hide read cards" --
    this is the one page where that default is flipped).
    """
    st.markdown("### Read")
    st.caption("Every card you've marked as read, across the whole app.")

    read_ids = read_state.read_ids(_CURRENT_USER)
    if not read_ids:
        st.info("No cards marked read yet.")
        return

    dated: list[tuple[dt.date, str, str]] = []
    summaries = _article_summaries()
    for ticker in tracked_universe():
        dated += [
            (c.created, c.id, _claim_card_html(c, summaries.get(c.source_link)))
            for c in load_claims(ticker) if c.id in read_ids
        ]
        for ev in load_fundamental_history(ticker):
            cid = _card_id(ticker, "fundamental", ev["date"])
            if cid in read_ids:
                dated.append((dt.date.fromisoformat(ev["date"]), cid, _fundamental_card_html(ev, ticker)))
        for ev in load_earnings_call_history(ticker):
            cid = _card_id(ticker, "earnings_call", ev["date"])
            if cid in read_ids:
                dated.append((dt.date.fromisoformat(ev["date"]), cid, _earnings_call_card_html(ev, ticker)))
        # Reminder/result cards are only ever *shown* within their own freshness window (see
        # _upcoming_earnings_cards/_recent_earnings_result_cards), but their read-mark should still
        # be findable here indefinitely, same as every other card type -- scanned by id rather than
        # re-applying that window, since a card marked read while fresh should stay findable after
        # its window has long since passed. Only bothers with the precise, tz-aware
        # _next_earnings_info live call when finance.data's own disk-cached history (already fetched
        # below, and kept warm by run_loop_a) actually has a not-yet-reported row -- skips a live
        # per-ticker yfinance round-trip for every ticker with nothing scheduled.
        ticker_earnings_history = get_earnings_history(ticker)
        has_upcoming = (
            ticker_earnings_history["reported_eps"].isna()
            & (ticker_earnings_history["earnings_date"].dt.date >= dt.date.today())
        ).any()
        if has_upcoming:
            info = _next_earnings_info(ticker)
            if info is not None:
                cid = _earnings_reminder_card_id(ticker, info["when"])
                if cid in read_ids:
                    dated.append((
                        info["when"].date(), cid,
                        _earnings_reminder_card_html(ticker, info["when"], info["eps_estimate"], dt.date.today()),
                    ))
        for _, row in ticker_earnings_history.dropna(subset=["reported_eps", "surprise_pct"]).iterrows():
            report_date = row["earnings_date"].date()
            cid = _earnings_result_card_id(ticker, report_date.isoformat())
            if cid in read_ids:
                dated.append((report_date, cid, _earnings_result_card_html(ticker, row, dt.date.today())))

    # The monthly-period narrative is now the back face of the same weekly card (see
    # _macro_narrative_card_html), not a separate card/id -- one lookup per tile, not two.
    for tile in _cached_macro_snapshot():
        weekly_id = _macro_weekly_card_id(tile)
        if weekly_id in read_ids:
            weekly = latest_narrative(tile["key"])
            if weekly:
                dated.append((dt.date.fromisoformat(weekly["date"]), weekly_id, _macro_narrative_card_html(tile)))

    if not dated:
        st.info("No cards marked read yet.")
        return
    dated.sort(key=lambda item: item[0], reverse=True)
    _render_keep_card_grid([(cid, card_html) for _, cid, card_html in dated], show_read=True, key="feed_read")


def _render_favorites_page() -> None:
    """Every card, of any type, across the whole app that the current user has ever swiped right/
    starred (finance.read_state's separate favorite_ids -- independent of read status, see that
    module's own docstring). Same whole-universe traversal as _render_read_page, just filtered by
    favorite_ids instead of read_ids, and passes show_favorites=True into the shared grid so it
    shows this page's cards (with an "Unfavorite" action) rather than the grid's own default of
    hiding already-read ones.
    """
    st.markdown("### Favorites")
    st.caption("Every card you've starred, across the whole app.")

    favorite_ids = read_state.favorite_ids(_CURRENT_USER)
    if not favorite_ids:
        st.info("No cards favorited yet -- swipe a card right to star it.")
        return

    dated: list[tuple[dt.date, str, str]] = []
    summaries = _article_summaries()
    for ticker in tracked_universe():
        dated += [
            (c.created, c.id, _claim_card_html(c, summaries.get(c.source_link)))
            for c in load_claims(ticker) if c.id in favorite_ids
        ]
        for ev in load_fundamental_history(ticker):
            cid = _card_id(ticker, "fundamental", ev["date"])
            if cid in favorite_ids:
                dated.append((dt.date.fromisoformat(ev["date"]), cid, _fundamental_card_html(ev, ticker)))
        for ev in load_earnings_call_history(ticker):
            cid = _card_id(ticker, "earnings_call", ev["date"])
            if cid in favorite_ids:
                dated.append((dt.date.fromisoformat(ev["date"]), cid, _earnings_call_card_html(ev, ticker)))
        # See the matching comment in _render_read_page -- same reasoning, favorite_ids instead.
        ticker_earnings_history = get_earnings_history(ticker)
        has_upcoming = (
            ticker_earnings_history["reported_eps"].isna()
            & (ticker_earnings_history["earnings_date"].dt.date >= dt.date.today())
        ).any()
        if has_upcoming:
            info = _next_earnings_info(ticker)
            if info is not None:
                cid = _earnings_reminder_card_id(ticker, info["when"])
                if cid in favorite_ids:
                    dated.append((
                        info["when"].date(), cid,
                        _earnings_reminder_card_html(ticker, info["when"], info["eps_estimate"], dt.date.today()),
                    ))
        for _, row in ticker_earnings_history.dropna(subset=["reported_eps", "surprise_pct"]).iterrows():
            report_date = row["earnings_date"].date()
            cid = _earnings_result_card_id(ticker, report_date.isoformat())
            if cid in favorite_ids:
                dated.append((report_date, cid, _earnings_result_card_html(ticker, row, dt.date.today())))

    # The monthly-period narrative is now the back face of the same weekly card (see
    # _macro_narrative_card_html), not a separate card/id -- one lookup per tile, not two.
    for tile in _cached_macro_snapshot():
        weekly_id = _macro_weekly_card_id(tile)
        if weekly_id in favorite_ids:
            weekly = latest_narrative(tile["key"])
            if weekly:
                dated.append((dt.date.fromisoformat(weekly["date"]), weekly_id, _macro_narrative_card_html(tile)))

    if not dated:
        st.info("No cards favorited yet -- swipe a card right to star it.")
        return
    dated.sort(key=lambda item: item[0], reverse=True)
    _render_keep_card_grid(
        [(cid, card_html) for _, cid, card_html in dated], show_favorites=True, key="feed_favorites",
    )


_DISCOVERY_NAME_STRIP_RE = re.compile(r"[^a-z0-9]+")


def _normalize_discovery_name(name: str) -> str:
    """Groups "OpenAI"/"openai"/"Open AI" (case/spacing/punctuation variants a non-deterministic
    LLM call is prone to producing across different articles/runs) under one key -- deliberately
    simple (no fuzzy/embedding matching) for a first pass; see finance.newsloop's own docstring re:
    grouping being a "later problem." Strips everything but letters/digits, so "Zhipu AI" and
    "Zhipu-AI" collide too, which is the intent, not a bug.
    """
    return _DISCOVERY_NAME_STRIP_RE.sub("", name.lower())


def _group_discovery_candidates(candidates: list[dict]) -> list[dict]:
    """Groups finance.newsloop.load_discovery_candidates' flat entries by (type,
    _normalize_discovery_name) -- type is part of the key so a company name and an unrelated theme
    tag that happen to normalize to the same string never merge into one group (a theme's "name" is
    a short reused tag like "oil"/"geopolitics", so a real collision with some company's name is
    the one case worth guarding against even though it'd be rare). Newest-mention-count first (ties
    broken by most recent mention) -- a name/tag several different sources keep independently
    flagging is a much stronger signal than a single one-off mention, so that's what surfaces
    first. Each group: "type", "display_name" (the most common raw spelling seen, so the page shows
    real article wording rather than the stripped normalization key), "ticker_guess" (companies
    only -- the most common non-null guess, or None if the LLM never offered one, typical for a
    private company; always None for a theme), "count"/"source_count", and "entries" (every
    underlying mention, newest first).
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in candidates:
        kind = c.get("type", TYPE_COMPANY_DISCOVERY)
        groups.setdefault((kind, _normalize_discovery_name(c["name"])), []).append(c)

    result = []
    for (kind, _key), entries in groups.items():
        entries = sorted(entries, key=lambda c: c["date"], reverse=True)
        display_name = Counter(c["name"] for c in entries).most_common(1)[0][0]
        ticker_guesses = Counter(c["ticker_guess"] for c in entries if c.get("ticker_guess"))
        result.append({
            "type": kind,
            "display_name": display_name,
            "ticker_guess": ticker_guesses.most_common(1)[0][0] if ticker_guesses else None,
            "count": len(entries),
            "source_count": len({c["source"] for c in entries}),
            "entries": entries,
        })
    result.sort(key=lambda g: (g["count"], g["entries"][0]["date"]), reverse=True)
    return result


def _discovery_card_id(kind: str, display_name: str) -> str:
    return _card_id("discovery", kind, _normalize_discovery_name(display_name))


def _discard_discovery_group(card_id: str) -> None:
    """"Discard" for a Discovery card -- unlike every other card's read/favorite actions, this
    permanently removes the group's underlying entries from candidates.json rather than setting a
    read_state flag (there's no "undo" page for this, matching the swipe gesture's own permanence).
    card_id is an opaque hash (read_state.card_id), not reversible on its own, so this recomputes
    the same groups _render_discovery_page just rendered and matches by id to find which (type,
    normalized-name) key to remove -- same "recompute and compare" trick other card ids in this
    app already rely on instead of trying to invert a hash.
    """
    candidates = load_discovery_candidates()
    groups = _group_discovery_candidates(candidates)
    target = next((g for g in groups if _discovery_card_id(g["type"], g["display_name"]) == card_id), None)
    if target is None:
        return
    discard_key = (target["type"], _normalize_discovery_name(target["display_name"]))
    remaining = [
        c for c in candidates
        if (c.get("type", TYPE_COMPANY_DISCOVERY), _normalize_discovery_name(c["name"])) != discard_key
    ]
    save_discovery_candidates(remaining)


_CARD_ACTIONS["discard"] = _discard_discovery_group


_DISCOVERY_MENTIONS_SHOWN = 10  # total across both faces -- see _discovery_card_html


def _mention_item_html(e: dict) -> str:
    return (
        f'<div class="keep-card-summary keep-card-risk-item">'
        f'[{html.escape(e["date"])}] ({html.escape(e["source"])}) {html.escape(e["why"])}</div>'
    )


def _discovery_card_html(group: dict) -> str:
    """Same keep-card visual language as every other card type (_flip_card_html/claim/fundamental/
    earnings-call cards) rather than a plain st.container -- front face is the at-a-glance summary
    (name, mention/source counts) plus the newer half of recent mentions, back face gets the older
    half, same "front = glance, back = full detail" split _fundamental_card_html's risks and
    _macro_narrative_card_html's headlines already use. Capped at _DISCOVERY_MENTIONS_SHOWN total
    (oldest beyond that just noted as a count, not rendered) and split evenly across both faces --
    unlike those other card types, a discovery group's mention count is genuinely unbounded (the
    same name/tag can keep getting flagged for months), and the two faces share one CSS grid cell
    sized to whichever is taller (components/card_feed/index.html's .keep-flip-inner) -- so an
    all-on-the-back list of everything grew that shared cell (and thus the whole card, front
    included, even before it's ever flipped) without bound. A theme group never has a ticker tag
    (see _group_discovery_candidates) and gets "#Theme" instead of "#Discovery" as its source tag,
    so the two kinds read as distinct at a glance despite sharing this one renderer.
    """
    is_theme = group["type"] == TYPE_THEME_DISCOVERY
    tag = f" #{html.escape(group['ticker_guess'])}" if group["ticker_guess"] else ""
    source_label = "Theme" if is_theme else "Discovery"
    entries = group["entries"]  # newest first
    shown = entries[:_DISCOVERY_MENTIONS_SHOWN]
    split = (len(shown) + 1) // 2
    front_entries, back_entries = shown[:split], shown[split:]
    omitted = len(entries) - len(shown)

    card_body = (
        f'<div class="keep-card-source">\U0001f50d #{source_label}{tag}</div>'
        f'<div class="keep-card-claim">{html.escape(group["display_name"])}</div>'
        + "".join(_mention_item_html(e) for e in front_entries)
        + f'<div class="keep-card-meta">{group["count"]} mention(s) · {group["source_count"]} source(s) · '
        f'{html.escape(entries[0]["date"])}</div>'
    )
    back_bits = ['<div class="keep-card-summary-title">\U0001f4f0 More mentions</div>']
    back_bits += [_mention_item_html(e) for e in back_entries]
    if omitted:
        back_bits.append(f'<div class="keep-card-summary">+{omitted} more not shown</div>')
    return _flip_card_html(card_body, "".join(back_bits))


def _render_discovery_page() -> None:
    """Companies extract_event flagged as discussed in real depth by an already-fetched article,
    but NOT in the tracked universe (finance.newsloop's "other_companies_mentioned" -- see that
    module's docstring for why this exists and its current limits, namely: only catches candidates
    riding along inside articles that already passed the tracked-company pre-filter for an
    unrelated reason). Grouped by name (_group_discovery_candidates) so a name several different
    articles/sources independently flagged surfaces as one card with every mention, not scattered
    one-off entries -- the accumulated count/source-diversity IS the signal here, there's
    deliberately no LLM synthesis step yet (see the module docstring's "later problem" framing).

    Rendered through the same _render_keep_card_grid every other card type uses, id'd by
    _discovery_card_id (a stable hash of (type, normalized name)) -- but with primary_action=
    "discard" instead of the usual read/favorite: swiping/tapping a card here permanently deletes
    its underlying entries from candidates.json (_discard_discovery_group), not a read_state flag.
    No read/favorite tracking at all for this page -- read_state has nothing to do with "should this
    still exist," and there's no undo once discarded, matching the swipe gesture's own permanence.
    """
    st.markdown("### Discovery")
    st.caption(
        "Companies and themes outside your tracked coverage that your news sources discussed in "
        "real depth -- not extracted claims, just a signal worth a closer look. Grouped by name, "
        "most-mentioned first. Swipe or tap to discard permanently."
    )
    candidates = load_discovery_candidates()
    if not candidates:
        st.info("No discovery candidates recorded yet -- these accumulate as run_loop_a processes articles.")
        return

    groups = _group_discovery_candidates(candidates)
    _render_keep_card_grid(
        [(_discovery_card_id(g["type"], g["display_name"]), _discovery_card_html(g)) for g in groups],
        primary_action="discard", key="feed_discovery",
    )


def _render_macro_page() -> None:
    """The global macro dashboard -- deterministic FRED/yfinance stat tiles (finance.macro), not
    tied to any ticker. See finance.macro's own module docstring for why nothing here costs an
    LLM call or carries hallucination risk, unlike every other card type in this app.
    """
    st.markdown("### Macro")
    st.caption("Data source: Finnhub/GNews/FRED/Yfinance")
    if st.button("Refresh now", key="macro_refresh_btn"):
        _cached_macro_snapshot.clear()
        st.rerun()

    tiles = _cached_macro_snapshot()
    if not tiles:
        st.info("Macro data unavailable right now -- FRED/yfinance fetch failed for every series. Try again shortly.")
        return

    # Sidebar sub-selection (see page_ticker's Macro expander) -- clicking a specific series (e.g.
    # "Gold") narrows this page down to just that series' own card(s); clicking "Macro" itself
    # (or the "All" entry) leaves this unset, showing every series exactly as before that feature
    # existed. Filtered up front so every grouping/splitting step below stays unchanged either way.
    focused_series = st.session_state.get("ticker_page_macro_series")
    if focused_series:
        tiles = [t for t in tiles if t["key"] == focused_series]
        if not tiles:
            st.info(f"No data available right now for {focused_series!r}.")
            return

    weekly = [t for t in tiles if t["cadence"] == "weekly"]
    monthly = [t for t in tiles if t["cadence"] == "monthly"]
    weekly_charts = [t for t in weekly if "history" in t]
    weekly_tiles = [t for t in weekly if "history" not in t]
    monthly_charts = [t for t in monthly if "history" in t]
    monthly_tiles = [t for t in monthly if "history" not in t]
    # Only a series in finance.macro.NARRATIVE_SERIES has a news-grounded weekly narrative --
    # everything else with a chart gets the plain trend card, no "no narrative yet" filler text
    # (see _macro_chart_card_html's own docstring). Applies to both weekly- and monthly-cadence
    # charts -- a monthly series (PCE, unemployment) can have a narrative just as well as a weekly
    # one, the narrative's own "week" field just means "the week this read was generated," not
    # anything about the underlying series' own release cadence.
    weekly_narrative_charts = [t for t in weekly_charts if t["key"] in NARRATIVE_SERIES]
    weekly_plain_charts = [t for t in weekly_charts if t["key"] not in NARRATIVE_SERIES]
    monthly_narrative_charts = [t for t in monthly_charts if t["key"] in NARRATIVE_SERIES]
    monthly_plain_charts = [t for t in monthly_charts if t["key"] not in NARRATIVE_SERIES]
    # A weekly-cadence series' "bigger picture" 1-month view (history_extra) used to be its own
    # separate mini-chart card down in the Monthly section -- it's now the back face of the same
    # weekly card instead (see _macro_narrative_card_html/_macro_monthly_back_html), so there's
    # nothing left to render for it separately here.
    if weekly_charts:
        st.write("**Weekly**")
        _render_keep_card_grid(
            [(_macro_weekly_card_id(t), _macro_narrative_card_html(t)) for t in weekly_narrative_charts]
            + [(_macro_chart_card_id(t), _macro_chart_card_html(t)) for t in weekly_plain_charts],
            key="feed_macro_weekly",
        )
    if weekly_tiles:
        if not weekly_charts:
            st.write("**Weekly**")
        _render_macro_tiles(weekly_tiles)
    if monthly_charts or monthly_tiles:
        st.write("**Monthly**")
        if monthly_charts:
            _render_keep_card_grid(
                [(_macro_weekly_card_id(t), _macro_narrative_card_html(t)) for t in monthly_narrative_charts]
                + [(_macro_chart_card_id(t), _macro_chart_card_html(t)) for t in monthly_plain_charts],
                key="feed_macro_monthly",
            )
        if monthly_tiles:
            _render_macro_tiles(monthly_tiles)


@st.cache_data(ttl=3600)
def _cached_macro_snapshot() -> list[dict]:
    """Cached at the Streamlit layer (not finance.macro itself, which stays framework-agnostic
    like every other finance.* module) -- macro_snapshot() does up to 9 live network fetches, too
    slow to redo on every Streamlit rerun (a rerun happens on nearly every widget interaction).
    1-hour TTL, since these are daily-cadence-or-slower series -- nothing meaningfully changes
    within an hour. The Refresh button above clears this cache directly for an on-demand update.
    """
    return macro_snapshot()


def page_ticker() -> None:
    """One ticker at a time: pick from config_loop_a's full tracked universe
    in the sidebar (every configured ticker, even ones with no claims/thesis
    yet), then see that ticker's thesis as a summary card and every claim
    behind it as a Google-Keep-style card grid. A focused alternative to
    Research's browse-every-ticker list, for when you already know which
    company you want to dig into.
    """
    universe = tracked_universe()
    if not universe:
        st.info("No tickers configured -- see config_loop_a.json.")
        return

    options = sorted(universe, key=lambda t: universe[t])  # by display name
    sectors = ticker_sectors()
    grouped: dict[str, list[str]] = {}
    for t in options:
        grouped.setdefault(sectors.get(t, "Other"), []).append(t)

    # Fixed display order (not alphabetical) -- any sector not listed here (e.g. a new one used in
    # config_loop_a.json's "universe" entries) still shows, just appended after these.
    _SECTOR_ORDER = [
        "Semiconductors", "Big Tech", "AI Infrastructure", "Futuristic",
        "Energy", "Commodities", "Crypto", "Defense & Aerospace",
    ]
    ordered_sectors = [s for s in _SECTOR_ORDER if s in grouped]
    ordered_sectors += [s for s in grouped if s not in _SECTOR_ORDER]

    if "ticker_page_selected_ticker" not in st.session_state:
        st.session_state["ticker_page_selected_ticker"] = options[0]
    if "ticker_page_view" not in st.session_state:
        st.session_state["ticker_page_view"] = "recent"
    with st.sidebar:
        is_recent_view = st.session_state["ticker_page_view"] == "recent"
        st.button(
            "\U0001f195 Recent", key="ticker_page_recent_btn",
            type="primary" if is_recent_view else "secondary",
            on_click=_select_recent_view, width="stretch",
        )
        is_read_view = st.session_state["ticker_page_view"] == "read"
        st.button(
            "\U0001f4d6 Read", key="ticker_page_read_btn",
            type="primary" if is_read_view else "secondary",
            on_click=_select_read_view, width="stretch",
        )
        is_favorites_view = st.session_state["ticker_page_view"] == "favorites"
        st.button(
            "⭐ Favorites", key="ticker_page_favorites_btn",
            type="primary" if is_favorites_view else "secondary",
            on_click=_select_favorites_view, width="stretch",
        )
        is_discovery_view = st.session_state["ticker_page_view"] == "discovery"
        st.button(
            "\U0001f50d Discovery", key="ticker_page_discovery_btn",
            type="primary" if is_discovery_view else "secondary",
            on_click=_select_discovery_view, width="stretch",
        )
        is_macro_view = st.session_state["ticker_page_view"] == "macro"
        focused_series = st.session_state.get("ticker_page_macro_series")
        with st.expander("\U0001f30d Macro", expanded=True):
            st.button(
                "All", key="ticker_page_macro_all_btn",
                type="primary" if (is_macro_view and not focused_series) else "secondary",
                on_click=_select_macro_view, width="stretch",
            )
            # 2 columns, not 3 (unlike the sector ticker grids below, which stay 3 -- short
            # symbols like "AAPL" fit fine there) -- the longest tags here ("Unemployment",
            # "HY-Spread") wrap onto two lines at 1/3 sidebar width, so this group needs the
            # extra room a 2-column layout gives every button.
            cols = st.columns(2)
            for i, (series_key, config) in enumerate(MACRO_SERIES.items()):
                is_selected = is_macro_view and focused_series == series_key
                label = f'{config.get("icon", "")} {config["tag"]}'.strip()
                cols[i % 2].button(
                    label, key=f"ticker_page_macro_btn_{series_key}",
                    type="primary" if is_selected else "secondary",
                    on_click=_select_macro_view, args=(series_key,),
                    width="stretch",
                )
        st.divider()
        # Sector-grouped expanders of plain buttons, same visual structure the old sidebar's
        # QUICK_PICK_CATEGORIES pickers used -- buttons instead of st.pills specifically because
        # exactly one ticker must be selected *across every sector at once*; st.pills' selection
        # is per-widget, so N independent single-select pills groups (one per sector) can't share
        # one global "currently selected" without fighting each other. A button's on_click just
        # writes the one shared session_state key directly, so highlighting always agrees.
        is_ticker_view = st.session_state["ticker_page_view"] == "ticker"
        for sector in ordered_sectors:
            with st.expander(sector, expanded=True):
                cols = st.columns(3)
                for i, t in enumerate(grouped[sector]):
                    # Gated on is_ticker_view too, not just a ticker-name match -- otherwise
                    # whichever ticker was last selected stayed highlighted "primary" even while
                    # Recent/Read/Macro was the active page, which read as if a ticker were still
                    # selected when it wasn't.
                    is_selected = is_ticker_view and t == st.session_state["ticker_page_selected_ticker"]
                    cols[i % 3].button(
                        t, key=f"ticker_sector_btn_{t}",
                        type="primary" if is_selected else "secondary",
                        on_click=_select_page_ticker, args=(t,),
                        width="stretch",
                    )
        st.divider()
    _maybe_close_sidebar_on_mobile()
    _render_card_display_settings()
    if st.session_state["ticker_page_view"] == "recent":
        _render_recent_page()
        return
    if st.session_state["ticker_page_view"] == "read":
        _render_read_page()
        return
    if st.session_state["ticker_page_view"] == "favorites":
        _render_favorites_page()
        return
    if st.session_state["ticker_page_view"] == "discovery":
        _render_discovery_page()
        return
    if st.session_state["ticker_page_view"] == "macro":
        _render_macro_page()
        return

    selected_ticker = st.session_state["ticker_page_selected_ticker"]

    tt = load_ticker_thesis(selected_ticker)
    all_claims = load_claims(selected_ticker)
    # Independent of tt/all_claims -- a fundamental snapshot can exist for a ticker with no claims
    # or thesis at all (finance.thesis.refresh_fundamentals doesn't need either), so this is
    # its own guard input rather than gated behind `if tt is not None` below.
    fundamental_history = load_fundamental_history(selected_ticker)
    earnings_call_history = load_earnings_call_history(selected_ticker)

    if tt is None and not all_claims and not fundamental_history and not earnings_call_history:
        st.info(f"No research yet for {selected_ticker} -- run Loop A to populate this.")
        return

    if tt is not None:
        arrow = _DIRECTION_ARROW.get(tt.direction, "➖")
        logo_html = _ticker_logo_html(selected_ticker)
        st.markdown(
            f'<h3>{logo_html}{selected_ticker}  ·  {arrow}  ·  conf={tt.confidence:.0%}</h3>',
            unsafe_allow_html=True,
        )
        st.write(f"**Thesis:** {_md(tt.thesis)}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Direction", tt.direction)
        m2.metric("Confidence", f"{tt.confidence:.0%}")
        m3.metric("Expected return", f"{tt.expected_return_pct:+.1f}%")
        m4.metric("Horizon", f"{tt.expected_horizon_days}d")
        # No Catalysts/Invalidation here -- already shown per-aggregation inside the Theses
        # dialog below (finance.thesis.aggregate_claims's own catalysts/invalidation),
        # same reasoning Research's cards already apply.

        aggregated_events = [ev for ev in tt.history if ev["event"] == "aggregated"]
        critic_events = [ev for ev in tt.history if ev["event"] == "critic"]
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if critic_events and st.button(
                f"Critic ({len(critic_events)})", key=f"ticker_page_critic_btn_{selected_ticker}"
            ):
                _critic_dialog(selected_ticker, critic_events)
        with btn_col2:
            if aggregated_events and st.button(
                f"Theses ({len(aggregated_events)})", key=f"ticker_page_agg_btn_{selected_ticker}"
            ):
                _aggregation_history_dialog(selected_ticker, aggregated_events)
    else:
        st.info(f"{selected_ticker} has claims but no synthesized thesis yet.")

    st.divider()
    all_sources = sorted({c.source or "unknown" for c in all_claims})
    # Every filter widget's key includes selected_ticker -- otherwise switching tickers keeps the
    # previous ticker's widget state (e.g. 2 sources selected out of its 5), which for a different
    # ticker's different source list can silently filter down to zero claims. A fresh key per
    # ticker makes each one start over at its own defaults: all sources, All dates, All importance,
    # both card types. Three lines total: label beside its own pills (not above) -- Sources on its
    # own line, Dates+Importance sharing the second, Cards on its own third line -- still pills
    # throughout, same picking interaction.
    sources_label_col, sources_pills_col = st.columns([1, 6], vertical_alignment="center")
    with sources_label_col:
        st.write("**Sources**")
    with sources_pills_col:
        selected_sources = st.pills(
            "Sources", options=all_sources, default=all_sources, selection_mode="multi",
            key=f"ticker_page_sources_{selected_ticker}", label_visibility="collapsed",
        )

    dates_label_col, dates_pills_col, importance_label_col, importance_pills_col = st.columns(
        [1, 2, 1, 2], vertical_alignment="center",
    )
    with dates_label_col:
        st.write("**Dates**")
    with dates_pills_col:
        date_filter = st.pills(
            "Dates", options=["1d", "1w", "1m", "All"], default="All",
            selection_mode="single", key=f"ticker_page_date_filter_{selected_ticker}",
            label_visibility="collapsed",
        )
    with importance_label_col:
        st.write("**Importance**")
    with importance_pills_col:
        importance_filter = st.pills(
            "Importance", options=["8+", "6+", "4+", "All"], default="All",
            selection_mode="single", key=f"ticker_page_importance_filter_{selected_ticker}",
            label_visibility="collapsed",
        )

    cards_label_col, cards_pills_col = st.columns([1, 6], vertical_alignment="center")
    with cards_label_col:
        st.write("**Cards**")
    with cards_pills_col:
        # Sources/Importance above only ever apply to claims (fundamental/earnings-call snapshots
        # have neither), but Dates applies to all three -- so unchecking a type here is the only
        # way to hide it, not the other filters.
        selected_card_types = st.pills(
            "Cards", options=["Claims", "Fundamentals", "Earnings Calls"],
            default=["Claims", "Fundamentals", "Earnings Calls"],
            selection_mode="multi", key=f"ticker_page_card_types_{selected_ticker}",
            label_visibility="collapsed",
        )

    date_filter = date_filter or "All"  # single-select pills can be clicked off, leaving None
    cutoff = None
    if date_filter != "All":
        lookback_days = {"1d": 1, "1w": 7, "1m": 30}[date_filter]
        cutoff = dt.date.today() - dt.timedelta(days=lookback_days)

    visible_claims: list = []
    if "Claims" in selected_card_types:
        visible_claims = [c for c in all_claims if (c.source or "unknown") in selected_sources]
        if cutoff is not None:
            visible_claims = [c for c in visible_claims if c.created >= cutoff]
        importance_filter = importance_filter or "All"
        if importance_filter != "All":
            threshold = int(importance_filter.rstrip("+"))
            visible_claims = [c for c in visible_claims if c.importance >= threshold]

    visible_fundamentals: list = []
    if "Fundamentals" in selected_card_types:
        visible_fundamentals = fundamental_history
        if cutoff is not None:
            visible_fundamentals = [
                ev for ev in visible_fundamentals if dt.date.fromisoformat(ev["date"]) >= cutoff
            ]

    visible_earnings_calls: list = []
    if "Earnings Calls" in selected_card_types:
        visible_earnings_calls = earnings_call_history
        if cutoff is not None:
            visible_earnings_calls = [
                ev for ev in visible_earnings_calls if dt.date.fromisoformat(ev["date"]) >= cutoff
            ]

    st.subheader(f"Cards ({len(visible_claims) + len(visible_fundamentals) + len(visible_earnings_calls)})")
    _render_mixed_keep_cards(visible_claims, visible_fundamentals, visible_earnings_calls, selected_ticker)


def page_home() -> None:
    """Research + This Week + Portfolio -- the daily Loop A workflow: browse
    theses, check what's new this week, manage paper-trading portfolios.
    Deliberately no sidebar (see picked_tickers/tickers_input's module-level
    comment) -- none of these three need a ticker picker, so this page stays
    uncluttered.
    """
    tab_names = ["Research", "This Week"]
    if not HOSTED:
        tab_names.append("Portfolio")
    _render_equal_width_tab_css("home_active_tab", len(tab_names))
    active_tab = st.segmented_control(
        "Section", options=tab_names, default=tab_names[0], required=True,
        key="home_active_tab", label_visibility="collapsed",
    )
    if active_tab == "Research":
        render_research_tab()
    elif active_tab == "This Week":
        render_weekly_tab()
    elif active_tab == "Portfolio":
        render_portfolio_tab()


def page_explore() -> None:
    """Every other tool -- comparisons, correlations, momentum, factor
    ranking, calendars, backtesting/simulation -- all of which read from the
    sidebar's ticker picker (picked_tickers/tickers_input, set here, read as
    plain module globals by the render_*_tab functions below), so the picker
    lives on this page only rather than cluttering Home.
    """
    global picked_tickers, tickers_input
    with st.sidebar:
        with st.expander("Return calculator", expanded=False):
            calc_col0, calc_col1, calc_col2 = st.columns([1, 1, 1])
            with calc_col0:
                calc_ticker = st.text_input("Ticker", value="AAPL", key="calc_ticker").strip().upper()
            with calc_col1:
                calc_start = st.date_input(
                    "Start", value=dt.date(2026, 1, 1), max_value=dt.date.today(), key="calc_start"
                )
            with calc_col2:
                calc_end = st.date_input("End", value=dt.date.today(), max_value=dt.date.today(), key="calc_end")

            if calc_ticker and calc_start <= calc_end:
                calc_fetch_end = calc_end + dt.timedelta(days=1)  # yfinance's end is exclusive
                calc_prices = get_prices([calc_ticker], start=calc_start.isoformat(), end=calc_fetch_end.isoformat())
                calc_prices = calc_prices.loc[
                    (calc_prices.index >= pd.Timestamp(calc_start)) & (calc_prices.index <= pd.Timestamp(calc_end))
                ]
                if calc_ticker not in calc_prices.columns or calc_prices[calc_ticker].dropna().empty:
                    st.caption(f"No data for {calc_ticker} in that range.")
                else:
                    series = calc_prices[calc_ticker].dropna()
                    calc_return = series.iloc[-1] / series.iloc[0] - 1
                    st.metric(f"{calc_ticker} return", f"{calc_return:+.2%}")
            elif calc_start > calc_end:
                st.caption("Start date must be on/before end date.")

        picked: set[str] = set()
        for category, options in QUICK_PICK_CATEGORIES.items():
            with st.expander(category, expanded=True):
                picked.update(
                    st.pills(
                        category,
                        options=list(options.values()),
                        selection_mode="multi",
                        default=[],
                        key=f"pick_{category}",
                        label_visibility="collapsed",
                    )
                )
        picked_tickers = picked
        tickers_input = st.text_input(
            "Other tickers (comma-separated)", value=", ".join(load_custom_tickers()),
            help="Persisted -- still here next time you open the app.",
        )
        _typed_custom_tickers = sorted({t.strip().upper() for t in tickers_input.split(",") if t.strip()})
        if _typed_custom_tickers != load_custom_tickers():
            save_custom_tickers(_typed_custom_tickers)

    tab_names = []
    if not HOSTED:
        tab_names.append("Sim")
    tab_names += [
        "Compare", "Corr", "Mom", "buy-dip", "Calendar",
        "PEAD", "Insider", "Inst'", "Rank", "LT data",
    ]
    _render_equal_width_tab_css("explore_active_tab", len(tab_names))
    active_tab = st.segmented_control(
        "Section", options=tab_names, default=tab_names[0], required=True,
        key="explore_active_tab", label_visibility="collapsed",
    )
    if active_tab == "Sim":
        render_simulation_tab()
    elif active_tab == "Compare":
        render_compare_tab()
    elif active_tab == "Corr":
        render_correlations_tab()
    elif active_tab == "Mom":
        render_momentum_tab()
    elif active_tab == "buy-dip":
        render_dip_tab()
    elif active_tab == "Calendar":
        render_calendar_tab()
    elif active_tab == "PEAD":
        render_pead_tab()
    elif active_tab == "Insider":
        render_insider_tab()
    elif active_tab == "Inst'":
        render_ownership_tab()
    elif active_tab == "Rank":
        render_ranking_tab()
    elif active_tab == "LT data":
        render_panel_tab()


page_ticker()
