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
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from finance.news import NEWS_SOURCES, get_all_news
from finance.newsloop import CONCENTRATION_PROFILES, HORIZON_PROFILES, RISK_PROFILES, RULE_NAME
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
from finance.summarize import get_cached_summary, has_api_key, summarize_article
from finance.thesis import open_positions
from finance.tickerthesis import list_tickers_with_thesis, load_ticker_thesis
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

st.set_page_config(page_title="Market comparisons", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    "<style>"
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
    # Below 768px, keep Streamlit's native header/sidebar controls untouched --
    # stExpandSidebarButton (the only way to reopen a collapsed sidebar on a
    # phone) is rendered *inside* stHeader, so hiding stHeader unconditionally
    # was silently deleting a phone's only way to open the sidebar at all.
    # Only strip these on wide (desktop) viewports, where the sidebar stays
    # permanently visible and this chrome is genuinely unused.
    "@media (min-width: 768px){"
    "div.block-container{padding-top:1rem}"
    "[data-testid='stHeader']{display:none}"
    "[data-testid='stSidebarHeader']{display:none}"
    "[data-testid='stSidebarCollapseButton']{display:none}"
    "[data-testid='stExpandSidebarButton']{display:none}"
    "}"
    "</style>",
    unsafe_allow_html=True,
)

picked_tickers: set[str] = set()
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

    for category, options in QUICK_PICK_CATEGORIES.items():
        with st.expander(category, expanded=True):
            picked_tickers.update(
                st.pills(
                    category,
                    options=list(options.values()),
                    selection_mode="multi",
                    default=[],
                    key=f"pick_{category}",
                    label_visibility="collapsed",
                )
            )
    tickers_input = st.text_input(
        "Other tickers (comma-separated)", value=", ".join(load_custom_tickers()),
        help="Persisted -- still here next time you open the app.",
    )
    _typed_custom_tickers = sorted({t.strip().upper() for t in tickers_input.split(",") if t.strip()})
    if _typed_custom_tickers != load_custom_tickers():
        save_custom_tickers(_typed_custom_tickers)

_tab_names = ["Research", "This Week"]
if not HOSTED:
    _tab_names += ["Portfolio", "Sim"]
_tab_names += [
    "Compare", "Corr", "Mom", "buy-dip", "Calendar",
    "PEAD", "Insider", "Inst'", "Rank", "LT data",
]
# A plain st.tabs() computes every tab's content on every script run regardless of which is
# visible (they're all rendered into the page, just CSS-hidden) -- with 13 tabs, several of
# which fetch prices for a whole universe of tickers or build multi-year factor panels, a
# full run can take well over a minute even before you can see anything past the first tab.
# segmented_control instead reports only the *selected* section, so only that one render
# function below actually runs each time -- switching sections is a fast, cheap rerun instead
# of everything computing eagerly upfront.
# Equal-width segments -- st.segmented_control otherwise sizes each button to fit its own label,
# so "This Week" ends up visibly wider than "Rank"; forcing flex:1 on every button makes them
# uniform regardless of label length.
st.markdown(
    """
    <style>
    div[data-testid="stButtonGroup"] > div { display: flex; width: 100%; }
    div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] { flex: 1 1 0; }
    </style>
    """,
    unsafe_allow_html=True,
)
active_tab = st.segmented_control(
    "Section", options=_tab_names, default=_tab_names[0], required=True,
    key="active_tab", label_visibility="collapsed",
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

_FUNDAMENTAL_STYLE: dict[str, tuple[str, str]] = {
    "supports": ("\U0001f7e2", "Fundamentals: Supports"),
    "contradicts": ("\U0001f534", "Fundamentals: Contradicts"),
    "neutral": ("⚪", "Fundamentals: Neutral"),
}


@st.dialog("Claims", width="medium")
def _claims_dialog(ticker: str, claims: list) -> None:
    """Experimental alternative to nested expanders: every claim for `ticker`
    shown as its own card in a scrollable modal, closeable without losing
    your place in the Theses list underneath. Shows the source article's AI
    summary if one's already cached (finance.summarize, e.g. from browsing
    the This Week tab) -- never generates one on demand, that'd be a fresh
    LLM call just to populate a claim card.
    """
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
            article_summary = get_cached_summary(c.source_link)
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
            a2.metric("Blended confidence", f"{ev['confidence']:.0%}")
            a3.metric(
                "Expected return", f"{ev['expected_return_pct']:+.1f}%",
                f"{ev['expected_horizon_days']}d horizon", delta_color="off",
            )
            news_confidence = ev.get("news_confidence")
            fundamental_confidence = ev.get("fundamental_confidence")
            with st.container(key=f"confidence_breakdown_{ticker}_{n}"):
                b1, b2 = st.columns(2)
                b1.metric("News confidence", f"{news_confidence:.0%}" if news_confidence is not None else "--")
                b2.metric(
                    "Fundamental support score",
                    f"{fundamental_confidence:.0%}" if fundamental_confidence is not None else "--",
                )
            st.write(f"Thesis: {_md(ev['thesis'])}")
            if ev.get("catalysts"):
                st.write("Catalysts: " + _md(", ".join(ev["catalysts"])))
            if ev.get("invalidation"):
                st.write("Invalidation: " + _md(", ".join(ev["invalidation"])))
            st.caption(_md(ev["reasoning"]))


@st.dialog("Fundamentals", width="medium")
def _fundamentals_dialog(ticker: str, fundamental_events: list) -> None:
    """Same card-in-a-scrollable-modal treatment as the Claims dialog, for
    the independent fundamental second opinion's history.
    """
    st.caption(f"{ticker}  ·  {len(fundamental_events)} check(s), newest first")
    for n, ev in reversed(list(enumerate(fundamental_events, 1))):
        with st.container(border=True, key=f"fund_card_{ticker}_{n}"):
            icon, headline = _FUNDAMENTAL_STYLE.get(ev["assessment"], ("", ev["assessment"]))
            st.markdown(f"**#{n}  {icon}  {headline}  ·  {ev['date']}**")
            direction = ev.get("direction")
            if ev.get("thesis"):
                st.write(f"**Thesis scored:** {_md(ev['thesis'])}")
            metric_label = f"Support for '{direction}'" if direction else "Fundamental confidence"
            st.metric(
                metric_label, f"{ev['fundamental_confidence']:.0%}",
                help=(
                    "0% = fundamentals strongly argue AGAINST this direction, 100% = fundamentals "
                    "strongly SUPPORT it, 50% = no strong bearing either way."
                ),
            )
            fv, cp = ev.get("fair_value_estimate"), ev.get("current_price")
            if fv and cp:
                implied = ev.get("implied_return_pct")
                implied_text = f"  ·  implied return {implied:+.1f}%" if implied is not None else ""
                st.caption(f"Analyst fair value estimate: \\${fv:.2f}  ·  Current price: \\${cp:.2f}{implied_text}")
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
            st.caption(_md(ev["reasoning"]))


@st.dialog("Critic", width="medium")
def _critic_dialog(ticker: str, critic_events: list) -> None:
    """Same card-in-a-scrollable-modal treatment as the Claims dialog, for
    the critic pass's history: deterministic guardrails (source
    concentration, evidence thinness, staleness) plus one LLM red-team call
    -- both dampen confidence, never raise it (see finance.critic and
    finance.tickerthesis._deterministic_critic_flags).
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
        "extracted) synthesized by finance.tickerthesis into one current view per ticker. Read-only: "
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
        claims = sorted(load_claims(ticker), key=lambda c: c.created, reverse=True)
        theses_rows.append((ticker, tt, claims))
    theses_rows.sort(key=lambda row: row[2][0].created if row[2] else dt.date.min, reverse=True)
    for ticker, tt, claims in theses_rows:
        n_aggregations = sum(1 for ev in tt.history if ev["event"] == "aggregated")
        latest_claim_date = f", latest {claims[0].created.isoformat()}" if claims else ""
        title = (
            f"{ticker}  ·  {tt.direction}  ·  conf={tt.confidence:.0%}  ·  "
            f"{n_aggregations} aggregation(s)  ·  {len(claims)} claim(s){latest_claim_date}"
        )
        with st.expander(title, expanded=False):
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
            st.write(f"**Catalysts:** {_md(', '.join(tt.catalysts)) if tt.catalysts else '--'}")
            st.write(f"**Invalidation:** {_md(', '.join(tt.invalidation)) if tt.invalidation else '--'}")

            aggregated_events = [ev for ev in tt.history if ev["event"] == "aggregated"]
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
        with st.popover("+ New portfolio"):
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
        earnings_by_ticker = {t: get_earnings_history(t, limit=EARNINGS_LOOKBACK_QUARTERS) for t in universe_tickers}

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
                hist = get_earnings_history(t, limit=EARNINGS_LOOKBACK_QUARTERS)
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
        earnings_by_ticker = {t: get_earnings_history(t, limit=4, refresh=weekly_refresh) for t in universe_tickers}
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

    st.divider()
    st.subheader("Semiconductor news")
    ai_summaries_on = has_api_key()
    st.caption(
        "Last 7 days from " + ", ".join(name for name, _ in NEWS_SOURCES) + (
            " -- AI-summarized (OpenRouter free tier), falling back to the source's own teaser if a "
            "summary isn't available." if ai_summaries_on else " (source's own teaser -- no OPENROUTER_API_KEY configured for AI summaries)."
        )
    )
    with st.spinner("Fetching news..."):
        news_df = get_all_news(refresh=weekly_refresh)
    recent_news = news_df[news_df["published"] >= window_start] if not news_df.empty else news_df
    if recent_news.empty:
        st.caption("No new articles in the last 7 days.")
    else:
        blocks = []
        for row in recent_news.itertuples():
            meta_bits = [row.source]
            if row.author:
                meta_bits.append(row.author)
            meta_bits.append(row.published.strftime("%Y-%m-%d"))
            text, is_ai = row.summary, False
            if ai_summaries_on:
                ai_text = summarize_article(row.link, row.title, row.content)
                if ai_text:
                    text, is_ai = ai_text, True

            block = f"**[{row.title}]({row.link})**  \n*{' · '.join(meta_bits)}*"
            if text:
                block += f"\n\n{text}"
                if is_ai:
                    block += "\n\n🤖 *AI summary*"
            blocks.append(block)
        st.markdown("\n\n---\n\n".join(blocks))

    st.divider()
    st.subheader("Compare")
    st.caption("Cumulative return of your sidebar picks (plus SPY by default) vs. the S&P 500.")
    compare_period = st.pills(
        "Period",
        options=["Last week", "1 month", "YTD"],
        selection_mode="single",
        default="Last week",
        key="weekly_compare_period",
    )
    compare_typed = {t.strip().upper() for t in tickers_input.split(",") if t.strip()}
    compare_tickers = sorted(picked_tickers | compare_typed | {SP500_BENCHMARK})
    if not compare_period:
        st.info("Pick a period above.")
    else:
        compare_start = {
            "Last week": today - pd.Timedelta(days=7),
            "1 month": today - pd.DateOffset(months=1),
            "YTD": pd.Timestamp(dt.date(today.year, 1, 1)),
        }[compare_period]

        with st.spinner("Fetching prices..."):
            compare_prices = get_prices(
                compare_tickers, start=compare_start.date().isoformat(), refresh=weekly_refresh
            )
        compare_prices = compare_prices.loc[compare_prices.index >= compare_start].dropna(axis=1, how="all")

        missing = set(compare_tickers) - set(compare_prices.columns)
        if missing:
            st.warning(f"No data found for: {', '.join(sorted(missing))}")

        if compare_prices.empty:
            st.error("No price data available for this selection/period.")
        else:
            compare_cumulative = compare_prices / compare_prices.bfill().iloc[0] - 1
            fig_compare = go.Figure()
            for col in compare_cumulative.columns:
                is_benchmark = col == SP500_BENCHMARK
                fig_compare.add_trace(
                    go.Scatter(
                        x=compare_cumulative.index,
                        y=compare_cumulative[col],
                        name=col,
                        line=dict(width=3, color=SP500_LINE_COLOR) if is_benchmark else dict(width=2),
                    )
                )
            fig_compare.update_layout(
                yaxis_tickformat=".0%",
                yaxis_title="Cumulative return",
                xaxis_title="Date",
                hovermode="x unified",
                legend_title_text="",
                height=400,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_compare, width="stretch", key="chart_weekly_compare")


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


if active_tab == "This Week":
    render_weekly_tab()
elif active_tab == "Research":
    render_research_tab()
elif active_tab == "Portfolio":
    render_portfolio_tab()
elif active_tab == "Sim":
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
