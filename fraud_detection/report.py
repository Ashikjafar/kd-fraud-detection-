"""Analytics dashboard generator:  python -m fraud_detection.report

Reads reports/metrics.json, curves.json and latency.json and writes a single
self-contained reports/dashboard.html (inline SVG + a little JS for tooltips;
follows the OS light/dark setting).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--seq:#2a78d6;--good:#006300;--bad:#b42626}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--seq:#3987e5;--good:#0ca30c;--bad:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--seq:#3987e5;--good:#0ca30c;--bad:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page,#f9f9f7)}
.viz-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--page);padding:24px 16px 48px;line-height:1.45}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 16px;font-size:14px}
.note{border:1px solid var(--border);background:var(--surface);border-radius:10px;padding:12px 14px;font-size:13.5px;color:var(--ink2);margin:0 0 20px}
.note b{color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .v{font-size:30px;font-weight:650}.tile .l{font-size:13px;color:var(--ink2)}.tile .d{font-size:12px;color:var(--muted);margin-top:2px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--ink2);margin:6px 0 4px}
.legend i{display:inline-block;width:22px;height:0;border-top:2.5px solid;vertical-align:middle;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--ink2);font-weight:600}
.tablewrap{overflow-x:auto}
.split th:nth-child(2),.split td:nth-child(2){text-align:right}.split th:last-child,.split td:last-child{text-align:left}
.badge{font-size:11px;border:1px solid var(--border);border-radius:999px;padding:1px 7px;margin-left:6px;color:var(--ink2)}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--ink2);font-size:11.5px;font-family:inherit}
svg .muted{fill:var(--muted)}
svg .axis{stroke:var(--axis);stroke-width:1}svg .grid{stroke:var(--grid);stroke-width:1}
svg .ring{stroke:var(--surface);stroke-width:2}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;
padding:6px 9px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.18);display:none;z-index:10;max-width:260px}
.cm{width:auto;margin-top:6px}.cm td,.cm th{text-align:center;padding:10px 14px}
.cm td.n{font-size:22px;font-weight:650}.cm small{display:block;font-size:11.5px;color:var(--muted);font-weight:400}
ul.plain{margin:6px 0 0;padding-left:18px;color:var(--ink2);font-size:13px}
.foot{font-size:12.5px;color:var(--muted);margin-top:18px}
@media (forced-colors:active){svg path,svg rect{forced-color-adjust:auto;stroke:CanvasText}}
"""

JS = """
(function(){
var tip=document.getElementById('tip');
function show(t,x,y){tip.textContent=t;tip.style.display='block';var w=tip.offsetWidth;tip.style.left=Math.min(x+14,innerWidth-w-8)+'px';tip.style.top=(y+14)+'px';}
function hide(){tip.style.display='none';}
document.addEventListener('mousemove',function(e){
 var t=e.target.closest&&e.target.closest('[data-tip]');
 if(t){show(t.getAttribute('data-tip'),e.clientX,e.clientY);return;}
 var s=e.target.closest&&e.target.closest('svg.line');
 if(s){var d=JSON.parse(s.getAttribute('data-series')),r=s.getBoundingClientRect(),k=r.width/s.viewBox.baseVal.width;
  var mx=(e.clientX-r.left)/k,my=(e.clientY-r.top)/k,best=null,bd=1e9;
  d.forEach(function(se){se.pts.forEach(function(p){var dd=Math.hypot(p[0]-mx,p[1]-my);if(dd<bd){bd=dd;best=[se,p];}});});
  var m=s.querySelector('.hov');
  if(best&&bd<28){m.setAttribute('cx',best[1][0]);m.setAttribute('cy',best[1][1]);m.setAttribute('r',5);m.style.fill='var('+best[0].color+')';
   show(best[0].name+': '+s.getAttribute('data-xl')+' '+best[1][2]+', '+s.getAttribute('data-yl')+' '+best[1][3],e.clientX,e.clientY);return;}
  m.setAttribute('r',0);hide();return;}
 hide();
});
document.addEventListener('mouseleave',hide);
})();
"""

FEATURE_LABELS = {
    "log_amount": "Transaction amount (log)", "amount_ratio": "Amount vs customer's usual spend",
    "new_device": "New device for this customer", "location_change": "City changed since last transaction",
    "txn_count_24h": "Transactions in the last 24h", "time_since_last_min": "Time since last transaction",
    "hour_sin": "Time of day (sine)", "hour_cos": "Time of day (cosine)", "is_night": "Night-time (23:00-05:00)",
    "txn_index": "Length of customer history",
}


def nice_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for prefix, label in (("merchant_category_", "Merchant: "), ("channel_", "Channel: ")):
        if name.startswith(prefix):
            return label + name[len(prefix):]
    return name


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def fmt(v, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# ------------------------------------------------------------------ SVG building blocks
def _round_bar_h(x, y, w, h, r=4) -> str:
    """Horizontal bar, left end square on the baseline, right end rounded."""
    r = min(r, w, h / 2)
    return (f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z")


def _round_bar_v(x, y, w, h, r=4) -> str:
    """Vertical bar, bottom square on the baseline, top rounded."""
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f}v-{h - r:.1f}a{r},{r} 0 0 1 {r},-{r}h{w - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 {r},{r}v{h - r:.1f}z")


def hbar_chart(rows, vmax, color_for, value_fmt, width=520, label_w=190) -> str:
    """rows: [(label, value, tooltip)] ; color_for(i) -> css colour."""
    bh, gap, top = 18, 10, 6
    h = top + len(rows) * (bh + gap)
    plot_w = width - label_w - 64
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" style="max-width:{width}px">']
    out.append(f'<line class="axis" x1="{label_w}" y1="0" x2="{label_w}" y2="{h}"/>')
    for i, (label, val, tip) in enumerate(rows):
        y = top + i * (bh + gap)
        w = max(2.0, plot_w * val / vmax) if vmax else 2.0
        out.append(f'<text x="{label_w - 8}" y="{y + bh / 2 + 4}" text-anchor="end">{esc(label)}</text>')
        out.append(f'<path d="{_round_bar_h(label_w, y, w, bh)}" style="fill:{color_for(i)}" data-tip="{esc(tip)}"/>')
        out.append(f'<text x="{label_w + w + 6}" y="{y + bh / 2 + 4}">{esc(value_fmt(val))}</text>')
    out.append("</svg>")
    return "".join(out)


def column_chart(values, tips, labels, ymax, ylabel, width=1000, height=240) -> str:
    ml, mr, mt, mb = 46, 8, 10, 30
    pw, ph = width - ml - mr, height - mt - mb
    n = len(values)
    slot = pw / n
    bw = slot * 0.62
    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for k in range(5):
        v = ymax * k / 4
        y = mt + ph - ph * k / 4
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}"/>')
        out.append(f'<text class="muted" x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.1f}{ylabel}</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt + ph}" x2="{width - mr}" y2="{mt + ph}"/>')
    for i, v in enumerate(values):
        h = ph * v / ymax if ymax else 0
        x = ml + i * slot + (slot - bw) / 2
        if h > 0.5:
            out.append(f'<path d="{_round_bar_v(x, mt + ph - h, bw, h)}" style="fill:var(--seq)" data-tip="{esc(tips[i])}"/>')
        else:
            out.append(f'<rect x="{x:.1f}" y="{mt + ph - 1.5}" width="{bw:.1f}" height="1.5" style="fill:var(--seq)" data-tip="{esc(tips[i])}"/>')
        if i % 3 == 0:
            out.append(f'<text class="muted" x="{x + bw / 2:.1f}" y="{height - 10}" text-anchor="middle">{esc(labels[i])}</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart(series, kind, width=500, height=360) -> str:
    """series: [(name, css_var, dash, curve)] with curve = {"x": [...], "y": [...]}"""
    ml, mr, mt, mb = 48, 14, 12, 42
    pw, ph = width - ml - mr, height - mt - mb
    xl, yl = ("false-positive rate", "true-positive rate") if kind == "roc" else ("recall", "precision")
    px = lambda v: ml + pw * v
    py = lambda v: mt + ph * (1 - v)
    out = []
    for k in range(5):
        v = k / 4
        out.append(f'<line class="grid" x1="{ml}" y1="{py(v):.1f}" x2="{width - mr}" y2="{py(v):.1f}"/>')
        out.append(f'<text class="muted" x="{ml - 6}" y="{py(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        out.append(f'<text class="muted" x="{px(v):.1f}" y="{mt + ph + 16}" text-anchor="middle">{v:.2f}</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt + ph}" x2="{width - mr}" y2="{mt + ph}"/>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}"/>')
    if kind == "roc":
        out.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" style="stroke:var(--muted);stroke-width:1;stroke-dasharray:2 4"/>')
    data = []
    for name, color, dash, curve in series:
        pts = list(zip(curve["x"], curve["y"]))
        d = "M" + "L".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path d="{d}" fill="none" style="stroke:var({color});stroke-width:2;stroke-linejoin:round"{dash_attr}/>')
        step = max(1, len(pts) // 60)
        data.append({"name": name, "color": color,
                     "pts": [[round(px(x), 1), round(py(y), 1), f"{x:.3f}", f"{y:.3f}"] for x, y in pts[::step]]})
    out.append(f'<text x="{ml + pw / 2}" y="{height - 6}" text-anchor="middle">{xl}</text>')
    out.append(f'<text transform="translate(12 {mt + ph / 2}) rotate(-90)" text-anchor="middle">{yl}</text>')
    out.append('<circle class="hov ring" cx="0" cy="0" r="0"/>')
    return (f'<svg class="line" viewBox="0 0 {width} {height}" role="img" data-xl="{xl}" data-yl="{yl}" '
            f'data-series="{esc(json.dumps(data))}">' + "".join(out) + "</svg>")


# ------------------------------------------------------------------ page
def build_dashboard(reports_dir: str = "reports") -> Path:
    rd = Path(reports_dir)
    m = json.loads((rd / "metrics.json").read_text())
    curves = json.loads((rd / "curves.json").read_text())
    lat = json.loads((rd / "latency.json").read_text()) if (rd / "latency.json").exists() else None

    best = m["best_model"]
    bm = m["models"][best]
    ds = m["dataset"]
    test = ds["splits"]["test"]
    majority_acc = 1 - test["fraud"] / test["rows"]

    tiles = [
        (fmt(bm["f1"]), "F1-score", "balance of precision and recall"),
        (fmt(bm["precision"]), "Precision", "flagged transactions that were fraud"),
        (fmt(bm["recall"]), "Recall", "fraud that was caught"),
        (fmt(bm["roc_auc"]), "ROC-AUC", "fraud vs genuine separation"),
    ]
    if lat:
        h = lat["http_round_trip_localhost"]
        tiles.append((f"{h['p95_ms']:.1f} ms", "API response (p95)", f"mean {h['mean_ms']:.1f} ms, local"))
    kpi_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div><div class="d">{d}</div></div>'
                       for v, l, d in tiles)

    order = ["Logistic Regression", "Random Forest", [k for k in m["models"] if k in ("XGBoost", "Gradient Boosting (HistGB)")][0],
             "Isolation Forest", "Fixed rules (baseline)"]
    rows = []
    for name in order:
        r = m["models"][name]
        badge = '<span class="badge">selected</span>' if name == best else ""
        rows.append(
            f"<tr><td>{esc(name)}{badge}</td><td>{esc(r['kind'])}</td><td>{fmt(r['accuracy'])}</td><td>{fmt(r['precision'])}</td>"
            f"<td>{fmt(r['recall'])}</td><td>{fmt(r['f1'])}</td><td>{fmt(r['roc_auc'])}</td><td>{fmt(r['pr_auc'])}</td></tr>")
    table = ('<div class="tablewrap"><table><thead><tr><th>Model</th><th>Type</th><th>Accuracy</th><th>Precision</th><th>Recall</th>'
             '<th>F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")

    f1_rows = sorted(((n, m["models"][n]["f1"], f"{n}: F1 {m['models'][n]['f1']:.3f}") for n in order), key=lambda t: -t[1])
    f1_chart = hbar_chart(f1_rows, 1.0, lambda i: "var(--seq)" if f1_rows[i][0] == best else "var(--muted)", lambda v: f"{v:.3f}")

    sup = [n for n in order if m["models"][n]["kind"] == "supervised"]
    palette = [("--s1", None), ("--s2", "7 4"), ("--s3", "2 3")]

    def lines_for(kind: str):
        lines = [(n, palette[i][0], palette[i][1], curves[n][kind]) for i, n in enumerate(sup)]
        lines.append(("Isolation Forest", "--muted", "1 0", curves["Isolation Forest"][kind]))
        return lines

    legend_items = [(n, palette[i][0], palette[i][1]) for i, n in enumerate(sup)] + [("Isolation Forest", "--muted", "1 0")]
    legend = "".join(f'<span><i style="border-color:var({c});{"border-top-style:dashed;" if d and d != "1 0" else ""}"></i>{esc(n)}</span>'
                     for n, c, d in legend_items)
    roc_svg = line_chart(lines_for("roc"), "roc")
    pr_svg = line_chart(lines_for("pr"), "pr")

    imp = [(nice_feature(i["feature"]), max(i["importance"], 0.0)) for i in m["feature_importance"][:10]]
    imax = max((v for _, v in imp), default=1.0) or 1.0
    imp_rows = [(n, v, f"{n}: score drop {v:.4f} when shuffled") for n, v in imp]
    imp_chart = hbar_chart(imp_rows, imax, lambda i: "var(--seq)", lambda v: f"{v:.3f}", width=520, label_w=230)

    by_hour = {d["hour"]: d for d in ds["fraud_rate_by_hour"]}
    hours = list(range(24))
    rates = [100.0 * by_hour[h]["fraud"] / by_hour[h]["total"] if h in by_hour and by_hour[h]["total"] else 0.0 for h in hours]
    tips = [f"{h:02d}:00-{h:02d}:59 - {rates[h]:.1f}% fraud ({by_hour[h]['fraud']} of {by_hour[h]['total']:,})" if h in by_hour else f"{h:02d}:00 - no data" for h in hours]
    ymax = max(5.0, float(int(max(rates) / 5.0) + 1) * 5.0)
    hour_chart = column_chart(rates, tips, [f"{h:02d}:00" for h in hours], ymax, "%")

    cm = (f'<table class="cm"><tr><th></th><th>Predicted genuine</th><th>Predicted fraud</th></tr>'
          f'<tr><th>Actually genuine</th><td class="n">{bm["tn"]:,}<small>correctly passed</small></td><td class="n">{bm["fp"]:,}<small>false alarms</small></td></tr>'
          f'<tr><th>Actually fraud</th><td class="n">{bm["fn"]:,}<small>missed fraud</small></td><td class="n">{bm["tp"]:,}<small>caught fraud</small></td></tr></table>')

    ab = m["ablation_smote"]
    ab_rows = "".join(
        f"<tr><td>{label}</td><td>{fmt(a['precision'])}</td><td>{fmt(a['recall'])}</td><td>{fmt(a['f1'])}</td><td>{fmt(a['roc_auc'])}</td><td>{fmt(a['pr_auc'])}</td></tr>"
        for label, a in (("With SMOTE", ab["with_smote"]), ("Without SMOTE", ab["without_smote"])))
    ab_table = f'<table><thead><tr><th>{esc(best)}</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead><tbody>{ab_rows}</tbody></table>'

    sp = ds["splits"]
    cl = ds["cleaning"]
    split_rows = "".join(
        f"<tr><td>{n.title()}</td><td>{s['rows']:,}</td><td>{s['fraud']:,}</td><td>{s['from'][:10]} to {s['to'][:10]}</td></tr>" for n, s in sp.items())
    th = m["thresholds"]

    body = f"""
<div class="wrap">
<h1>Cloud Based Fraud Detection System - Analytics Dashboard</h1>
<p class="sub">Model {esc(m['version'])} | selected model: {esc(best)} | trained {esc(m['trained_at'][:10])}</p>
<p class="note"><b>Synthetic data.</b> All figures come from a generated transaction dataset ({ds['rows_clean']:,} transactions, {ds['fraud_rate']:.1%} fraud)
that mimics card/UPI traffic. They demonstrate that the pipeline works end to end; they are not performance on a real bank's data.
Metrics are on the <b>untouched test split</b> (the most recent {test['rows']:,} transactions); thresholds were chosen on a separate validation split.</p>
<div class="kpis">{kpi_html}</div>

<div class="card"><h2>Model comparison</h2>
<p class="sub">Test split, {test['fraud']} fraud cases out of {test['rows']:,}. A model that approves everything would score accuracy {majority_acc:.3f} while catching no fraud, so precision, recall and F1 are the measures that matter.</p>
{table}
<h2 style="margin-top:16px">F1-score by model</h2>{f1_chart}</div>

<div class="grid2">
<div class="card"><h2>ROC curves</h2><p class="sub">Higher and further left is better. Dotted diagonal = random guessing.</p><div class="legend">{legend}</div>{roc_svg}</div>
<div class="card"><h2>Precision-recall curves</h2><p class="sub">The more informative view when fraud is rare.</p><div class="legend">{legend}</div>{pr_svg}</div>
</div>

<div class="grid2">
<div class="card"><h2>Confusion matrix - {esc(best)}</h2><p class="sub">Review threshold {th['review']:.3f}, block threshold {th['block']:.3f} (fraud probability).</p>{cm}</div>
<div class="card"><h2>Effect of SMOTE</h2><p class="sub">Training set: {ds['smote']['fraud_before']:,} fraud and {ds['smote']['genuine_before']:,} genuine rows, oversampled to {ds['smote']['fraud_after']:,} fraud.</p>{ab_table}</div>
</div>

<div class="card"><h2>What drives the predictions</h2>
<p class="sub">Permutation importance on validation data: how much the model's PR-AUC drops when a feature is shuffled.</p>{imp_chart}</div>

<div class="card"><h2>Fraud rate by hour of day</h2><p class="sub">Share of transactions in each hour that are fraudulent (whole dataset).</p>{hour_chart}</div>

<div class="card"><h2>Data and pipeline</h2>
<div class="tablewrap"><table class="split"><thead><tr><th>Split</th><th>Transactions</th><th>Fraud</th><th>Period</th></tr></thead><tbody>{split_rows}</tbody></table></div>
<ul class="plain"><li>Cleaning: {cl['rows_in']:,} raw rows, {cl['duplicates_removed']} duplicates and {cl['invalid_rows_removed']} invalid rows removed, {cl['missing_category_filled']} missing merchant categories filled as "unknown".</li>
<li>Splits are chronological (older data trains, newer data tests), and every behavioural feature uses only earlier transactions of the same customer.</li>
<li>Boosting backend: {esc(m['boosting_backend'])}. Live case queue: <code>GET /cases</code> on the running API.</li></ul></div>
<p class="foot">Generated by fraud_detection.report from reports/metrics.json.</p>
</div><div id="tip" role="tooltip"></div>
"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Fraud Detection Dashboard</title><style>{CSS}</style></head><body><div class="viz-root">{body}</div><script>{JS}</script></body></html>')
    out = rd / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default="reports")
    print(f"wrote {build_dashboard(ap.parse_args().reports_dir)}")


if __name__ == "__main__":
    main()
