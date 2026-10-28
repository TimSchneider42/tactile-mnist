// Baseline bar charts on the backs of the environment cards.
// Data: data.json (written by tools/make_data_json.py), the per-seed {eval,train}_mean_loss_detailed curves.
// The computation is the one of paper-typst/tools/scores.py; the bars show "mean"/"std" (the loss averaged over the
// training run), as in the paper's results overview (Fig. 5, "Avg. Loss").
(function (root) {
    "use strict";

    const FINAL_POINTS = 2;
    const FIELD = "mean";  // "mean" (as in Fig. 5) or "final"
    const METHODS = [
        ["crossq", "CrossQ", "#59A14F"],
        ["sac", "SAC", "#F28E2B"],
        ["ppo", "PPO", "#B07AA1"],
        ["ham", "HAM", "#9C755F"],
        ["rand_act", "Random", "#8E8E8E"],
    ];

    // pandas semantics: NaN cells are skipped, a mean over no values is NaN
    const nanMean = (xs) => {
        const v = xs.filter((x) => !Number.isNaN(x));
        return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
    };
    const nanStd = (xs) => {  // ddof = 0
        const v = xs.filter((x) => !Number.isNaN(x));
        if (!v.length) return NaN;
        const m = nanMean(v);
        return Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / v.length);
    };

    // per-seed curves (null = missing) -> {mean, std, final, final_std, seeds}
    function scoresFromSeeds(curves) {
        const seeds = curves.map((c) => c.map((x) => (x === null || Number.isNaN(x) ? NaN : Math.min(x, 1.0))));  // clip(upper=1.0)
        const perSeed = seeds.map(nanMean);
        const final = seeds.map((s) => nanMean(s.slice(-FINAL_POINTS)));
        return {
            mean: nanMean(perSeed), std: nanStd(perSeed),
            final: nanMean(final), final_std: nanStd(final), seeds: seeds.length,
        };
    }

    // CSV text (the baseline-data/ layout) -> {mean, std, final, final_std, seeds}
    function computeScores(text) {
        const lines = text.trim().split(/\r?\n/);
        const header = lines[0].split(",").map((h) => h.trim());
        const cols = header.map((h, i) => [h, i]).filter(([h]) => h.startsWith("seed_")).map(([, i]) => i);
        const rows = lines.slice(1).map((l) => l.split(","));
        return scoresFromSeeds(cols.map((i) => rows.map((r) => {
            const s = (r[i] || "").trim();
            return s === "" ? NaN : Number(s.toLowerCase() === "nan" ? NaN : s);
        })));
    }

    if (typeof module !== "undefined" && module.exports) {
        module.exports = { computeScores, scoresFromSeeds, METHODS };
        return;
    }

    let dataPromise = null;
    const taskPromises = {};

    function loadTask(task) {
        if (!dataPromise) {
            dataPromise = fetch("data.json").then((r) => {
                if (!r.ok) throw new Error(r.status);
                return r.json();
            });
        }
        if (!taskPromises[task]) {
            taskPromises[task] = dataPromise.then((data) => {
                const methods = (data.tasks || {})[task];
                if (!methods) throw new Error("unknown task " + task);
                return METHODS.filter(([m]) => methods[m]).map(([m, label, color]) => (
                    { method: m, label, color, ...scoresFromSeeds(methods[m].seeds) }));
            });
        }
        return taskPromises[task];
    }

    const pct = (v) => (100 * Math.min(Math.max(v, 0), 1)).toFixed(1) + "%";
    const el = (tag, cls, style) => {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (style) e.style.cssText = style;
        return e;
    };

    function draw(box, env, rows) {
        const sdKey = FIELD === "mean" ? "std" : FIELD + "_std";
        rows = rows.filter((r) => Number.isFinite(r[FIELD]));
        if (!rows.length) throw new Error("no data");
        box.textContent = "";
        for (const r of rows) {
            const mu = r[FIELD], sd = Number.isFinite(r[sdKey]) ? r[sdKey] : 0;
            box.appendChild(el("span", "sb-lbl")).textContent = r.label;
            const track = box.appendChild(el("span", "sb-track"));
            track.appendChild(el("span", "sb-bar", `width:${pct(mu)};background:${r.color}`));
            const lo = Math.max(mu - sd, 0), hi = Math.min(mu + sd, 1);
            if (hi > lo) track.appendChild(el("span", "sb-err", `left:${pct(lo)};width:${pct(hi - lo)}`));
        }
        box.appendChild(el("span", "sb-lbl"));
        const ticks = box.appendChild(el("span", "sb-ticks"));
        for (const [t, x] of [["0", "0"], ["0.5", "50%"], ["1", "100%"]]) {
            ticks.appendChild(el("span", "", `left:${x}`)).textContent = t;
        }
        box.appendChild(el("span", "sb-cap")).textContent = "loss (1 = blind guess)";
        box.removeAttribute("aria-hidden");
        box.setAttribute("role", "img");
        box.setAttribute("aria-label", `Baseline loss on ${env} (1 = blind guess, lower is better): `
            + rows.map((r) => `${r.label} ${r[FIELD].toFixed(2)}`).join(", "));
        box.classList.add("ready");
    }

    function show(card) {
        const box = card.querySelector(".score-bars");
        if (!box || box.dataset.state) return;
        box.dataset.state = "loading";
        loadTask(box.dataset.task)
            .then((rows) => { draw(box, box.dataset.env, rows); box.dataset.state = "done"; })
            .catch(() => { box.dataset.state = "failed"; box.remove(); });  // just the description
    }

    function init() {
        document.querySelectorAll(".flip-card").forEach((card) => {
            const go = () => show(card);
            card.addEventListener("mouseenter", go);
            card.addEventListener("focusin", go);
            card.addEventListener("touchstart", go, { passive: true });
            card.addEventListener("click", go);
        });
    }

    root.ScoreBars = { computeScores, loadTask };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})(typeof window !== "undefined" ? window : globalThis);
