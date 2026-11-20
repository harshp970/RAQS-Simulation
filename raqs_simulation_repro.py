"""
Expanded reproducible simulation suite (v2) for HireMind.AI RAQS:
- Deterministic 4-profile verification
- Multi-seed (5 seeds x 200 trials = 1,000 trials/cohort) Monte Carlo at sigma=8
- Multi-noise sweep (sigma = 4, 8, 12, 16) across 5 seeds
- Initialization tier (l1=1 vs 2) and smoothing factor (alpha) ablation
- Generates Fig. 1 (System Architecture & RAQS Feedback Loop) + Figs. 2-4 (Quantitative plots)
"""
import json
import math
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

MID_DIR = "artifacts/file_generation/ttl=63d/intermediate"
os.makedirs(MID_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STATIC_SCHEDULE = [1, 1, 2, 3, 3, 3]  # E, E, M, H, H, H
SEEDS = [42, 101, 202, 303, 404]

PROFILES = {
    "Novice": [42, 48, 52, 55, 50, 54],
    "Intermediate": [72, 76, 68, 74, 70, 75],
    "Advanced": [88, 92, 86, 90, 89, 94],
    "Mixed": [50, 82, 85, 55, 84, 88],
}

COHORT_MEANS = {
    "Novice (μ=48)": [48.0] * 6,
    "Intermediate (μ=71)": [71.0] * 6,
    "Advanced (μ=89)": [89.0] * 6,
    "Mixed (μ=70)": [50.0, 82.0, 85.0, 55.0, 84.0, 88.0],
}


def target_tier(score):
    """Half-open rubric band mapping: [0,60)->1, [60,80)->2, [80,100]->3."""
    if score >= 80.0:
        return 3
    elif score >= 60.0:
        return 2
    return 1


def step_delta(score):
    if score >= 80.0:
        return 1
    elif score >= 60.0:
        return 0
    return -1


def simulate_episode(scores, policy="raqs", alpha=0.60, l1=2, rng=None):
    theta = 0.50
    level = l1
    levels = []
    thetas = []
    for t, s in enumerate(scores, start=1):
        if policy == "raqs":
            cur_l = level
        elif policy == "fixed":
            cur_l = STATIC_SCHEDULE[t - 1]
        elif policy == "random":
            cur_l = rng.choice([1, 2, 3])
        else:
            raise ValueError(policy)
        levels.append(cur_l)
        theta = alpha * (s / 100.0) + (1.0 - alpha) * theta
        thetas.append(theta)
        if policy == "raqs":
            level = max(1, min(3, level + step_delta(s)))

    targets = [target_tier(s) for s in scores]
    mae = sum(abs(levels[i] - targets[i]) for i in range(1, 6)) / 5.0
    exact = sum(1 for i in range(1, 6) if levels[i] == targets[i]) / 5.0
    over_ch = sum(1 for i in range(1, 6) if levels[i] > targets[i]) / 5.0
    return {
        "levels": levels,
        "thetas": thetas,
        "mae": mae,
        "exact": exact,
        "over_ch": over_ch,
        "theta6": thetas[-1],
    }


def draw_architecture_diagram(out_path):
    """Draw a clean, publication-grade IEEE System Architecture & RAQS Feedback Loop diagram."""
    fig, ax = plt.subplots(figsize=(3.45, 3.65), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#F4F6F9", ec="#2C3E50", lw=1.0, title_col="#111111"):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(rect)
        if sub:
            ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                    fontsize=7.3, fontweight="bold", color=title_col)
            ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                    fontsize=6.2, color="#333333")
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=7.3, fontweight="bold", color=title_col)

    def arrow(x1, y1, x2, y2, color="#2C3E50", lw=1.0, rad=0.0, ls="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.45,head_width=0.25",
                color=color, lw=lw, linestyle=ls,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    # Top layer: Candidate -> React Client -> API Gateway
    box(0.5, 9.55, 3.8, 0.85, "Candidate", "Text / Code Answers", fc="#EEF4FA", ec="#1B4F72")
    box(5.3, 9.55, 4.2, 0.85, "React 19 Client", "Monaco Editor, Timer, Recharts", fc="#EEF4FA", ec="#1B4F72")
    arrow(4.35, 9.97, 5.24, 9.97, color="#1B4F72")

    box(2.0, 8.20, 6.0, 0.82, "Express API Gateway (:8000)", "Firebase Auth, Routing, Redis Cache Check", fc="#F4F6F9", ec="#34495E")
    arrow(7.4, 9.50, 6.5, 9.07, color="#2C3E50")

    # LangGraph StateGraph container
    lg_box = FancyBboxPatch(
        (0.35, 2.05), 9.3, 5.65,
        boxstyle="round,pad=0.08,rounding_size=0.22",
        facecolor="#FAFBFC", edgecolor="#1B4F72", linewidth=1.1, linestyle="--",
    )
    ax.add_patch(lg_box)
    ax.text(5.0, 7.48, "Interview Service (:8002) — LangGraph StateGraph (Groq gpt-oss-120b)",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#1B4F72")

    # Nodes inside LangGraph
    box(0.75, 6.20, 3.9, 0.90, "1. Interview Agent", "Opening Probe q₁ (l₁=2, Resume)", fc="#EBF5FB", ec="#1B4F72")
    box(5.35, 6.20, 3.9, 0.90, "2. Feedback Agent", "Score Sₜ (0–100) + 8-D Vector vₜ", fc="#FEF5E7", ec="#B7791F")
    arrow(4.70, 6.65, 5.30, 6.65, color="#2C3E50")
    arrow(3.5, 8.15, 2.7, 7.16, color="#2C3E50")
    ax.text(2.45, 7.82, "start", fontsize=6.0, style="italic", color="#444444")
    arrow(6.5, 8.15, 7.3, 7.16, color="#2C3E50")
    ax.text(7.35, 7.82, "submitAnswer", fontsize=6.0, style="italic", color="#444444")

    # RAQS Selector (highlighted core contribution)
    box(4.65, 4.35, 4.7, 1.15, "3. RAQS Adaptive Selector",
        "Eq.(1) θ̂ₜ  |  Eq.(2) lₜ₊₁  |  Eq.(3) k*ₜ\nMode mₜ₊₁ + Sliding Window W ≤ 4",
        fc="#E8F8F5", ec="#148F77", lw=1.4, title_col="#0E6251")
    arrow(7.3, 6.15, 7.3, 5.56, color="#148F77", lw=1.2)
    ax.text(7.52, 5.83, "t < T", fontsize=6.2, fontweight="bold", color="#0E6251")

    # Next Question box + feedback loop arrow
    box(0.75, 4.45, 3.2, 0.95, "Next Question qₜ₊₁", "Delivered to Candidate\n(Turns 5–6: Monaco Code)", fc="#E8F8F5", ec="#148F77")
    arrow(4.60, 4.92, 4.02, 4.92, color="#148F77", lw=1.2)

    # Closed-loop feedback arrow back up to Feedback Agent / Candidate
    arrow(1.20, 5.45, 1.20, 9.50, color="#148F77", lw=1.2, ls="-")
    ax.text(0.56, 7.85, "RAQS\nLoop", fontsize=6.2, fontweight="bold", color="#0E6251", ha="center")

    # Summary Agent (terminal branch)
    box(2.1, 2.35, 5.8, 0.92, "4. Summary Agent (t = T)", "overallScore, 80–120w Summary, Strengths & Weaknesses", fc="#F4ECF7", ec="#6C3483")
    arrow(5.6, 6.15, 4.6, 3.33, color="#6C3483", lw=1.0, ls="--")
    ax.text(4.55, 3.90, "t = T", fontsize=6.1, fontweight="bold", color="#6C3483")

    # Persistence layer at bottom
    box(0.6, 0.35, 4.1, 0.95, "MongoDB (Mongoose)", "Transcripts, 8-D Scores, θ̂ₜ", fc="#F4F6F9", ec="#566573")
    box(5.3, 0.35, 4.1, 0.95, "Redis Cache (:6379)", "Invalidate interviews:${userId}", fc="#F4F6F9", ec="#566573")
    arrow(3.5, 2.00, 2.65, 1.36, color="#566573")
    arrow(6.5, 2.00, 7.35, 1.36, color="#566573")

    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    results = {
        "deterministic": {},
        "multiseed_sigma8": {},
        "multinoise": {},
        "ablation": {},
    }

    # 1. Deterministic 4-profile verification
    lvl_map = {1: "E", 2: "M", 3: "H"}
    for name, scores in PROFILES.items():
        r_raqs = simulate_episode(scores, policy="raqs", alpha=0.60, l1=2)
        r_fix = simulate_episode(scores, policy="fixed", alpha=0.60, l1=2)
        results["deterministic"][name] = {
            "raqs": r_raqs,
            "fixed": r_fix,
            "seq": ",".join(lvl_map[x] for x in r_raqs["levels"]),
        }

    # 2. Multi-seed Monte Carlo at sigma=8 (5 seeds x 200 trials = 1,000 trials/cohort)
    print("=== 5-SEED MONTE CARLO (SIGMA=8, 200 TRIALS/SEED/COHORT) ===")
    for cname, base_means in COHORT_MEANS.items():
        results["multiseed_sigma8"][cname] = {}
        for pol in ("raqs", "fixed", "random"):
            seed_maes, seed_exacts, seed_ocs = [], [], []
            for sd in SEEDS:
                rng = random.Random(sd)
                maes, exs, ocs = [], [], []
                for _ in range(200):
                    sc = [max(0.0, min(100.0, rng.gauss(m, 8.0))) for m in base_means]
                    ep = simulate_episode(sc, policy=pol, alpha=0.60, l1=2, rng=rng)
                    maes.append(ep["mae"])
                    exs.append(ep["exact"] * 100.0)
                    ocs.append(ep["over_ch"] * 100.0)
                seed_maes.append(float(np.mean(maes)))
                seed_exacts.append(float(np.mean(exs)))
                seed_ocs.append(float(np.mean(ocs)))
            results["multiseed_sigma8"][cname][pol] = {
                "mae_mean": float(np.mean(seed_maes)),
                "mae_sd": float(np.std(seed_maes, ddof=1)),
                "exact_mean": float(np.mean(seed_exacts)),
                "oc_mean": float(np.mean(seed_ocs)),
            }
        r = results["multiseed_sigma8"][cname]
        print(
            f"{cname:20s} | RAQS={r['raqs']['mae_mean']:.3f}±{r['raqs']['mae_sd']:.3f} (Ex={r['raqs']['exact_mean']:.1f}%, OC={r['raqs']['oc_mean']:.1f}%) | "
            f"Fixed={r['fixed']['mae_mean']:.3f}±{r['fixed']['mae_sd']:.3f} (OC={r['fixed']['oc_mean']:.1f}%) | "
            f"Rand={r['random']['mae_mean']:.3f}±{r['random']['mae_sd']:.3f}"
        )

    # Overall mean across all 4 cohorts per seed
    print("--- OVERALL 4-COHORT MEAN ACROSS 5 SEEDS (SIGMA=8) ---")
    for pol in ("raqs", "fixed", "random"):
        vals = [results["multiseed_sigma8"][c][pol]["mae_mean"] for c in COHORT_MEANS]
        ocs = [results["multiseed_sigma8"][c][pol]["oc_mean"] for c in COHORT_MEANS]
        print(f"  {pol:8s}: MAE={np.mean(vals):.3f}, OC={np.mean(ocs):.1f}%")

    # 3. Multi-noise sweep: sigma in {4, 8, 12, 16} across 5 seeds (4 cohorts x 200 trials/seed)
    print("\n=== MULTI-NOISE SWEEP (SIGMA = 4, 8, 12, 16; 5 SEEDS x 200 TRIALS) ===")
    sigmas = [4, 8, 12, 16]
    for sig in sigmas:
        results["multinoise"][str(sig)] = {}
        for pol in ("raqs", "fixed", "random"):
            seed_means, seed_ocs = [], []
            for sd in SEEDS:
                rng = random.Random(sd)
                all_m, all_oc = [], []
                for base_means in COHORT_MEANS.values():
                    for _ in range(200):
                        sc = [max(0.0, min(100.0, rng.gauss(m, float(sig)))) for m in base_means]
                        ep = simulate_episode(sc, policy=pol, alpha=0.60, l1=2, rng=rng)
                        all_m.append(ep["mae"])
                        all_oc.append(ep["over_ch"] * 100.0)
                seed_means.append(float(np.mean(all_m)))
                seed_ocs.append(float(np.mean(all_oc)))
            results["multinoise"][str(sig)][pol] = {
                "mae_mean": float(np.mean(seed_means)),
                "mae_sd": float(np.std(seed_means, ddof=1)),
                "oc_mean": float(np.mean(seed_ocs)),
            }
        mn = results["multinoise"][str(sig)]
        print(
            f"sigma={sig:2d} | RAQS={mn['raqs']['mae_mean']:.3f}±{mn['raqs']['mae_sd']:.3f} (OC={mn['raqs']['oc_mean']:.1f}%) | "
            f"Fixed={mn['fixed']['mae_mean']:.3f}±{mn['fixed']['mae_sd']:.3f} (OC={mn['fixed']['oc_mean']:.1f}%) | "
            f"Rand={mn['random']['mae_mean']:.3f}±{mn['random']['mae_sd']:.3f}"
        )

    # 4. Ablation on l1 and alpha across 5 seeds (stationary cohorts, sigma=8)
    print("\n=== ABLATION (5 SEEDS, STATIONARY COHORTS, SIGMA=8) ===")
    for l1 in (1, 2):
        for a in (0.2, 0.4, 0.6, 0.8):
            seed_maes, seed_rmses = [], []
            for sd in SEEDS:
                rng = random.Random(sd)
                ms, sqs = [], []
                for m_val in (48.0, 71.0, 89.0):
                    true_th = m_val / 100.0
                    for _ in range(200):
                        sc = [max(0.0, min(100.0, rng.gauss(m_val, 8.0))) for _ in range(6)]
                        ep = simulate_episode(sc, policy="raqs", alpha=a, l1=l1)
                        ms.append(ep["mae"])
                        sqs.append((ep["theta6"] - true_th) ** 2)
                seed_maes.append(float(np.mean(ms)))
                seed_rmses.append(math.sqrt(float(np.mean(sqs))))
            key = f"l1={l1}_a={a:.1f}"
            results["ablation"][key] = {
                "mae": float(np.mean(seed_maes)),
                "rmse": float(np.mean(seed_rmses)),
                "rmse_sd": float(np.std(seed_rmses, ddof=1)),
            }
            print(f"{key:12s} | MAE={results['ablation'][key]['mae']:.3f} | RMSE={results['ablation'][key]['rmse']:.4f}±{results['ablation'][key]['rmse_sd']:.4f}")

    with open(os.path.join(MID_DIR, "raqs_sim_results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate Fig. 1: System Architecture & RAQS Feedback Loop
    draw_architecture_diagram(os.path.abspath(os.path.join(MID_DIR, "fig1_architecture.png")))

    # Generate Fig. 2: Proficiency trajectories (deterministic)
    fig, ax = plt.subplots(figsize=(3.45, 2.25), dpi=300)
    turns = [1, 2, 3, 4, 5, 6]
    palette = {
        "Advanced": ("#1B4F72", "o", "-"),
        "Intermediate": ("#1E8449", "s", "--"),
        "Mixed": ("#B7791F", "^", "-."),
        "Novice": ("#922B21", "D", ":"),
    }
    for name, (col, mk, ls) in palette.items():
        th = results["deterministic"][name]["raqs"]["thetas"]
        ax.plot(turns, th, color=col, marker=mk, linestyle=ls, linewidth=1.5, markersize=4.0, label=name)
    ax.axhline(0.80, color="#666666", linestyle=":", linewidth=0.8)
    ax.axhline(0.60, color="#666666", linestyle=":", linewidth=0.8)
    ax.text(6.08, 0.80, "Strong", fontsize=6.5, va="center", color="#444444")
    ax.text(6.08, 0.60, "Basic", fontsize=6.5, va="center", color="#444444")
    ax.set_xlim(0.85, 6.65)
    ax.set_ylim(0.38, 1.00)
    ax.set_xticks(turns)
    ax.set_xlabel("Interview Turn (t)")
    ax.set_ylabel("Proficiency Estimate (θ̂)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.48, 1.02), ncol=2, frameon=False)
    fig.tight_layout(pad=0.3)
    p2 = os.path.abspath(os.path.join(MID_DIR, "fig2_trajectories.png"))
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # Generate Fig. 3: Multi-noise MAE comparison (sigma = 4, 8, 12, 16; 5 seeds)
    fig, ax = plt.subplots(figsize=(3.45, 2.25), dpi=300)
    x = np.arange(len(sigmas))
    w = 0.25
    rv = [results["multinoise"][str(s)]["raqs"]["mae_mean"] for s in sigmas]
    fv = [results["multinoise"][str(s)]["fixed"]["mae_mean"] for s in sigmas]
    nv = [results["multinoise"][str(s)]["random"]["mae_mean"] for s in sigmas]
    b1 = ax.bar(x - w, rv, width=w, color="#1B4F72", label="RAQS (Ours)")
    b2 = ax.bar(x, fv, width=w, color="#A93226", label="Fixed Schedule")
    b3 = ax.bar(x + w, nv, width=w, color="#7F8C8D", label="Random Tier")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=6.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ = {s}" for s in sigmas])
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Overall Mean Tier MAE")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    p3 = os.path.abspath(os.path.join(MID_DIR, "fig3_multinoise_mae.png"))
    fig.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")

    # Generate Fig. 4: Alpha sensitivity across sigma = 4, 8, 12, 16
    fig, ax = plt.subplots(figsize=(3.45, 2.20), dpi=300)
    alpha_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for sig, col, mk, ls in [(4, "#1E8449", "o", "-"), (8, "#1B4F72", "s", "--"), (12, "#B7791F", "^", "-."), (16, "#922B21", "d", ":")]:
        rmses = []
        for a in alpha_grid:
            sqs = []
            for sd in SEEDS:
                rng = random.Random(sd)
                for m_val in (48.0, 71.0, 89.0):
                    true_th = m_val / 100.0
                    for _ in range(200):
                        sc = [max(0.0, min(100.0, rng.gauss(m_val, float(sig)))) for _ in range(6)]
                        ep = simulate_episode(sc, policy="raqs", alpha=a, l1=2)
                        sqs.append((ep["theta6"] - true_th) ** 2)
            rmses.append(math.sqrt(float(np.mean(sqs))))
        ax.plot(alpha_grid, rmses, color=col, marker=mk, linestyle=ls, linewidth=1.35, markersize=3.8, label=f"σ={sig}")
    ax.axvline(0.60, color="#555555", linestyle=":", linewidth=0.8)
    ax.text(0.615, 0.122, "α=0.60", fontsize=6.8, color="#333333")
    ax.set_xlabel("Smoothing Factor (α)")
    ax.set_ylabel("Final Proficiency RMSE (θ̂₆)")
    ax.set_ylim(0.02, 0.138)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, columnspacing=0.7, handletextpad=0.35)
    fig.tight_layout(pad=0.3)
    p4 = os.path.abspath(os.path.join(MID_DIR, "fig4_alpha_sensitivity.png"))
    fig.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p4}")


if __name__ == "__main__":
    main()