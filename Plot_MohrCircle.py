# ══════════════════════════════════════════════════════════════════════════════
# By Ashish Bahuguna, email: bahugunaashish92@gmail.com 
# ══════════════════════════════════════════════════════════════════════════════


import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from numpy.linalg import lstsq
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker

# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#f4f4f4"
PANEL     = "#ffffff"
CARD      = "#f0f0f0"
ACCENT    = "#2563eb"
ACCENT2   = "#dc2626"
TEXT      = "#111111"
MUTED     = "#555555"
SUCCESS   = "#16a34a"
ERROR     = "#dc2626"
BORDER    = "#d1d1d1"

CIRCLE_COLORS = [
    "#4f9cf9", "#4fcf8f", "#f97b4f",
    "#c97bf9", "#f9d44f", "#f94f8a",
    "#4ff9e2", "#f9a84f",
]

FONT_TITLE  = ("Arial", 13, "bold")
FONT_LABEL  = ("Arial", 10)
FONT_SMALL  = ("Arial",  9)
FONT_RESULT = ("Arial", 11, "bold")
FONT_MONO   = ("Arial", 10)

# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_envelope(centers, radii):
    """Least-squares common tangent: r_i = A + B·cx_i  =>  phi', c'."""
    if len(centers) < 2:
        return None, None
    X = np.column_stack([np.ones_like(centers), centers])
    (A, B), *_ = lstsq(X, radii, rcond=None)
    B = np.clip(B, -0.9999, 0.9999)
    phi_rad = np.arcsin(B)
    c_prime = A / np.cos(phi_rad)
    return phi_rad, c_prime


def tangent_point(cx, r, phi_rad):
    tx = cx - r * np.sin(phi_rad)
    ty =      r * np.cos(phi_rad)
    return tx, ty


def shear_strength(c_prime, phi_rad, sigma_eff):
    return c_prime + sigma_eff * np.tan(phi_rad)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class MohrApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Mohr-Coulomb Analyser")
        self.configure(bg=BG)
        self.state("zoomed")           # maximise on open

        # ── style ──────────────────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT,
                        fieldbackground=CARD, font=FONT_LABEL)
        style.configure("TScrollbar", background=PANEL, troughcolor=BG,
                        arrowcolor=MUTED)

        # ── soil / layer params (mutable) ───────────────────────────────────
        self.gamma_sat = tk.DoubleVar(value=19.0)
        self.gamma_w   = tk.DoubleVar(value=9.81)
        self.depth     = tk.DoubleVar(value=2.0)

        # ── test rows: list of (sigma3_var, delta_var, pore_var) ────────────
        self.rows: list[tuple] = []

        # ── build UI ────────────────────────────────────────────────────────
        self._build_layout()
        self._add_row()          # start with one empty row
        self._load_example()     # fill example data
        self._zoom_active = False
        self._zoom_rect   = None
        self._press_xy    = None
        self._zoom_btn    = None
        self._update_plot()
    
    # ── Exit ───────────────────────────────────────────────────────────────
    def _exit_app(self):
        self.destroy()

    def _zoom_reset(self):
        self._update_plot()
    # ── LAYOUT ───────────────────────────────────────────────────────────────
    def _build_layout(self):
        # left panel + right plot side by side
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg=BG, width=380)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)

        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_left(left)
        self._build_right(right)

    # ── LEFT PANEL ────────────────────────────────────────────────────────────

    def _build_left(self, parent):
        parent.rowconfigure(3, weight=0)
        parent.columnconfigure(0, weight=1)

        # ── title bar ──────────────────────────────────────────────────────
        title_bar = tk.Frame(parent, bg=ACCENT, height=48)
        title_bar.grid(row=0, column=0, sticky="ew")
        title_bar.grid_propagate(False)
        tk.Label(title_bar, text="⬡  MOHR-COULOMB ANALYSER",
                 bg=ACCENT, fg="#fff", font=FONT_TITLE,
                 anchor="w", padx=14).pack(fill="x", pady=10)

        # ── soil params ────────────────────────────────────────────────────
        soil_frame = tk.Frame(parent, bg=PANEL)
        soil_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 4))
        soil_frame.columnconfigure((1, 3), weight=1)

        tk.Label(soil_frame, text="SOIL PARAMETERS",
                 bg=PANEL, fg=MUTED, font=FONT_SMALL).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(8, 4))

        params = [
            ("γ_sat (kN/m³)", self.gamma_sat, "Depth z (m)", self.depth),
            ("γ_w  (kN/m³)",  self.gamma_w,  None,           None),
        ]
        for r, (l1, v1, l2, v2) in enumerate(params):
            tk.Label(soil_frame, text=l1, bg=PANEL, fg=TEXT,
                     font=FONT_SMALL).grid(row=r+1, column=0, sticky="w",
                                           padx=(10,4), pady=3)
            e = tk.Entry(soil_frame, textvariable=v1, width=8,
                         bg=CARD, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=FONT_MONO,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         highlightcolor=ACCENT)
            e.grid(row=r+1, column=1, sticky="ew", padx=(0,12), pady=3)
            e.bind("<KeyRelease>", lambda e: self._update_plot())
            if l2:
                tk.Label(soil_frame, text=l2, bg=PANEL, fg=TEXT,
                         font=FONT_SMALL).grid(row=r+1, column=2, sticky="w",
                                               padx=(0,4), pady=3)
                e2 = tk.Entry(soil_frame, textvariable=v2, width=8,
                              bg=CARD, fg=TEXT, insertbackground=TEXT,
                              relief="flat", font=FONT_MONO,
                              highlightthickness=1,
                              highlightbackground=BORDER,
                              highlightcolor=ACCENT)
                e2.grid(row=r+1, column=3, sticky="ew", padx=(0,10), pady=3)
                e2.bind("<KeyRelease>", lambda e: self._update_plot())

        tk.Frame(soil_frame, bg=BORDER, height=1).grid(
            row=10, column=0, columnspan=4, sticky="ew", padx=10, pady=6)

        # ── column headers ────────────────────────────────────────────────
        hdr = tk.Frame(parent, bg=BG)
        hdr.grid(row=2, column=0, sticky="ew", padx=10, pady=(4,0))
        for c, (txt, w) in enumerate([
                ("#", 3), ("σ₃  (kPa)", 9), ("Δσf (kPa)", 9), ("u   (kPa)", 9), ("", 3)
        ]):
            tk.Label(hdr, text=txt, bg=BG, fg=MUTED,
                     font=FONT_SMALL, width=w, anchor="center").grid(
                row=0, column=c, padx=2)

        # ── scrollable test rows ───────────────────────────────────────────
        # scroll_outer = tk.Frame(parent, bg=BG)
        scroll_outer = tk.Frame(parent, bg=BG, height=180)
        scroll_outer.grid_propagate(False)
        scroll_outer.grid(row=3, column=0, sticky="ew", padx=10, pady=2)
        scroll_outer.rowconfigure(0, weight=1)
        scroll_outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_outer, bg=BG, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(scroll_outer, orient="vertical",
                           command=canvas.yview)
        sb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=sb.set)

        self.rows_frame = tk.Frame(canvas, bg=BG)
        self.rows_frame.bind("<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")

        # ── bottom buttons ─────────────────────────────────────────────────
        btn_frame = tk.Frame(parent, bg=PANEL)
        btn_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=4)

        # self._btn(btn_frame, "+ ADD ROW", self._add_row,
        #           ACCENT, "#fff").pack(side="left", padx=6, pady=6)
        # self._btn(btn_frame, "CLEAR ALL", self._clear_all,
        #           CARD, ERROR).pack(side="left", padx=2, pady=6)
        # self._btn(btn_frame, "EXAMPLE",   self._load_example,
        #           CARD, SUCCESS).pack(side="left", padx=2, pady=6)
        # self._btn(btn_frame, "EXIT", self._exit_app, 
        #           ERROR, "#fff").pack(side="right", padx=6, pady=6)
        # self._btn(btn_frame, "🔍+", self._zoom_in,    
        #           CARD, ACCENT).pack(side="left", padx=2, pady=6)
        # self._btn(btn_frame, "🔍-", self._zoom_out,   
        #           CARD, ACCENT).pack(side="left", padx=2, pady=6)
        # self._btn(btn_frame, "⟳",   self._zoom_reset, 
        #           CARD, MUTED ).pack(side="left", padx=2, pady=6)
        self._btn(btn_frame, "+ ADD ROW", self._add_row,
          ACCENT, "#fff").pack(side="left", padx=6, pady=6)
        self._btn(btn_frame, "CLEAR ALL", self._clear_all,
                CARD, ERROR).pack(side="left", padx=2, pady=6)
        self._btn(btn_frame, "EXAMPLE",   self._load_example,
                CARD, SUCCESS).pack(side="left", padx=2, pady=6)
        self._btn(btn_frame, "EXIT", self._exit_app, 
                ERROR, "#fff").pack(side="left", padx=6, pady=6)
        # ── results panel ─────────────────────────────────────────────────
        res = tk.Frame(parent, bg=PANEL)
        res.grid(row=5, column=0, sticky="ew", padx=10, pady=(4, 10))
        res.columnconfigure(1, weight=1)

        tk.Label(res, text="RESULTS", bg=PANEL, fg=MUTED,
                 font=FONT_SMALL).grid(row=0, column=0, columnspan=2,
                                       sticky="w", padx=10, pady=(8, 2))

        self.lbl_phi  = self._result_lbl(res, "φ' =", 1)
        self.lbl_c    = self._result_lbl(res, "c' =", 2)
        self.lbl_tau  = self._result_lbl(res, "τf (mid-layer) =", 3)
        self.lbl_info = tk.Label(res, text="", bg=PANEL, fg=MUTED,
                                 font=FONT_SMALL, justify="left")
        self.lbl_info.grid(row=4, column=0, columnspan=2,
                           sticky="w", padx=10, pady=(0, 8))

    def _result_lbl(self, parent, caption, row):
        tk.Label(parent, text=caption, bg=PANEL, fg=MUTED,
                 font=FONT_SMALL).grid(row=row, column=0,
                                       sticky="w", padx=(10, 6), pady=2)
        lbl = tk.Label(parent, text="—", bg=PANEL, fg=ACCENT,
                       font=FONT_RESULT)
        lbl.grid(row=row, column=1, sticky="w", pady=2)
        return lbl

    @staticmethod
    def _btn(parent, text, cmd, bg, fg):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg=fg, relief="flat",
                      font=FONT_SMALL, padx=10, pady=4,
                      activebackground=ACCENT, activeforeground="#fff",
                      cursor="hand2")
        return b

    # ── RIGHT PLOT PANEL ──────────────────────────────────────────────────────

    def _build_right(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)
        parent.columnconfigure(0, weight=1)

        plt.rcParams.update({
            "figure.facecolor":  "#f4f4f4",
            "axes.facecolor":    "#ffffff",
            "axes.edgecolor":    "#aaaaaa",
            "axes.labelcolor":   "#111111",
            "xtick.color":       "#333333",
            "ytick.color":       "#333333",
            "grid.color":        "#dddddd",
            "text.color":        "#111111",
            "legend.facecolor":  "#ffffff",
            "legend.edgecolor":  "#cccccc",
            "font.family":       "sans-serif",
        })

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.fig.subplots_adjust(left=0.09, right=0.97,
                                top=0.92, bottom=0.10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    # ── ROW MANAGEMENT ────────────────────────────────────────────────────────

    def _add_row(self, sigma3=None, delta=None, pore=None):
        idx = len(self.rows)
        row_frame = tk.Frame(self.rows_frame, bg=BG)
        row_frame.pack(fill="x", pady=1)

        # colour swatch
        col = CIRCLE_COLORS[idx % len(CIRCLE_COLORS)]
        swatch = tk.Frame(row_frame, bg=col, width=4, height=28)
        swatch.pack(side="left", padx=(0, 4))

        # test number
        tk.Label(row_frame, text=f"{idx+1}", width=3,
                 bg=BG, fg=MUTED, font=FONT_SMALL,
                 anchor="center").pack(side="left", padx=2)

        v_s3    = tk.StringVar(value="" if sigma3 is None else str(sigma3))
        v_delta = tk.StringVar(value="" if delta  is None else str(delta))
        v_pore  = tk.StringVar(value="" if pore   is None else str(pore))

        for var in (v_s3, v_delta, v_pore):
            e = tk.Entry(row_frame, textvariable=var, width=9,
                         bg=CARD, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=FONT_MONO,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         highlightcolor=col)
            e.pack(side="left", padx=3, pady=3)
            e.bind("<KeyRelease>", lambda ev: self._update_plot())

        # delete button
        def _del(rf=row_frame, r=(v_s3, v_delta, v_pore)):
            self.rows.remove(r)
            rf.destroy()
            self._renumber()
            self._update_plot()

        tk.Button(row_frame, text="✕", command=_del,
                  bg=BG, fg=ERROR, relief="flat",
                  font=FONT_SMALL, padx=4,
                  cursor="hand2").pack(side="left", padx=2)

        triplet = (v_s3, v_delta, v_pore)
        self.rows.append(triplet)

    def _renumber(self):
        """Re-label the # column and update swatch colours after deletion."""
        for frame in self.rows_frame.winfo_children():
            frame.destroy()
        saved = list(self.rows)
        self.rows.clear()
        for i, (vs, vd, vp) in enumerate(saved):
            self._add_row(vs.get() or None,
                          vd.get() or None,
                          vp.get() or None)

    def _clear_all(self):
        for frame in self.rows_frame.winfo_children():
            frame.destroy()
        self.rows.clear()
        self._add_row()
        self._update_plot()

    def _load_example(self):
        for frame in self.rows_frame.winfo_children():
            frame.destroy()
        self.rows.clear()
        data = [(100, 170, -15), (200, 260, -40), (300, 360, -80)]
        for s3, d, p in data:
            self._add_row(s3, d, p)
        self.gamma_sat.set(19.0)
        self.gamma_w.set(9.81)
        self.depth.set(2.0)
        self._update_plot()

    # ── PLOT UPDATE ───────────────────────────────────────────────────────────

    def _parse_rows(self):
        sigma3_list, delta_list, pore_list = [], [], []
        for v_s3, v_d, v_p in self.rows:
            try:
                s3 = float(v_s3.get())
                d  = float(v_d.get())
                p  = float(v_p.get())
                sigma3_list.append(s3)
                delta_list.append(d)
                pore_list.append(p)
            except ValueError:
                pass
        return (np.array(sigma3_list),
                np.array(delta_list),
                np.array(pore_list))

    def _update_plot(self, *_):
        sigma3_arr, delta_arr, pore_arr = self._parse_rows()

        ax = self.ax
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.axhline(0, color=BORDER, linewidth=0.8)
        ax.axvline(0, color=BORDER, linewidth=0.8)
        ax.set_xlabel("Effective Normal Stress  σ' (kPa)", fontsize=10)
        ax.set_ylabel("Shear Stress  τ (kPa)", fontsize=10)
        ax.tick_params(labelsize=8)

        n = len(sigma3_arr)

        if n == 0:
            ax.set_title("No valid test data", color=MUTED, fontsize=11)
            ax.set_xlim(-20, 300)
            ax.set_ylim(-5, 100)
            self.canvas.draw_idle()
            self.lbl_phi.config(text="—")
            self.lbl_c.config(text="—")
            self.lbl_tau.config(text="—")
            self.lbl_info.config(text="Enter at least 1 test row.")
            return

        # effective stresses
        sigma3_eff = sigma3_arr - pore_arr
        sigma1_eff = sigma3_arr + delta_arr - pore_arr
        centers    = (sigma1_eff + sigma3_eff) / 2
        radii      = (sigma1_eff - sigma3_eff) / 2

        # axis limits
        max_x = max(sigma1_eff) * 1.12
        max_y = max(radii)      * 1.25
        ax.set_xlim(-max_x * 0.04, max_x)
        ax.set_ylim(-max_y * 0.04, max_y)

        theta = np.linspace(0, np.pi, 400)

        # ── draw circles ──────────────────────────────────────────────────
        for i, (cx, r) in enumerate(zip(centers, radii)):
            col = CIRCLE_COLORS[i % len(CIRCLE_COLORS)]
            x   = cx + r * np.cos(theta)
            y   =      r * np.sin(theta)
            ax.plot(x, y, color=col, linewidth=1.8,
                    label=f"Test {i+1}  σ₃'={sigma3_eff[i]:.0f},"
                          f" σ₁'={sigma1_eff[i]:.0f} kPa",
                    zorder=3)
            # diameter on x-axis (dashed)
            ax.plot([cx - r, cx + r], [0, 0],
                    color=col, linewidth=0.5,
                    linestyle="--", alpha=0.35, zorder=2)

        # ── envelope (need ≥ 2 circles) ───────────────────────────────────
        phi_rad = c_prime = None
        if n >= 2:
            phi_rad, c_prime = compute_envelope(centers, radii)

            if phi_rad is not None and np.isfinite(phi_rad) and np.isfinite(c_prime):
                phi_deg = np.degrees(phi_rad)

                # tangent points
                for i, (cx, r) in enumerate(zip(centers, radii)):
                    col     = CIRCLE_COLORS[i % len(CIRCLE_COLORS)]
                    tx, ty  = tangent_point(cx, r, phi_rad)
                    ax.vlines(tx, 0, ty, linestyles="dotted",
                              colors=col, linewidth=0.8, alpha=0.7)
                    ax.plot(tx, ty, "o", color=col, markersize=6,
                            markeredgecolor=BG, markeredgewidth=0.8, zorder=5)

                # envelope line
                sigma_env = np.array([-max_x * 0.04, max_x])
                tau_env   = c_prime + sigma_env * np.tan(phi_rad)
                ax.plot(sigma_env, tau_env, color=ACCENT2,
                        linewidth=2, linestyle="-",
                        label=f"Failure envelope  "
                              f"c'={c_prime:.1f} kPa, φ'={phi_deg:.1f}°",
                        zorder=6)

                # c' intercept
                if 0 <= c_prime <= max_y:
                    ax.plot(0, c_prime, "^",
                            color=ACCENT2, markersize=8, zorder=7)
                    ax.annotate(f"c'={c_prime:.1f} kPa",
                                xy=(0, c_prime),
                                xytext=(max_x * 0.06, c_prime + max_y * 0.06),
                                color=ACCENT2, fontsize=8,
                                arrowprops=dict(arrowstyle="->",
                                                color=ACCENT2, lw=1))

                # φ' arc
                arc_r  = min(max_x * 0.09, 60)
                arc_th = np.linspace(0, phi_rad, 120)
                ax.plot(arc_r * np.cos(arc_th),
                        c_prime + arc_r * np.sin(arc_th),
                        color=MUTED, linewidth=0.8)
                ax.text(arc_r * 0.55, c_prime + arc_r * 0.28,
                        f"φ'={phi_deg:.1f}°",
                        color=MUTED, fontsize=8)

                # update result labels
                self.lbl_phi.config(
                    text=f"{phi_deg:.2f}°")
                self.lbl_c.config(
                    text=f"{c_prime:.2f} kPa")

                # mid-layer shear strength
                try:
                    gs  = float(self.gamma_sat.get())
                    gw  = float(self.gamma_w.get())
                    z   = float(self.depth.get())
                    s_eff = (gs - gw) * z
                    tau_f = shear_strength(c_prime, phi_rad, s_eff)
                    self.lbl_tau.config(
                        text=f"{tau_f:.2f} kPa")
                    self.lbl_info.config(
                        text=f"σ'v = {s_eff:.2f} kPa at z = {z} m")

                    # plot point
                    ax.plot(s_eff, tau_f, "*",
                            color=ERROR, markersize=13, zorder=7,
                            label=f"Mid-layer τf = {tau_f:.1f} kPa")
                    ax.vlines(s_eff, 0, tau_f,
                              linestyles="dotted", colors=ERROR,
                              linewidth=1)
                    ax.hlines(tau_f, 0, s_eff,
                              linestyles="dotted", colors=ERROR,
                              linewidth=1)
                    ax.annotate(f"τf={tau_f:.1f} kPa",
                                xy=(s_eff, tau_f),
                                xytext=(s_eff + max_x*0.06,
                                        tau_f + max_y*0.05),
                                color=ERROR, fontsize=8,
                                arrowprops=dict(arrowstyle="->",
                                                color=ERROR, lw=1))
                except (ValueError, ZeroDivisionError):
                    self.lbl_tau.config(text="—")
                    self.lbl_info.config(text="Check soil parameters.")
            else:
                self.lbl_phi.config(text="—")
                self.lbl_c.config(text="—")
                self.lbl_tau.config(text="—")
                self.lbl_info.config(text="Could not fit envelope.")

        elif n == 1:
            self.lbl_phi.config(text="—")
            self.lbl_c.config(text="—")
            self.lbl_tau.config(text="—")
            self.lbl_info.config(text="Add ≥ 2 tests for envelope.")

        # ── title & legend ────────────────────────────────────────────────
        ax.set_title("Mohr-Coulomb Failure Envelope — CU Triaxial Tests",
                     color=TEXT, fontsize=11, pad=10)
        ax.set_aspect("equal", adjustable="datalim")
        leg = ax.legend(loc="upper left", fontsize=7.5,
                        framealpha=0.85, edgecolor=BORDER)
        for t in leg.get_texts():
            t.set_color(TEXT)

        self.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MohrApp()
    app.mainloop()

