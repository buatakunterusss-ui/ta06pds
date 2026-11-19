"""
TA-06 Pemodelan dan Simulasi - Modul 6 Dinamik System Modelling
Studi kasus : Dinamika suhu ruangan dengan pemanas (neraca energi, ODE orde-1)
Penulis     : Najib Fadhil Akbar (301240019)

Model:  C dT/dt = c*u(t) + UA*(T_out - T)
  C     : kapasitas panas ruangan   (J/K)
  UA    : konduktansi termal dinding (W/K)
  c     : daya pemanas per 1% bukaan (W/%)
  u(t)  : sinyal kontrol pemanas     (%)
  T_out : suhu luar (gangguan)       (degC)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Parameter ----------------
C = 7.2e5        # J/K
UA = 150.0       # W/K
Q_MAX = 1500.0   # W pada u = 100 %
c = Q_MAX / 100  # W/%
T_OUT = 20.0     # degC
T0 = 20.0        # degC
dt = 10.0        # s
t_end = 6 * 3600 # 6 jam
t = np.arange(0, t_end + dt, dt)
tau = C / UA     # konstanta waktu (s)


def simulate(u_func, tout_func=lambda tt: T_OUT):
    """Integrasi Euler eksplisit."""
    T = np.zeros_like(t)
    u = np.zeros_like(t)
    T[0] = T0
    u[0] = u_func(t[0])
    for i in range(1, len(t)):
        dTdt = (c * u[i - 1] + UA * (tout_func(t[i - 1]) - T[i - 1])) / C
        T[i] = T[i - 1] + dTdt * dt
        u[i] = u_func(t[i])
    return T, u


# ---------------- Sinyal kontrol ----------------
t_on = 600.0
step = lambda tt: 50.0 if tt >= t_on else 0.0
impulse = lambda tt: 100.0 if t_on <= tt < t_on + 60 else 0.0   # pulsa 60 s (energi 90 kJ)
ramp = lambda tt: min(0.01 * (tt - t_on), 100.0) if tt >= t_on else 0.0
sinus = lambda tt: 50.0 + 30.0 * np.sin(2 * np.pi * tt / 3600.0)

# ---------------- 1. Steady state ----------------
u_ss = 40.0
T_ss = T_OUT + c * u_ss / UA
T_a, u_a = simulate(lambda tt: u_ss)
print(f"tau = {tau:.0f} s ({tau/60:.0f} menit)")
print(f"T_ss teori = {T_ss:.2f} C ; T simulasi akhir = {T_a[-1]:.3f} C")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t / 3600, T_a, "b", label="T(t) simulasi")
ax.axhline(T_ss, color="orange", ls="--", label=f"Steady state ({T_ss:.0f} °C)")
ax.axvline(3 * tau / 3600, color="gray", ls=":", label="3τ")
ax.set(xlabel="Waktu (jam)", ylabel="Suhu ruangan (°C)", title="Steady state, u = 40 %")
ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig("grafik_steady_state.png", dpi=150)

# ---------------- 2. Respon terhadap sinyal kontrol ----------------
signals = [("Step (50 % mulai t = 600 s)", step),
           ("Impulse (pulsa 100 % selama 60 s)", impulse),
           ("Ramp (0,01 %/s mulai t = 600 s)", ramp),
           ("Sinusoida (50 % ± 30 %, periode 1 jam)", sinus)]
fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
for row, (name, f) in zip(axes, signals):
    T, u = simulate(f)
    row[0].plot(t / 3600, T, "b"); row[0].set_ylabel("T (°C)"); row[0].set_title(name, fontsize=9)
    row[1].plot(t / 3600, u, "r--"); row[1].set_ylabel("u (%)")
    for a in row: a.grid(True)
    print(f"{name}: T_maks = {T.max():.2f} C pada t = {t[T.argmax()]/60:.0f} menit; T_akhir = {T[-1]:.2f} C")
axes[-1][0].set_xlabel("t (jam)"); axes[-1][1].set_xlabel("t (jam)")
fig.tight_layout(); fig.savefig("grafik_respon_sinyal.png", dpi=150)

# ---------------- 3. Gangguan suhu luar ----------------
tout_step = lambda tt: 20.0 if tt < 3600 else 28.0
T_d, _ = simulate(lambda tt: u_ss, tout_step)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t / 3600, T_d, "b", label="T(t)")
ax.plot(t / 3600, [tout_step(x) for x in t], "g--", label="T_out (gangguan)")
ax.set(xlabel="Waktu (jam)", ylabel="Suhu (°C)", title="Respon terhadap gangguan suhu luar (u = 40 %)")
ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig("grafik_gangguan.png", dpi=150)
print(f"Gangguan: T_akhir = {T_d[-1]:.2f} C (teori {28 + c*u_ss/UA:.2f} C)")

# ---------------- 4. Tabel perhitungan manual (dt = 60 s, step 50 %) ----------------
print("\nTabel manual Euler dt = 60 s, u = 50 %:")
Tm = T0
for k in range(6):
    d = (c * 50 + UA * (T_OUT - Tm)) / C
    print(f"{k*60:4d} s | T = {Tm:.5f} | dT/dt = {d:.6f} | T(k+1) = {Tm + d*60:.5f}")
    Tm += d * 60
