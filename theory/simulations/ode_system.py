import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Параметры (пример)
params = {
    'eta': [0.8, 0.6, 0.4],
    'lambda_': [0.2, 0.5, 0.9],
    'kappa': [0.5, 0.5, 0.5],
    'delta': [0.1, 0.1, 0.1],
    'mu': [0.2, 0.2, 0.2],
    'El': [1.2, 1.1, 1.0],
    'c': [0.7, 0.7, 0.7],
    'beta': [0.9, 0.9, 0.9],
    'nu': 0.5,
    'epsilon': 0.01
}

def system(y, t, p):
    Gamma = y[:3]; alpha = y[3:]
    dG = np.zeros(3); da = np.zeros(3)
    # поток снизу
    for l in range(3):
        flow = 0
        for k in range(l):
            flow += p['beta'][k] * Gamma[k] * alpha[l] / (1 + p['nu']*abs(Gamma[k]-Gamma[l]))
        dG[l] = p['eta'][l] * flow * (1-alpha[l])**2 - p['lambda_'][l] * 0.1 * alpha[l] * Gamma[l]
        if l < 2: # downward causation
            dG[l] += 0.05 * Gamma[l+1]
    for l in range(3):
        da[l] = p['kappa'][l] * alpha[l]*(1-alpha[l])*(p['El'][l]-p['c'][l]) - p['delta'][l]*(1-alpha[l])*Gamma[l]
        for k in range(l):
            da[l] += p['mu'][l] * 0.5 * (Gamma[k]-Gamma[l])
    return np.concatenate([dG, da])

y0 = [1.0, 0.85, 0.65, 0.25, 0.45, 0.65]
t = np.linspace(0, 80, 200)
sol = odeint(system, y0, t, args=(params,))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
for i in range(3):
    plt.plot(t, sol[:,i], label=f'Γ_{i}')
plt.legend(); plt.title('Когерентность Γ')
plt.subplot(1,2,2)
for i in range(3):
    plt.plot(t, sol[:,3+i], label=f'α_{i}')
plt.legend(); plt.title('Интеграция α')
plt.tight_layout()
plt.show()
