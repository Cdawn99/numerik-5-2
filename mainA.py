import numpy as np
import matplotlib.pyplot as plt
import ArwpFdm as FDM

a, b = 0, 1
tend = 1
Nnod = 10+1
ort = np.array([a, b, Nnod])
h = (b-a)/(Nnod-1)

exa = lambda x,t: np.sin(x) * np.cos(x)

kco = lambda x: 1
qco = lambda x: 1
rco = lambda x: 1
fco = lambda x,t: (np.cos(x) + 2*np.sin(x)) * np.cos(t) - np.sin(t) * np.sin(x)

rba = lambda t: 0
rbb = lambda t: np.cos(t) * np.sin(b)
phi = lambda x: np.sin(x)

# Explizit Euler
sigma = 0
tau = h*h/2

# Implizit Euler
# sigma = 1
# tau = h*h

# Crank-Nicolson
# sigma = 1/2
# tau = h

M = round(tend/tau)
zeit = np.array([0, tend, M+1])

uw, xw, tw = FDM.ArwpFdm1d(ort, zeit, kco, rco, qco, fco, rba, rbb, phi, sigma)

uexa = -uw.copy()
for j in range(0, len(tw)):
    exat = exa(xw, tw[j])
    uexa[j,:] = exat
print(f'sigma={sigma}, h={h}, tau={tau}, MaxF={np.max(abs(uw-uexa))}')

Xw, Tw = np.meshgrid(xw, tw)

fig = plt.figure()

ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(Xw, Tw, exa(Xw, Tw), cmap=plt.cm.coolwarm)
ax.set(xlabel="Ort x", ylabel="Zeit t", title="exact - Surface plot")

ay = fig.add_subplot(122, projection='3d')
ay.plot_surface(Xw, Tw, uw, cmap=plt.cm.coolwarm)
ay.set(xlabel="Ort x", ylabel="Zeit t", title="FDM - Surface plot")

plt.show()
