import ArwpFdm as FDM
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <Method>')
    print("Method can be one of: explicit, implicit, nicolson")
    sys.exit()

a, b = 0, 1
tend = 1
Nnod = 10+1
ort = np.array([a, b, Nnod])
h = (b-a)/(Nnod-1)


def exa(x, t): return np.sin(x) * np.cos(t)


def kco(x): return 1
def qco(x): return 1
def rco(x): return 1
def fco(x, t): return (np.cos(x)+2*np.sin(x))*np.cos(t)-np.sin(t)*np.sin(x)


def rba(t): return 0
def rbb(t): return np.cos(t) * np.sin(b)
def phi(x): return np.sin(x)


if sys.argv[1] == "explicit":
    # Explizit Euler
    sigma = 0
    tau = h*h/2
elif sys.argv[1] == "implicit":
    # Implizit Euler
    sigma = 1
    tau = h*h
elif sys.argv[1] == "nicolson":
    # Crank-Nicolson
    sigma = 1/2
    tau = h
else:
    print("Invalid method! Must be one of: explicit, implicit, nicolson")
    sys.exit()

M = round(tend/tau)
zeit = np.array([0, tend, M+1])

uw, xw, tw = FDM.ArwpFdm1d(ort, zeit, kco, rco, qco, fco, rba, rbb, phi, sigma)

uexa = -uw.copy()
for j in range(0, len(tw)):
    exat = exa(xw, tw[j])
    uexa[j, :] = exat
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
