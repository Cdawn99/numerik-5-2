import ArwpFdm as FDM
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print(f'Usage: {sys.argv[0]} <Method> <tau>')
    print("Method can be one of: explicit, implicit, nicolson")
    print("tau can be one of: lin, sq, sq/6")
    sys.exit()

a, b = 0, 1
tend = 1
Nnod = np.linspace(10+1, 20+1, 11)
h = (b-a)/(Nnod-1)


def exa(x, t): return np.exp(-2*t) * np.cos(np.pi*x)


def kco(x): return 1 + 2*x*x
def qco(x): return 2 + x*(1-x)
def rco(x): return x - 1/2


def fco(x, t):
    return np.exp(-2*t) * (np.cos(np.pi*x)
                           * (2*np.pi*np.pi*x*x - x*x + x + np.pi*np.pi)
                           + 3*np.pi*np.sin(np.pi*x)*(x+1/6))


def rba(t): return np.exp(-2*t)
def rbb(t): return -np.exp(-2*t)
def phi(x): return np.cos(np.pi*x)


if sys.argv[1] == "explicit":
    # Explizit Euler
    sigma = 0
elif sys.argv[1] == "implicit":
    # Implizit Euler
    sigma = 1
elif sys.argv[1] == "nicolson":
    # Crank-Nicolson
    sigma = 1/2
else:
    print("Invalid method! Must be one of: explicit, implicit, nicolson")
    sys.exit()

if sys.argv[2] == "lin":
    tau = h
elif sys.argv[2] == "sq":
    tau = h**2
elif sys.argv[2] == "sq/6":
    tau = h**2/6
else:
    print("Invalid tau! Must be one of: lin, sq, sq/6")
    sys.exit()

M = np.round(tend/tau)

fig = plt.figure()

ax = fig.add_subplot(111)

for i in range(len(Nnod)):
    ort = np.array([a, b, Nnod[i]])
    zeit = np.array([0, tend, M[i]+1])

    uw, xw, tw = FDM.ArwpFdm1d(ort, zeit,
                               kco, rco, qco, fco, rba, rbb, phi, sigma)

    uexa = -uw.copy()
    for j in range(0, len(tw)):
        exat = exa(xw, tw[j])
        uexa[j, :] = exat
    ax.plot(h[i], np.max(abs(uw-uexa)), 'ro')

ax.set(title="Fehlerverhalten", xlabel="h", ylabel="maxFehler")
ax.grid()

plt.show()
