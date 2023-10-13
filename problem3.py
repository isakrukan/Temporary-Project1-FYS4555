import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import warnings
warnings.filterwarnings("ignore") # The code creates a lot of warnings in the limit cos(theta) -> 1

class Muon_Pair_Production:
    alpha = 1/137

    def __init__(self, unit_conversion=1):
        self._unit = unit_conversion 

    def annihilation(self, theta, s):
        return self._unit *self.alpha**2 / (4*s) * (1 + np.cos(theta)**2)
    
    def scattering(self, theta, s):
        return self._unit * 2*self.alpha**2 / s * (1 + 1/4*(1 + np.cos(theta))**2) / (1 - np.cos(theta))**2
    

class Bhabha:
    alpha = 1/137

    def __init__(self, unit_conversion=1):
        self._unit = unit_conversion 

    def annihilation(self, theta, s):
        return self._unit * self.alpha**2 / (2*s) * (1 + np.cos(theta)** 2) / 2 

    def scattering(self, theta, s):
        return self._unit * self.alpha**2 / (2*s) * ((1+np.cos(theta))** 2 + 4) / (1 - np.cos(theta))**2

    def interference(self, theta, s):
        return self._unit * self.alpha**2 / (2*s) * (1+np.cos(theta))**2 / (1-np.cos(theta))


    def diff_cross_section(self, theta, s):
        return self.annihilation(theta, s) + self.scattering(theta, s) - self.interference(theta, s)

if __name__ == "__main__":

    s_sqrt = 14 # [GeV]
    s = s_sqrt**2
    theta = np.linspace(0, np.pi, 100_000)
    cos_theta = np.cos(theta)

    GeV_to_mb = 0.3894
    GeV_to_nb = GeV_to_mb*1e6

    bhabha = Bhabha(GeV_to_nb)
    muon_pair = Muon_Pair_Production(GeV_to_nb)
    
    plt.title(r"Differential cross section of Bhabha scattering with $\sqrt{s} = 14$ GeV")
    plt.plot(cos_theta, bhabha.annihilation(theta, s), color="blue", label="annihilation")
    plt.plot(cos_theta, bhabha.scattering(theta, s), color="orangered", label="scattering")
    plt.plot(cos_theta, bhabha.interference(theta, s), color="green", label="interference")
    plt.plot(cos_theta, bhabha.diff_cross_section(theta, s), color="black", label=r"$\frac{d\sigma}{d\Omega}$")
    plt.xlim((-1,1))
    plt.yscale("log")
    plt.ylim((1e-3, 1e3))
    plt.xlabel(r"$\cos{\theta}$")
    plt.ylabel(r"nb per steradian")
    plt.legend()
    plt.grid()
    plt.savefig("figs/problem3.png")
    plt.show()

    plt.title(r"Differential cross section for $\sqrt{s} = 14$ GeV")
    plt.plot(cos_theta, muon_pair.scattering(theta, s), color="red", label=r"$e^{+}e^{-} \rightarrow \mu^{+}\mu^{-}$, t-channel")
    plt.plot(cos_theta, muon_pair.annihilation(theta, s), color="navy", label=r"$e^{+}e^{-} \rightarrow \mu^{+}\mu^{-}$, s-channel")
    plt.plot(cos_theta, bhabha.scattering(theta, s), color="orangered", label=r"$e^{+}e^{-} \rightarrow e^{+}e^{-}$, t-channel")
    plt.plot(cos_theta, bhabha.annihilation(theta, s), color="blue", label=r"$e^{+}e^{-} \rightarrow e^{+}e^{-}$, s-channel")
    plt.plot(cos_theta, bhabha.interference(theta, s), color="green", label=r"$e^{+}e^{-} \rightarrow e^{+}e^{-}$, u-channel")
    plt.xlim((-1,1))
    plt.yscale("log")
    plt.xlabel(r"$\cos{\theta}$")
    plt.ylabel(r"$\frac{d\sigma}{d\Omega}$ (nb per steradian)")
    plt.ylim((1e-2, 1e3))
    plt.legend()
    plt.grid()
    plt.savefig("figs/problem3_2.png")
    plt.show()



    ax = plt.figure().add_subplot(projection='3d')
    plt.title("Differential cross section for Bhabha scattering")
    theta = np.linspace(0, np.pi)
    s_sqrt = np.linspace(1,40)

    THETA, S_SQRT = np.meshgrid(theta, s_sqrt)
    Z = GeV_to_nb*bhabha.diff_cross_section(THETA, S_SQRT**2)
    upper_bound = 1e-1
    indices_z = Z > upper_bound
    Z[indices_z] = upper_bound

    ax.plot_surface(np.cos(THETA), S_SQRT, Z, cmap='autumn')#, cstride=2, rstride=2)

    ax.set_ylabel(r"$\sqrt{s}$")
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_zlabel(r"$\sigma$")
    ax.set_zscale("log")

    plt.show()