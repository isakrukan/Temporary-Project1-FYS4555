from problem3 import Muon_Pair_Production as Muon_par_old
from problem3 import Bhabha as Bhabha_old
import numpy as np
import matplotlib.pyplot as plt

class Muon_Pair_Production(Muon_par_old):

    def __init__(self, unit):
        Muon_par_old.__init__(self, unit)
    
    def cross_section(self, s):
        return self._unit * 4*np.pi*self.alpha**2 / (3*s)

    def expected_events(self, s, L, eff=0.5):
        return self.cross_section(s)*L*eff 
    
class Bhabha(Bhabha_old):

    def __init__(self, unit):
        Bhabha_old.__init__(self, unit)

    def cross_section(self, s,a=0.98):
        return self._unit * np.pi*self.alpha**2 / s * (1/3*a**3 + 16*a/(1 - a**2) + 9*a + np.log(abs(a-1)/(a+1)))

    def expected_events(self, s, L, eff=0.5, a=0.98):
        return self.cross_section(s, a)*L*eff 


if __name__ == "__main__":
    s_sqrt = np.linspace(0.1,40,1000)
    s = s_sqrt**2

    GeV_to_mb = 0.3894
    GeV_to_nb = GeV_to_mb*1e6
    GeV_to_mub = GeV_to_nb*1e-3

    bhabha = Bhabha(GeV_to_nb)
    muon_pair = Muon_Pair_Production(GeV_to_nb)


    plt.plot(s_sqrt, muon_pair.cross_section(s), color="red", label=r"$e^{+}e^{-} \rightarrow \mu^{+}\mu^{-}$")
    plt.plot(s_sqrt, bhabha.cross_section(s), color="black", label="Bhabha scattering")
    plt.yscale("log")
    plt.xlim((0,40))
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel(r"$\sigma$ [nb]")
    plt.legend()
    plt.grid()
    plt.savefig("figs/problem4.png")
    plt.show()

    plt.title("Absolute difference in total cross section")
    plt.plot(s_sqrt, np.abs(bhabha.cross_section(s) - muon_pair.cross_section(s)), color="black")
    plt.xlim((0,40))
    plt.ylim((0,1e3))
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel(r"$\sigma$ [nb]")
    plt.grid()
    plt.savefig("figs/problem4_2.png")
    plt.show()


    # Expected events:
    integrated_lum = 10 # [pb^-1]
    # integrated_lum *= 1e3
    GeV_to_pb = GeV_to_nb*1e3
    bhabha = Bhabha(GeV_to_pb)
    muon_pair = Muon_Pair_Production(GeV_to_pb)
    s = 14**2
    print(f"With an integrated luminosity of {integrated_lum} pb⁻1, s = 14^2 and a detactor efficiency of 50\% we get N expected events for:")
    print(f"Bhabha: N = {bhabha.expected_events(s, integrated_lum):.0f}")
    print(f"Muon pair: N = {muon_pair.expected_events(s, integrated_lum):.0f}")
    print(f"\nWhich gives a ratio of: {bhabha.expected_events(s, integrated_lum)/muon_pair.expected_events(s, integrated_lum):.0f}")

    # Ignore
    ax = plt.figure().add_subplot(projection='3d')
    plt.title("Number of expected events for Bhabha scattering")
    # X: s_sqrt, Y: L
    X, Y = np.mgrid[5:40:0.25, 1:100*np.pi:0.25]

    Z = bhabha.expected_events(X, Y)

    ax.plot_surface(X, Y, Z, cmap='autumn')

    ax.set_xlabel(r"$\sqrt{s}$")
    ax.set_ylabel("L")
    ax.set_zlabel("N")
    ax.set_zscale("log")

    plt.show()