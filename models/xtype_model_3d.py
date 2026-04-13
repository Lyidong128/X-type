from numpy import *
import numpy.matlib as mtl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

#泡利矩阵
s0 = eye(2)
sx = array([[0, 1], [1, 0]], dtype=complex)
sy = array([[0, -1.j], [1.j, 0]], dtype=complex)
sz = array([[1, 0], [0, -1]], dtype=complex)


def calculate_reciprocal_lattice(a1, a2, a3):
    # 计算倒格矢
    cross_a2_a3 = cross(a2, a3)
    cross_a3_a1 = cross(a3, a1)
    cross_a1_a2 = cross(a1, a2)
    volume = dot(a1, cross_a2_a3)
    b1 = (2 * pi * cross_a2_a3) / volume
    b2 = (2 * pi * cross_a3_a1) / volume
    b3 = (2 * pi * cross_a1_a2) / volume
    return b1, b2, b3


# 基矢
a1 = array([1, 0, 0])
a2 = array([0, 1, 0])
a3 = array([0, 0, 2])
b1, b2, b3 = calculate_reciprocal_lattice(a1, a2, a3)

# 最近邻连接矢
r1 = 1 / 2 * a1
r2 = 1 / 2 * a2


def ph(k, r):
    return exp(1.j * dot(k, r))


# 次近邻连接矢
d1 = array([1 / 4, 1 / 4, 1])  # A3-A1
d2 = array([1 / 4, -1 / 4, 1])  # A2-A1
d3 = array([1 / 4, 1 / 4, -1])  # A4-A2
d4 = array([1 / 4, -1 / 4, -1])  # A4-A3

# 格点矢量
gk = b1 + b2 + b3
km = 2 * b1 + b2 + b3
mk2 = b1 + 2 * b2 + b3
k2g = b1 + b2 + 2 * b3

# 高对称点
G = array([0, 0, 0])
X = 0.5 * b1
Y = 0.5 * b2
M = 0.5 * b1 + 0.5 * b2
Z = 0.5 * b3

# K 点路径 Γ-X-Y-Γ-M-Γ-Z-Γ
gx = linspace(G, X, 500, endpoint=False)
xy = linspace(X, Y, 500, endpoint=False)
yg = linspace(Y, G, 500, endpoint=False)
gm = linspace(G, M, 500, endpoint=False)
mg = linspace(M, G, 500, endpoint=False)
gz = linspace(G, Z, 500, endpoint=False)
zg = linspace(Z, G, 500, endpoint=False)
PATH_SEGMENTS = [gx, xy, yg, gm, mg, gz, zg]
PATH_LABELS = [r"$\Gamma$", "X", "Y", r"$\Gamma$", "M", r"$\Gamma$", "Z", r"$\Gamma$"]

t = 0.3
v = 1
w = 1
lm = 0
J = 0


def Hxtype(k):
    # 格点能
    h_xtype = zeros((8, 8), dtype=complex)
    h0 = zeros((4, 4), dtype=complex)
    h00 = zeros((8, 8), dtype=complex)

    h0[0, 1] = t + t * ph(k, -a3)
    h0[0, 2] = t + t * ph(k, -a3)
    h0[0, 3] = v + w * ph(k, -a1)
    h0[1, 0] = t + t * ph(k, a3)
    h0[1, 2] = v + w * ph(k, -a2)
    h0[1, 3] = t + t * ph(k, a3)
    h0[2, 0] = t + t * ph(k, a3)
    h0[2, 1] = v + w * ph(k, a2)
    h0[2, 3] = t + t * ph(k, a3)
    h0[3, 0] = v + w * ph(k, a1)
    h0[3, 1] = t + t * ph(k, -a3)
    h0[3, 2] = t + t * ph(k, -a3)

    h00 = kron(h0, s0)

    h1 = zeros((4, 4), dtype=complex)
    h2 = zeros((4, 4), dtype=complex)
    hsoc = zeros((8, 8), dtype=complex)
    hj = zeros((8, 8), dtype=complex)
    h2[0, 0] = J
    h2[1, 1] = J
    h2[2, 2] = J
    h2[3, 3] = J

    h1[0, 1] = 1j * lm + 1j * lm * ph(k, -a3)
    h1[0, 2] = -1j * lm - 1j * lm * ph(k, -a3)
    h1[2, 0] = 1j * lm + 1j * lm * ph(k, a3)
    h1[2, 3] = -1j * lm - 1j * lm * ph(k, a3)
    h1[3, 2] = 1j * lm + 1j * lm * ph(k, -a3)
    h1[3, 1] = -1j * lm - 1j * lm * ph(k, -a3)
    h1[1, 0] = -1j * lm - 1j * lm * ph(k, a3)
    h1[1, 3] = 1j * lm + 1j * lm * ph(k, a3)

    hsoc = kron(h1, sz)
    hj = kron(h2, sz)
    h_xtype = h00 + hsoc + hj
    return h_xtype


def eHxtype(k):
    return linalg.eigh(Hxtype(k))[0]


def _occ_evecs(k, n_occ):
    _, evecs = linalg.eigh(Hxtype(k))
    return evecs[:, :n_occ]


def chern_number_fukui(nk=31, n_occ=4, k_origin=None):
    """
    Discretized Chern number on a k-grid using Fukui-Hatsugai-Suzuki method.
    Assumes a 2D BZ slice spanned by b1 and b2 at fixed kz (k_origin controls kz).
    """
    if k_origin is None:
        k_origin = array([0.0, 0.0, 0.0])

    ux = zeros((nk, nk), dtype=complex)
    uy = zeros((nk, nk), dtype=complex)

    occ = [[None for _ in range(nk)] for __ in range(nk)]
    for i in range(nk):
        u = i / nk
        for j in range(nk):
            v_ = j / nk
            k = k_origin + u * b1 + v_ * b2
            occ[i][j] = _occ_evecs(k, n_occ)

    def link(mat_a, mat_b):
        m = mat_a.conj().T @ mat_b
        detm = linalg.det(m)
        if abs(detm) < 1e-14:
            return 1.0 + 0.0j
        return detm / abs(detm)

    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            ux[i, j] = link(occ[i][j], occ[ip][j])
            uy[i, j] = link(occ[i][j], occ[i][jp])

    f12 = zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            f12[i, j] = log(ux[i, j] * uy[ip, j] / (ux[i, jp] * uy[i, j]))

    ch = f12.sum() / (2j * pi)
    return float(ch.real)


def run_band_and_topology_demo():
    eig_gx = array(list(map(eHxtype, gx)))
    eig_xy = array(list(map(eHxtype, xy)))
    eig_yg = array(list(map(eHxtype, yg)))
    eig_gm = array(list(map(eHxtype, gm)))
    eig_mg = array(list(map(eHxtype, mg)))
    eig_gz = array(list(map(eHxtype, gz)))
    eig_zg = array(list(map(eHxtype, zg)))

    bands = [
        hstack((eig_gx[:, b], eig_xy[:, b], eig_yg[:, b], eig_gm[:, b], eig_mg[:, b], eig_gz[:, b], eig_zg[:, b]))
        for b in range(8)
    ]
    colors = ["red", "blue", "pink", "purple", "orange", "green", "cyan", "magenta"]

    plt.figure(figsize=(6, 5))
    max_x = len(bands[0]) - 1
    plt.xlim(0, max_x)
    for band, color in zip(bands, colors):
        plt.plot(band, color=color)
    plt.xticks(
        [0, 500, 1000, 1500, 2000, 2500, 3000, max_x],
        [r"$\Gamma$", "X", "Y", r"$\Gamma$", "M", r"$\Gamma$", "Z", r"$\Gamma$"],
    )
    plt.ylabel("Energy")
    plt.show()

    try:
        ch = chern_number_fukui(nk=41, n_occ=4)
        print(f"[Topology] Chern number (nk=41, n_occ=4) ≈ {ch:.6f}")
    except Exception as e:
        print(f"[Topology] Chern number computation failed: {e}")


if __name__ == "__main__":
    run_band_and_topology_demo()
