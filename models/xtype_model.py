from numpy import *
import numpy.matlib as mtl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

#泡利矩阵
s0=eye(2)
sx=array([[0,1],[1,0]],dtype=complex)
sy=array([[0,-1.j],[1.j,0]],dtype=complex)
sz=array([[1,0],[0,-1]],dtype=complex)

#基矢
# 倒格矢
def calculate_reciprocal_lattice(a1, a2, a3):
    # 计算叉积
    cross_a2_a3 = cross(a2, a3)
    cross_a3_a1 = cross(a3, a1)
    cross_a1_a2 = cross(a1, a2)
    
    # 计算体积 V = a1 · (a2 × a3)
    volume = dot(a1, cross_a2_a3)
    
    # 计算倒格矢
    b1 = (2 * pi * cross_a2_a3) / volume
    b2 = (2 * pi * cross_a3_a1) / volume
    b3 = (2 * pi * cross_a1_a2) / volume
    
    return b1, b2, b3

a1=array([1,0,0])
a2=array([0,1,0])
a3=array([0,0,2])
b1, b2, b3 = calculate_reciprocal_lattice(a1, a2, a3)
#最近邻连接矢
r1=1/2*a1
r2=1/2*a2
#连接矢对应的相位部分
def ph(k,r):
    return exp(1.j*dot(k,r))

# 次近邻连接矢
d1=array([1/4,1/4,1]) #A3-A1
d2=array([1/4,-1/4,1]) #A2-A1
d3=array([1/4,1/4,-1]) #A4-A2
d4=array([1/4,-1/4,-1]) #A4-A3

# 格点矢量
gk = b1+b2+b3
km = 2*b1+b2+b3
mk2 = b1+2*b2+b3
k2g = b1+b2+2*b3

#高对称点
G=array([0,0,0])
X=0.5*b1
Y=0.5*b2
M=0.5*b1+0.5*b2





#K点路径G-X-Y-G
gx = linspace(G,X,500,endpoint=False)
xy = linspace(X,Y,500,endpoint=False)
yg = linspace(Y,G,500,endpoint=False)
gm = linspace(G,M,500,endpoint=False)
mg = linspace(M,G,500,endpoint=False)


t= 0.3
v= 1
w= 1
lm=0
J=0
       


def Hxtype(k):
 #格点能
    Hxtype=zeros((8,8),dtype=complex)
    H0=zeros((4,4),dtype=complex)
    H00=zeros((8,8),dtype=complex)
    H0[0,1]=t
    H0[0,2]=t
    H0[0,3]=v+w*ph(k,-a1)
    H0[1,0]=t
    H0[1,2]=v+w*ph(k,-a2)
    H0[1,3]=t
    H0[2,0]=t
    H0[2,1]=v+w*ph(k,a2)
    H0[2,3]=t
    H0[3,0]=v+w*ph(k,a1)
    H0[3,1]=t
    H0[3,2]=t


    H00=kron(H0,s0)

    H1=zeros((4,4),dtype=complex)
    H3=zeros((4,4),dtype=complex)
    HSOC=zeros((8,8),dtype=complex)
    HJ=zeros((8,8),dtype=complex)
    H3[0,0]=J
    H3[1,1]=-J
    H3[2,2]=J
    H3[3,3]=-J
    H1[0,1]=1j*lm
    H1[0,2]=-1j*lm
    H1[2,3]=-1j*lm
    H1[3,2]=1j*lm
    H1[3,1]=-1j*lm
    H1[1,0]=-1j*lm
    H1[2,0]=1j*lm
    H1[1,3]=1j*lm
    HSOC=kron(H1,sz)
    HJ=kron(H3,sz)
    Hxtype=H00+HSOC+HJ

    return Hxtype

def eHxtype(k):
    return linalg.eigh(Hxtype(k))[0]
Eig_gx = array(list(map(eHxtype,gx)))
Eig_xy = array(list(map(eHxtype,xy)))
Eig_yg = array(list(map(eHxtype,yg)))
Eig_gm = array(list(map(eHxtype,gm)))
Eig_mg = array(list(map(eHxtype,mg)))

def _occ_evecs(k, n_occ):
    # columns of evecs are eigenvectors
    evals, evecs = linalg.eigh(Hxtype(k))
    return evecs[:, :n_occ]

def chern_number_fukui(nk=31, n_occ=4, k_origin=None):
    """
    Discretized Chern number on a k-grid using Fukui-Hatsugai-Suzuki method.
    Assumes a 2D BZ spanned by b1 and b2 (kz fixed).
    """
    if k_origin is None:
        k_origin = array([0.0, 0.0, 0.0])

    # k = k_origin + u*b1 + v*b2, u,v in [0,1)
    Ux = zeros((nk, nk), dtype=complex)
    Uy = zeros((nk, nk), dtype=complex)

    occ = [[None for _ in range(nk)] for __ in range(nk)]
    for i in range(nk):
        u = i / nk
        for j in range(nk):
            v_ = j / nk
            k = k_origin + u * b1 + v_ * b2
            occ[i][j] = _occ_evecs(k, n_occ)

    def link(mat_a, mat_b):
        # gauge-invariant overlap determinant for multi-band occupied subspace
        m = mat_a.conj().T @ mat_b
        detm = linalg.det(m)
        if abs(detm) < 1e-14:
            return 1.0 + 0.0j
        return detm / abs(detm)

    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            Ux[i, j] = link(occ[i][j], occ[ip][j])
            Uy[i, j] = link(occ[i][j], occ[i][jp])

    F12 = zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            # plaquette
            F12[i, j] = log(Ux[i, j] * Uy[ip, j] / (Ux[i, jp] * Uy[i, j]))

    ch = F12.sum() / (2j * pi)
    return float(ch.real)

eig_vbm0 = hstack((Eig_gx[:,0],Eig_xy[:,0],Eig_yg[:,0],Eig_gm[:,0],Eig_mg[:,0]))
eig_vbm1 = hstack((Eig_gx[:,1],Eig_xy[:,1],Eig_yg[:,1],Eig_gm[:,1],Eig_mg[:,1]))
eig_vbm2 = hstack((Eig_gx[:,2],Eig_xy[:,2],Eig_yg[:,2],Eig_gm[:,2],Eig_mg[:,2]))
eig_vbm3 = hstack((Eig_gx[:,3],Eig_xy[:,3],Eig_yg[:,3],Eig_gm[:,3],Eig_mg[:,3]))
eig_cbm0 = hstack((Eig_gx[:,4],Eig_xy[:,4],Eig_yg[:,4],Eig_gm[:,4],Eig_mg[:,4]))
eig_cbm1 = hstack((Eig_gx[:,5],Eig_xy[:,5],Eig_yg[:,5],Eig_gm[:,5],Eig_mg[:,5]))
eig_cbm2 = hstack((Eig_gx[:,6],Eig_xy[:,6],Eig_yg[:,6],Eig_gm[:,6],Eig_mg[:,6]))
eig_cbm3 = hstack((Eig_gx[:,7],Eig_xy[:,7],Eig_yg[:,7],Eig_gm[:,7],Eig_mg[:,7]))



plt.figure(figsize=(6, 5))
plt.xlim(0, 1001)

plt.plot(eig_vbm0, color='red')
plt.plot(eig_vbm1, color='blue')
plt.plot(eig_vbm2, color='pink')
plt.plot(eig_vbm3, color='purple')
plt.plot(eig_cbm0, color='orange')
plt.plot(eig_cbm1, color='green')
plt.plot(eig_cbm2, color='cyan')
plt.plot(eig_cbm3, color='magenta')


plt.xticks([0, 500, 1000,1500,2000,2501], [r'$\Gamma$', 'X','Y', r'$\Gamma$','M', r'$\Gamma$'])
plt.ylabel('Energy')
plt.show()

# ---------------- Topology diagnostics (2D) ----------------
# For this 8-band model, half-filling is usually n_occ=4 (tune if your gap is elsewhere).
# - If TR is broken (e.g. J != 0), the relevant invariant is typically the Chern number.
# - If TR is preserved (e.g. J == 0 and SOC doesn't break it), you can compute Z2 via Wilson loop (not implemented here yet).
try:
    ch = chern_number_fukui(nk=41, n_occ=4)
    print(f"[Topology] Chern number (nk=41, n_occ=4) ≈ {ch:.6f}")
    if abs(ch) < 0.5:
        print("[Topology] Likely trivial (C≈0) for chosen filling, assuming the system is gapped.")
    else:
        print("[Topology] Likely topological (C≠0) for chosen filling, assuming the system is gapped.")
except Exception as e:
    print(f"[Topology] Chern number computation failed: {e}")
