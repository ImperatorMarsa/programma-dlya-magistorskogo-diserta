# @profile
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Struktura massiva chastic:
#     0 ::: RadiuseVecrtor chastici v prostranstve
#     1 ::: Napravlyayushie kosinusi
#
#     2 ::: Vektor skorosti
#     3 ::: Vektor uglovjy skorosti
#
#     4 ::: Vektor Sili
#     5 ::: Vector momentaSil
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numba as nb
import numpy as np
import math
import pickle
from random import random, gauss

np.warnings.filterwarnings('ignore')
# 808080___Fundamental'nie Poctoyannie___0808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080
CisloChastic = 20
KolvoIteraciy = 100

# Paskal*sekunda
Vyazkost = 2.15e-3

# sekund
Time = 1e-10
# 1.38e-23
kT = 273.16 * 1.38e-23
# Genri/metr
U0 = 4e-7 * np.pi

# metrov
Radiuse = 6.66e-9
# kilogramm/metr^3
Plotnost = 5000
Obyom = 4 / 3 * np.pi * Radiuse**3  # metrov^3
Massa = Obyom * Plotnost  # kilogramm
# Amper*metr^2((namagnichenost' nasisheniya=4.78*10^5 Amper/metr))
MagMom = 4.78e5 * Obyom

# metrov
GraniciVselennoy = 7.3e-8
Alpha = 5/GraniciVselennoy
# @!@!@!@!@!__Osnovnaya_Proga__!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!


@nb.jit(nopython=True, parallel=True)
def C(a):
    mag = (a[0]**2 + a[1]**2 + a[2]**2)**.5
    return (3 * math.erfc(Alpha*mag) + (2 * Alpha * mag / math.sqrt(np.pi)) * (3 + 2 * (Alpha * mag)**2)*math.exp(- (Alpha * mag)**2)) / mag**5


@nb.jit(nopython=True, parallel=True)
def B(a):
    mag = (a[0]**2 + a[1]**2 + a[2]**2)**.5
    return (3*math.erfc(Alpha*mag)+(2*Alpha*mag/math.sqrt(np.pi))*math.exp(-(Alpha * mag)**2)) / mag**3


@nb.jit(nopython=True, parallel=True)
def D(a):
    mag = (a[0]**2 + a[1]**2 + a[2]**2)**.5
    return (15 * math.erfc(Alpha * mag) + (2 * Alpha * mag / math.sqrt(np.pi)) * (15 + 10 * (Alpha * mag)**2 + 4 * (Alpha * mag)**4) * math.exp(- (Alpha * mag)**2)) / mag**7


@nb.jit(nopython=True, parallel=True)
def PrimoySila(mI, mJ, n, magN):
    rIJ = (mI[0] - mJ[0]) + n
    mI[4] += (rIJ * Dot(mI[1], mJ[1])
              + mJ[1] * Dot(mI[1], rIJ)
              + mI[1] * Dot(mJ[1], rIJ)) * C(rIJ) - (Dot(mI[1], rIJ) * Dot(mJ[1], rIJ))*D(rIJ)*rIJ

    return mI


@nb.jit(nopython=True, parallel=True)
def ObratniySila(mI, mJ, k, magK):
    rIJ = mI[0] - mJ[0]
    mI[4] += -(4 * np.pi / GraniciVselennoy**3) * (k / magK**2 * math.exp(-(np.pi * magK / Alpha / GraniciVselennoy)**2)) * Dot(mI[1], k) * Dot(mJ[1], k) * math.sin(Dot(k, rIJ))

    return mI


@nb.jit(nopython=True, parallel=True)
def PrimoyMoment(mI, mJ, n, magN):
    rIJ = (mI[0] - mJ[0]) + n
    mI[5] += Cross(mI[1], mJ[1]) * B(rIJ) - Cross(mI[1], rIJ * (Dot(mJ[1], rIJ) * C(rIJ)))

    return mI


@nb.jit(nopython=True, parallel=True)
def ObratniyMoment(mI, mJ, k, magK):
    rIJ = mI[0] - mJ[0]
    mI[5] += -(4 * np.pi / GraniciVselennoy**3) * (1 / magK**2 * math.exp(-(np.pi * magK / Alpha / GraniciVselennoy)**2)) * Cross(mI[1], k) * Dot(mJ[1], k) * math.cos(Dot(k, rIJ))

    return mI


@nb.jit(nopython=True, parallel=True)
def VneshPole(mI):
    B = np.array([1, 0, 0]) * H() * U0
    mI[5] += Cross(mI[1], B)
    return mI


@nb.jit(nopython=True, parallel=True)
# https://www.desmos.com/calculator/ddxmffkqrj
def SteerOttalk(mI, mJ):
    pom = mI[0] - mJ[0]
    dist = math.sqrt(
        pom[0]**2 + pom[1]**2 + pom[2]**2)
    # F_ster=vector(0, 0, 0)

    # Kakieto koefficienti #
    A = 31.3
    B = 73.0
    # ######################

    M = math.sqrt(mI[1][0]**2 + mI[1][1]**2 + mI[1][2]**2)  # Magnitniy moment chastici
    # Dlina volosni v metrah
    q = 2e-9
    # Diametr chastici s volosney
    a = 2.0 * (Radiuse + q)
    if dist < 2.0*Radiuse:
        e = math.exp(-B * (dist / a - 1))
        mI[4] += A * 3 * U0 * M**2 / (4 * np.pi * a**4) * e * pom

    return mI


@nb.jit(nopython=True, parallel=True)
# https://www.desmos.com/calculator/bhjmf8p0pf
def Kinematika(mass):
    pom = StahostSmeshLineynoe()
    mass[2] = mass[0]
    mass[0] = mass[0]+mass[4] / (6.0 * np.pi * Radiuse * Vyazkost) * Time + pom
    mass[2] = (mass[0]-mass[2])/Time

    pom = StahostSmeshUglovoe()
    mass[3] = mass[1]
    DeltaAlfa = mass[5] / (8.0 * np.pi * Radiuse**3 * Vyazkost) * Time + pom  # np.linalg.norm(DeltaAlfa) #
    buf = math.sqrt(DeltaAlfa[0]**2 + DeltaAlfa[1]**2 + DeltaAlfa[2]**2)
    mass[1] = RotatinVec(mass[1], DeltaAlfa, buf)
    mass[3] = (mass[1] - mass[3]) / Time

    mass = PorvrkaGrani(mass)
    mass[4] = np.zeros(3)
    mass[5] = np.zeros(3)

    return mass


@nb.jit(nopython=True, parallel=True)
def PorvrkaGrani(mass):
    if mass[0][0] > GraniciVselennoy:
        mass[0] = np.array([mass[0][0] - 2 * GraniciVselennoy, mass[0][1], mass[0][2]])
    elif mass[0][0] < -GraniciVselennoy:
        mass[0] = np.array([mass[0][0] + 2 * GraniciVselennoy, mass[0][1], mass[0][2]])

    if mass[0][1] > GraniciVselennoy:
        mass[0] = np.array([mass[0][0], mass[0][1] - 2 * GraniciVselennoy, mass[0][2]])
    elif mass[0][1] < -GraniciVselennoy:
        mass[0] = np.array([mass[0][0], mass[0][1] + 2 * GraniciVselennoy, mass[0][2]])

    if mass[0][2] > GraniciVselennoy:
        mass[0] = np.array([mass[0][0], mass[0][1], mass[0][2] - 2 * GraniciVselennoy])
    elif mass[0][2] < -GraniciVselennoy:
        mass[0] = np.array([mass[0][0], mass[0][1], mass[0][2] + 2 * GraniciVselennoy])

    return mass


@nb.jit(nopython=True, parallel=True)
def StahostSmeshLineynoe():
    difuz = kT / (6.0 * np.pi * Radiuse * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time)**.5 * gauss(0, 1))


@nb.jit(nopython=True, parallel=True)
def StahostSmeshUglovoe():
    difuz = kT / (8.0 * np.pi * Radiuse**3 * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time)**.5 * gauss(0, 1))

# @!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!


@nb.jit(nopython=True, parallel=True)
def H():
    global N
    # a = N+1
    # Amper/metr
    return 73e3


@nb.jit(nopython=True, parallel=True)
def GeneralLoop(mass):
    for x in range(len(mass)):
        mass[x] = Kinematika(mass[x])

    return mass


@nb.jit(nopython=True, parallel=True)
def MathKernel(mass):
    for i in range(len(mass)):
        mass[i] = VneshPole(mass[i])
        for j in range(len(mass)):
            mass[i] = SteerOttalk(mass[i], mass[j])

            mass[i][5] += -(4 * np.pi / 3 / GraniciVselennoy**3) * Cross(mass[i][1], mass[j][1])

            PredelSumm = 4
            for X in range(-PredelSumm, PredelSumm + 1):
                pom1 = int(math.sqrt(PredelSumm**2 - X**2))
                for Y in range(pom1, pom1 + 1):
                    pom2 = int(math.sqrt(PredelSumm**2 - X**2 - Y**2))
                    for Z in range(pom2, pom2 + 1):
                        n = np.array([X * GraniciVselennoy * 2, Y * GraniciVselennoy * 2, Z * GraniciVselennoy * 2])
                        magN = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)

                        if j != i:
                            mass[i] = PrimoySila(mass[i], mass[j], n, magN)
                            mass[i] = PrimoyMoment(mass[i], mass[j], n, magN)

            PredelSumm = 10
            for X in range(-PredelSumm, PredelSumm + 1):
                pom1 = int(math.sqrt(PredelSumm**2 - X**2))
                for Y in range(pom1, pom1 + 1):
                    pom2 = int(math.sqrt(PredelSumm**2 - X**2 - Y**2))
                    for Z in range(pom2, pom2 + 1):
                        k = np.array([X * GraniciVselennoy * 2, Y * GraniciVselennoy * 2, Z * GraniciVselennoy * 2])
                        magK = math.sqrt(k[0]**2 + k[1]**2 + k[2]**2)

                        if magK:
                            mass[i] = ObratniySila(mass[i], mass[j], k, magK)
                            mass[i] = ObratniyMoment(mass[i], mass[j], k, magK)

    return mass
# 8080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080


@nb.jit(nopython=True, parallel=True)
def Cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@nb.jit(nopython=True, parallel=True)
def Dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@nb.jit(nopython=True, parallel=True)
def RandNormVec():
    a1 = 1-2*random()
    a2 = 1-2*random()
    a3 = 1-2*random()
    sq = math.sqrt(a1**2 + a2**2 + a3**2)

    return np.array([a1 / sq, a2 / sq, a3 / sq])


@nb.jit(nopython=True, parallel=True)  # Oshipka skritaya
def RotatinVec(vec, axis, ugol):
    nK = math.sqrt(axis[0]**2 + axis[2]**2 + axis[1]**2)
    axis[0], axis[1], axis[2] = axis[0] / nK, axis[1] / nK, axis[2]/nK
    a = math.cos(ugol / 2.0)
    sinKoef = math.sin(ugol / 2.0)
    b, c, d = -axis[0] * sinKoef, - axis[1] * sinKoef, - axis[2] * sinKoef
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    q1, q2, q3 = aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)
    q4, q5, q6 = 2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)
    q7, q8, q9 = 2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc
    p0 = q1 * vec[0] + q2 * vec[1] + q3 * vec[2]
    p1 = q4 * vec[0] + q5 * vec[1] + q6 * vec[2]
    p2 = q7 * vec[0] + q8 * vec[1] + q9 * vec[2]

    return np.array([p0, p1, p2])


@nb.jit(nopython=True, parallel=True)
def fu():
    print("\n     -=< Poehali >=-")


# Funkciya generiruet massiv dinamichiskih peremennih chastic
@nb.jit(nopython=True, parallel=True)
def createrChastic():
    pom = np.ones((CisloChastic, 6, 3))
    for i in range(CisloChastic):
        pom[i][0] = np.ones(3) * GraniciVselennoy - np.array([random(), random(), random()]) * 2 * GraniciVselennoy
        pom[i][1] = RandNormVec() * MagMom
        pom[i][2] = np.zeros(3)
        pom[i][3] = np.zeros(3)
        pom[i][4] = np.zeros(3)
        pom[i][5] = np.zeros(3)

    return pom


fu()


# @profile
# def s(Chasichki):
#     Evaluciya = np.array([])
#     Evaluciya = np.append(Evaluciya, Chasichki)
#     for N in range(KolvoIteraciy):
#         # if True:
#         print('\r', N, end='')

#         Chasichki = MathKernel(Chasichki)
#         Chasichki = GeneralLoop(Chasichki)
#         Evaluciya = np.append(Evaluciya, Chasichki)


# $%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%
Chasichki = createrChastic()
Evaluciya = np.array([])
Evaluciya = np.append(Evaluciya, Chasichki)
print(Chasichki[0])

# s(Chasichki)
for N in range(KolvoIteraciy):
    # if True:
    print('\rProshlo', N, end='')

    Chasichki = MathKernel(Chasichki)
    Evaluciya = np.append(Evaluciya, Chasichki)
    Chasichki = GeneralLoop(Chasichki)

print('\n', Chasichki[0])

with open('C:/SciData/data_Evald.pickle', 'wb') as f:
    pickle.dump(Evaluciya.reshape((int(len(Evaluciya)/CisloChastic/6/3), CisloChastic, 6, 3)), f)
f.close()
# $%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%
