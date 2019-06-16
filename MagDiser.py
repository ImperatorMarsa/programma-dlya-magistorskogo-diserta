#%%
import numpy as np
import math
import pickle
import time
from random import random, gauss

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Struktura massiva chastic:
RadVek = 0  # ::: RadiuseVecrtor chastici v prostranstve
NaprUgl = 1  # ::: Napravlyayushie kosinusi

VecSkor = 2  # ::: Vektor skorosti
VekVrash = 3  # ::: Vektor uglovjy skorosti

VekSil = 4  # ::: Vektor Sili
VEkMomentov = 5  # ::: Vector momentaSil

ParamCastic = 6  # ::: Parametri chastic: Radius, Massa, ...
R_Chastici = 0  # ::: Radiuse chastici
M_Chastici = 1  # ::: Massa odnoy chastici
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# <+>!=<+>!=<+>!=___Fundamental'nie Poctoyannie___0<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
CisloChastic = 100
KolvoIteraciy = 30

Vyazkost = 2.15e-3  # Paskal*sekunda
Time = 1e-10  # sekund
kT = 273.16 * 1.38e-23 # 1.38e-23
U0 = 4e-7 * np.pi # Genri/метр
Radiuse = 6.66e-9 # Метров
Plotnost = 5000 # килограмм/метр^3
Obyom = 4 / 3 * np.pi * Radiuse ** 3  # Метров^3
Massa = Obyom * Plotnost  # килограмм
MagMom = 4.78e5 * Obyom # Ампер*метр^2((namagnichenost' nasisheniya=4.78*10^5 Ампер/метр))
GraniciVselennoy = 7.3e-8 # Метров
Alpha = 5 / GraniciVselennoy
# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=

#%%
# <|:|><|:|><|:|><|:|><|:|>__Osnovnaya_Proga__!<|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|>
def PrimoySila(mI, mJ, n, magN):
    rIJ = (mI[RadVek] - mJ[RadVek]) + n
    magR = math.sqrt(rIJ[0] ** 2 + rIJ[1] ** 2 + rIJ[2] ** 2)
    mI[VekSil] += (
        3
        * U0
        / (4 * np.pi * magR ** 5)
        * (
            Dot(mI[NaprUgl], rIJ) * mJ[NaprUgl]
            + Dot(mJ[NaprUgl], rIJ) * mI[NaprUgl]
            + Dot(mI[NaprUgl], mJ[NaprUgl]) * rIJ
            - 5 * Dot(mI[NaprUgl], rIJ) * Dot(mJ[NaprUgl], rIJ) * rIJ / magR ** 2
        )
    )

    return mI


def PrimoyMoment(mI, mJ, n, magN):
    rIJ = (mI[RadVek] - mJ[RadVek]) + n
    magR = math.sqrt(rIJ[0] ** 2 + rIJ[1] ** 2 + rIJ[2] ** 2)

    B_I = (
        U0
        / (4 * np.pi)
        * (Dot(mI[NaprUgl], rIJ) * 3 * rIJ / (magR) ** 5 - mI[NaprUgl] / (magR) ** 3)
    )
    mI[VEkMomentov] += Cross(mJ[NaprUgl], B_I)

    return mI


def VneshPole(mI, N):
    B = np.array([1, 0, 0]) * H(N) * U0
    mI[VEkMomentov] += Cross(mI[NaprUgl], B)
    return mI


# https://www.desmos.com/calculator/ddxmffkqrj
def SteerOttalk(mI, mJ):
    pom = mI[RadVek] - mJ[RadVek]
    dist = math.sqrt(pom[0] ** 2 + pom[1] ** 2 + pom[2] ** 2)
    # F_ster=vector(0, 0, 0)

    # Kakieto koefficienti #
    A = 31.3
    B = 73.0
    # ######################

    M = math.sqrt(
        mI[NaprUgl][0] ** 2 + mI[NaprUgl][1] ** 2 + mI[NaprUgl][2] ** 2
    )  # Magnitniy moment chastici

    # Dlina volosni v metrah
    q = 2e-9

    # Диаметр частиц с учётом длины молекул ПАВ
    a = mJ[ParamCastic][R_Chastici] + mI[ParamCastic][R_Chastici] + 2 * q

    if dist < (mJ[ParamCastic][R_Chastici] + mI[ParamCastic][R_Chastici] + 2 * q):
        e = math.exp(-B * (dist / a - 1))
        mI[VekSil] += A * 3 * U0 * M ** 2 / (4 * np.pi * a ** 4) * e * pom

    return mI


# https://www.desmos.com/calculator/bhjmf8p0pf
def Kinematika(C):
    pom = StahostSmeshLineynoe(C[ParamCastic][R_Chastici])
    C[VecSkor] = C[RadVek]
    C[RadVek] = (
        C[RadVek]
        + C[VekSil] / (6.0 * np.pi * C[ParamCastic][R_Chastici] * Vyazkost) * Time
        + pom
    )
    C[VecSkor] = (C[RadVek] - C[VecSkor]) / Time

    pom = StahostSmeshUglovoe(C[ParamCastic][R_Chastici])
    C[VekVrash] = C[NaprUgl]
    DeltaAlfa = (
        C[VEkMomentov]
        / (8.0 * np.pi * C[ParamCastic][R_Chastici] ** 3 * Vyazkost)
        * Time
        + pom
    )  # np.linalg.norm(DeltaAlfa) #
    buf = math.sqrt(DeltaAlfa[0] ** 2 + DeltaAlfa[1] ** 2 + DeltaAlfa[2] ** 2)
    C[NaprUgl] = RotatinVec(C[NaprUgl], DeltaAlfa, buf)
    C[VekVrash] = (C[NaprUgl] - C[VekVrash]) / Time

    C = PorvrkaGrani(C)
    C[VekSil] = np.zeros(3)
    C[VEkMomentov] = np.zeros(3)

    return C


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


def StahostSmeshLineynoe(Radiuse):
    difuz = kT / (6.0 * np.pi * Radiuse * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time) ** 0.5 * gauss(0, 1))


def StahostSmeshUglovoe(Radiuse):
    difuz = kT / (8.0 * np.pi * Radiuse ** 3 * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time) ** 0.5 * gauss(0, 1))


# <|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|>


def H(N):
    # a = N+1
    # Ампер/метр
    return 73e5 * math.cos(np.pi / 100 * N)


def GeneralLoop(mass):
    for x in range(len(mass)):
        mass[x] = Kinematika(mass[x])

    return mass


def MathKernel(mass, N):
    for i in range(len(mass)):
        mass[i] = VneshPole(mass[i], N)
        for j in range(len(mass)):
            mass[i] = SteerOttalk(mass[i], mass[j])

            PredelSumm = 15
            for X in range(-PredelSumm, PredelSumm + 1):
                pom1 = int(math.sqrt(PredelSumm ** 2 - X ** 2))
                for Y in range(pom1, pom1 + 1):
                    pom2 = int(math.sqrt(PredelSumm ** 2 - X ** 2 - Y ** 2))
                    for Z in range(pom2, pom2 + 1):
                        n = np.array(
                            [
                                X * GraniciVselennoy * 2,
                                Y * GraniciVselennoy * 2,
                                Z * GraniciVselennoy * 2,
                            ]
                        )
                        magN = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)

                        if j != i:
                            mass[i] = PrimoySila(mass[i], mass[j], n, magN)
                            mass[i] = PrimoyMoment(mass[i], mass[j], n, magN)

    return mass


# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
def Cross(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def Dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def RandNormVec():
    a1 = 1 - 2 * random()
    a2 = 1 - 2 * random()
    a3 = 1 - 2 * random()
    sq = math.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)

    return np.array([a1 / sq, a2 / sq, a3 / sq])


# Oshipka skritaya
def RotatinVec(vec, axis, ugol):
    nK = math.sqrt(axis[0] ** 2 + axis[2] ** 2 + axis[1] ** 2)
    axis[0], axis[1], axis[2] = axis[0] / nK, axis[1] / nK, axis[2] / nK
    a = math.cos(ugol / 2.0)
    sinKoef = math.sin(ugol / 2.0)
    b, c, d = -axis[0] * sinKoef, -axis[1] * sinKoef, -axis[2] * sinKoef
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    q1, q2, q3 = aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)
    q4, q5, q6 = 2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)
    q7, q8, q9 = 2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc
    p0 = q1 * vec[0] + q2 * vec[1] + q3 * vec[2]
    p1 = q4 * vec[0] + q5 * vec[1] + q6 * vec[2]
    p2 = q7 * vec[0] + q8 * vec[1] + q9 * vec[2]

    return np.array([p0, p1, p2])


def fu():
    print("\n     += Поехали =+")


# Funkciya generiruet massiv dinamichiskih peremennih chastic
def createrChastic():
    pom = np.ones((CisloChastic, 7, 3))
    for i in range(CisloChastic):
        pom[i][0] = (
            np.ones(3) * GraniciVselennoy
            - np.array([random(), random(), random()]) * 2 * GraniciVselennoy
        )
        pom[i][1] = RandNormVec() * MagMom

        pom[i][2] = np.zeros(3)
        pom[i][3] = np.zeros(3)

        pom[i][4] = np.zeros(3)
        pom[i][5] = np.zeros(3)

        pom[i][6] = np.array([Radiuse, Massa, 0])

    return pom

#%%
# =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>=
Chasichki = createrChastic()
# Evaluciya = np.array([])
# Evaluciya = np.append(Evaluciya, Chasichki)
# print(Chasichki[0])

start_time = time.time()
# s(Chasichki)
for N in range(KolvoIteraciy):
    print(
        "\rВыполнено %d из %d итераций \t\tМагнитное поле=%eH"
        % (N + 1, KolvoIteraciy, H(N)),
        end="",
    )

    Chasichki = MathKernel(Chasichki, N)
    # Evaluciya = np.append(Evaluciya, Chasichki)
    Chasichki = GeneralLoop(Chasichki)

# print("\n", Chasichki[0])
print("\nВремя выполнения составило {}".format(time.time() - start_time))

# with open("C:/SciData/data_Premoe.pickle", "wb") as f:
#     pickle.dump(
#         Evaluciya.reshape(
#             (int(len(Evaluciya) / CisloChastic / 6 / 3), CisloChastic, 6, 3)
#         ),
#         f,
#     )
# f.close()
# =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>=
