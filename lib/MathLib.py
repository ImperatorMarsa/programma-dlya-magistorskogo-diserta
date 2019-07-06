import math
import time
import pickle
import os.path
import numpy as np
from numba import jit
from random import random, gauss
from terminaltables import DoubleTable
# import cupy as cp

xp = np
# <+>!=<+>!=<+>!=___Fundamental'nie Poctoyannie___0<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
#@title Настройка
#@markdown Настройка переменных системы
#@markdown ---

CisloChastic = 234 #@param {type: "integer"}
KolvoIteraciy = 3 #@param {type: "integer"}

# Paskal*sekunda
Vyazkost = 2.15e-3 #@param {type: "number"}

# sekund
Time = 1e-10 #@param {type: "number"}

# 1.38e-23
PostoyanayaBolcmana = 1.38e-23 
Temperature = 273.16 #@param {type: "number"}
kT = Temperature * PostoyanayaBolcmana

U0 = 4e-7 * xp.pi # Genri/метр
# Метров
Radiuse = 6.66e-9 #@param {type: "number"}

H_max = 7.3e3 #@param {type: "number"}

Plotnost = 5000 # килограмм/метр^3

Obyom = 4 / 3 * xp.pi * Radiuse ** 3  # Метров^3

Massa = Obyom * Plotnost  # килограмм

Dlina_PAV = 2e-9 #@param {type: "number"}

NamagnicEdiniciMassi = 4.78e5 #@param {type: "number"}
MagMom = NamagnicEdiniciMassi * Obyom # Ампер*метр^2((namagnichenost' nasisheniya=4.78*10^5 Ампер/метр))

# Метров
Koncintraciya_obyomnaya = 0.10 #@param {type: "number"}
GraniciVselennoy = math.pow(Obyom * CisloChastic / Koncintraciya_obyomnaya, 1/3)

#@markdown ---
# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Struktura massiva chastic:
RadVek = 0  # ::: RadiuseVecrtor chastici v prostranstve
NaprUgl = 1  # ::: Napravlyayushie kosinusi

VecSkor = 2  # ::: Vektor skorosti
VekVrash = 3  # ::: Vektor uglovjy skorosti

VekSil = 4  # ::: Vektor Sili
VekMomentov = 5  # ::: Vector momentaSil

ParamCastic = 6  # ::: Parametri chastic: Radius, Massa, ...
R_Chastici = 0  # ::: Radiuse chastici
M_Chastici = 1  # ::: Massa odnoy chastici
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@jit(fastmath = True, nopython = True, parallel = True)
def PrimoySila(mI, mJ, n, magN):
    rIJ = (mI[RadVek] - mJ[RadVek]) + n
    magR = math.sqrt(rIJ[0] ** 2 + rIJ[1] ** 2 + rIJ[2] ** 2)
    mI[VekSil] += (
        3
        * U0
        / (4 * xp.pi * magR ** 5)
        * (
            Dot(mI[NaprUgl], rIJ) * mJ[NaprUgl]
            + Dot(mJ[NaprUgl], rIJ) * mI[NaprUgl]
            + Dot(mI[NaprUgl], mJ[NaprUgl]) * rIJ
            - 5 * Dot(mI[NaprUgl], rIJ) * Dot(mJ[NaprUgl], rIJ) * rIJ / magR ** 2
        )
    )

    return mI


@jit(fastmath = True, nopython = True, parallel = True)
def PrimoyMoment(mI, mJ, n, magN):
    rIJ = (mI[RadVek] - mJ[RadVek]) + n
    magR = math.sqrt(rIJ[0] ** 2 + rIJ[1] ** 2 + rIJ[2] ** 2)

    B_I = (
        U0
        / (4 * xp.pi)
        * (Dot(mI[NaprUgl], rIJ) * 3 * rIJ / (magR) ** 5 - mI[NaprUgl] / (magR) ** 3)
    )
    mI[VekMomentov] += Cross(mJ[NaprUgl], B_I)

    return mI


@jit(fastmath = True, nopython = True, parallel = True)
def VneshPole(mI, N):
    B = xp.array([1, 0, 0]) * H(N) * U0
    mI[VekMomentov] += Cross(mI[NaprUgl], B)
    return mI


# https://www.desmos.com/calculator/ddxmffkqrj
@jit(fastmath = True, nopython = True, parallel = True)
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
    q = Dlina_PAV

    # Диаметр частиц с учётом длины молекул ПАВ
    a = mJ[ParamCastic][R_Chastici] + mI[ParamCastic][R_Chastici] + 2 * q

    if dist < (mJ[ParamCastic][R_Chastici] + mI[ParamCastic][R_Chastici] + 2 * q):
        e = math.exp(-B * (dist / a - 1))
        mI[VekSil] += A * 3 * U0 * M ** 2 / (4 * xp.pi * a ** 4) * e * pom

    return mI


# https://www.desmos.com/calculator/bhjmf8p0pf
@jit(fastmath = True, nopython = True, parallel = True)
def Kinematika(C):
    pom = StahostSmeshLineynoe(C[ParamCastic][R_Chastici])
    C[VecSkor] = C[RadVek]
    C[RadVek] = (
        C[RadVek]
        + C[VekSil] / (6.0 * xp.pi * C[ParamCastic][R_Chastici] * Vyazkost) * Time
        + pom
    )
    C[VecSkor] = (C[RadVek] - C[VecSkor]) / Time

    pom = StahostSmeshUglovoe(C[ParamCastic][R_Chastici])
    C[VekVrash] = C[NaprUgl]
    DeltaAlfa = (
        C[VekMomentov]
        / (8.0 * xp.pi * C[ParamCastic][R_Chastici] ** 3 * Vyazkost)
        * Time
        + pom
    )  # xp.linalg.norm(DeltaAlfa) #
    buf = math.sqrt(DeltaAlfa[0] ** 2 + DeltaAlfa[1] ** 2 + DeltaAlfa[2] ** 2)
    C[NaprUgl] = RotatinVec(C[NaprUgl], DeltaAlfa, buf)
    C[VekVrash] = (C[NaprUgl] - C[VekVrash]) / Time

    C = PorvrkaGrani(C)
    C[VekSil] = xp.zeros(3)
    C[VekMomentov] = xp.zeros(3)

    return C


@jit(fastmath = True, nopython = True, parallel = True)
def PorvrkaGrani(mass):
    if mass[0][0] > GraniciVselennoy:
        mass[0] = xp.array([mass[0][0] - 2 * GraniciVselennoy, mass[0][1], mass[0][2]])
    elif mass[0][0] < -GraniciVselennoy:
        mass[0] = xp.array([mass[0][0] + 2 * GraniciVselennoy, mass[0][1], mass[0][2]])

    if mass[0][1] > GraniciVselennoy:
        mass[0] = xp.array([mass[0][0], mass[0][1] - 2 * GraniciVselennoy, mass[0][2]])
    elif mass[0][1] < -GraniciVselennoy:
        mass[0] = xp.array([mass[0][0], mass[0][1] + 2 * GraniciVselennoy, mass[0][2]])

    if mass[0][2] > GraniciVselennoy:
        mass[0] = xp.array([mass[0][0], mass[0][1], mass[0][2] - 2 * GraniciVselennoy])
    elif mass[0][2] < -GraniciVselennoy:
        mass[0] = xp.array([mass[0][0], mass[0][1], mass[0][2] + 2 * GraniciVselennoy])

    return mass


@jit(fastmath = True, nopython = True, parallel = True)
def StahostSmeshLineynoe(Radiuse):
    difuz = kT / (6.0 * xp.pi * Radiuse * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time) ** 0.5 * gauss(0, 1))


@jit(fastmath = True, nopython = True, parallel = True)
def StahostSmeshUglovoe(Radiuse):
    difuz = kT / (8.0 * xp.pi * Radiuse ** 3 * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time) ** 0.5 * gauss(0, 1))


@jit(fastmath = True, nopython = True, parallel = True)
def H(N):
    # a = N+1
    # Ампер/метр
    return H_max * xp.cos(xp.pi / 2500 * N)


@jit(fastmath = True, nopython = True, parallel = True)
def GeneralLoop(mass):
    for x in range(len(mass)):
        mass[x] = Kinematika(mass[x])

    return mass


@jit(fastmath = True, nopython = True, parallel = True)
def MathKernel(mass, N):
    for i in range(len(mass)):
        mass[i] = VneshPole(mass[i], N)
        for j in range(len(mass)):
            mass[i] = SteerOttalk(mass[i], mass[j])

            PredelSumm = 3
            for X in range(-PredelSumm, PredelSumm + 1):
                pom1 = int(math.sqrt(PredelSumm ** 2 - X ** 2))
                for Y in range(-pom1, pom1 + 1):
                    pom2 = int(math.sqrt(PredelSumm ** 2 - X ** 2 - Y ** 2))
                    for Z in range(-pom2, pom2 + 1):
                        n = xp.array(
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


@jit(fastmath = True, nopython = True, parallel = True)
def Cross(a, b):
    return xp.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@jit(fastmath = True, nopython = True, parallel = True)
def Dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit(fastmath = True, nopython = True, parallel = True)
def RandNormVec():
    a1 = 1 - 2 * random()
    a2 = 1 - 2 * random()
    a3 = 1 - 2 * random()
    sq = math.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)

    return xp.array([a1 / sq, a2 / sq, a3 / sq])


@jit(fastmath = True, nopython = True, parallel = True)
def Culculete(pom):
    return xp.sqrt(xp.sum(xp.square(xp.copy(pom.reshape(len(pom), 7 * 3)[:, NaprUgl * 3:NaprUgl * 3 + 3])), axis = 1)) * MagMom


@jit(fastmath = True, nopython = True, parallel = True)
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
    return xp.array([p0, p1, p2])
