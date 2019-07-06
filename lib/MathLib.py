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
delta_T = 1e-10 #@param {type: "number"}

# 1.38e-23
PostoyanayaBolcmana = 1.38e-23 
Temperature = 273.16 #@param {type: "number"}
kT = Temperature * PostoyanayaBolcmana

U0 = 4e-7 * xp.pi # Genri/метр
# Метров
Radiuse = 6.66e-9 #@param {type: "number"}
Dlina_PAV = 2e-9 #@param {type: "number"}

Hx_amplitud = 7.3e3 #@param {type: "number"}
Frequency = 300 #@param {type: "number"}
Faza = 0 #@param {type: "number"}
Hy_amplitud = 7.3e3 #@param {type: "number"}

Plotnost = 5000 # килограмм/метр^3

Obyom = 4 / 3 * xp.pi * Radiuse ** 3  # Метров^3

Massa = Obyom * Plotnost  # килограмм


NamagnicEdiniciMassi = 4.78e5 #@param {type: "number"}
MagMom = NamagnicEdiniciMassi * Obyom # Ампер*метр^2((namagnichenost' nasisheniya=4.78*10^5 Ампер/метр))

# Метров
Koncintraciya_obyomnaya = 0.10 #@param {type: "number"}
GraniciVselennoy = math.pow(Obyom * CisloChastic / Koncintraciya_obyomnaya, 1/3)

#@markdown ---
# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
@jit(fastmath = True, nopython = True, parallel = True)
def H(N):
    """Функция возврашающая вентор магнитной напряженности Н [A/м]
    
    Arguments:
        N {[integer]} -- Номер итерации
    """
    return xp.array([
        Hx_amplitud * xp.cos(2 * xp.pi * Frequency * (N * delta_T) + Faza),
        Hy_amplitud,
        0,
    ], dtype = xp.float64)


@jit(fastmath = True, nopython = True, parallel = True)
def _VneshPole(N, moment):
    """Функция расчёта силы действующая на частицу со стороны внешнего поля
    
    Возвращает мвссив моментов сил
    
    Arguments:
        N {integer} -- Номер итерации
        moment {array[float64]} -- Массив векторов намагниченности частиц
    """
    B = -H(N) * U0 # здесь стоит мину, потомучто я ленивая жопка и мне влом переписывать векторное произведение :З
    return xp.array([
        B[1] * moment[2] - B[2] * moment[1],
        B[2] * moment[0] - B[0] * moment[2],
        B[0] * moment[1] - B[1] * moment[0],
    ], dtype = xp.float64)

VneshPole = np.vectorize(_VneshPole, otypes=[float], signature='(),(n)->(n)')


# https://www.desmos.com/calculator/ddxmffkqrj
@jit(fastmath = True, nopython = True, parallel = True)
def _SteerOttalk(matrix, uglVek, radVek):
    """Функция расчёта стерического отталкивания частиц МЖ

    Возвращает массив сил отталкивания
    
    Arguments:
        matrix {array[float64]} -- Массив координат частиц в котором собственные координаты унесены на бесконечность
        uglVek {array[float64]} -- Вектор намагниченности
        radVek {array[float64]} -- Координаты частицы с которой производится взаимодействие
    """
    # Kakieto koefficienti #
    A = 31.3
    B = 73.0
    # ######################
    Rq2 = 2 * (Radiuse + Dlina_PAV)

    distVek = xp.copy(matrix - radVek)
    dist = xp.sqrt(xp.sum(xp.square(distVek)))
    distVek /= dist
    M = xp.sqrt(xp.sum(xp.square(xp.copy(uglVek))))
    if dist <= Rq2:
        return -distVek * dist * A * 3e-7 * M ** 2 / (Rq2 ** 4) * xp.exp(-B * (dist / Rq2 - 1))
    else:
        return xp.array([0, 0, 0,], dtype = xp.float64)

SteerOttalk = np.vectorize(_SteerOttalk, otypes=[float], signature='(m),(l),(k)->(k)')



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


# https://www.desmos.com/calculator/bhjmf8p0pf
@jit(fastmath = True, nopython = True, parallel = True)
def Kinematika(C):
    pom = StahostSmeshLineynoe(C[ParamCastic][R_Chastici])
    C[VecSkor] = C[RadVek]
    C[RadVek] = (
        C[RadVek]
        + C[VekSil] / (6.0 * xp.pi * C[ParamCastic][R_Chastici] * Vyazkost) * delta_T
        + pom
    )
    C[VecSkor] = (C[RadVek] - C[VecSkor]) / delta_T

    pom = StahostSmeshUglovoe(C[ParamCastic][R_Chastici])
    C[VekVrash] = C[NaprUgl]
    DeltaAlfa = (
        C[VekMomentov]
        / (8.0 * xp.pi * C[ParamCastic][R_Chastici] ** 3 * Vyazkost)
        * delta_T
        + pom
    )  # xp.linalg.norm(DeltaAlfa) #
    buf = math.sqrt(DeltaAlfa[0] ** 2 + DeltaAlfa[1] ** 2 + DeltaAlfa[2] ** 2)
    C[NaprUgl] = RotatinVec(C[NaprUgl], DeltaAlfa, buf)
    C[VekVrash] = (C[NaprUgl] - C[VekVrash]) / delta_T

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

    return RandNormVec() * ((2 * difuz * delta_T) ** 0.5 * gauss(0, 1))


@jit(fastmath = True, nopython = True, parallel = True)
def StahostSmeshUglovoe(Radiuse):
    difuz = kT / (8.0 * xp.pi * Radiuse ** 3 * Vyazkost)

    return RandNormVec() * ((2 * difuz * delta_T) ** 0.5 * gauss(0, 1))


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
