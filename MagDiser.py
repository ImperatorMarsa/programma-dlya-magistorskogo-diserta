# -*- coding: utf-8 -*-
"""MagDiser.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r7cBJODBB_Lji0B75G-kdAc-xDEOa-GZ
"""

#%%
import numpy as np
from numba import jit
import math
import os.path
import pickle
import time
from random import random, gauss

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

# <+>!=<+>!=<+>!=___Fundamental'nie Poctoyannie___0<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
#@title Настройка
#@markdown Настройка переменных системы
#@markdown ---

CisloChastic = 293 #@param {type: "integer"}
KolvoIteraciy = 30 #@param {type: "integer"}

# Paskal*sekunda
Vyazkost = 2.15e-3 #@param {type: "number"}

# sekund
Time = 1e-10 #@param {type: "number"}

# 1.38e-23
PostoyanayaBolcmana = 1.38e-23 
Temperature = 273.16 #@param {type: "number"}
kT = Temperature * PostoyanayaBolcmana

U0 = 4e-7 * np.pi # Genri/метр
# Метров
Radiuse = 6.66e-9 #@param {type: "number"}

H_max = 7.3e3 #@param {type: "number"}

Plotnost = 5000 # килограмм/метр^3

Obyom = 4 / 3 * np.pi * Radiuse ** 3  # Метров^3

Massa = Obyom * Plotnost  # килограмм

Dlina_PAV = 2e-9 #@param {type: "number"}

NamagnicEdiniciMassi = 4.78e5 #@param {type: "number"}
MagMom = NamagnicEdiniciMassi * Obyom # Ампер*метр^2((namagnichenost' nasisheniya=4.78*10^5 Ампер/метр))

# Метров
Koncintraciya_obyomnaya = 0.10 #@param {type: "number"}
GraniciVselennoy = math.pow(Obyom * CisloChastic / Koncintraciya_obyomnaya, 1/3)

#@markdown ---
# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=

pickelPath = "//content/drive/My Drive/Проектные работы(Курсач; Диплм)/MagistarskiyDisert/ExperimenralResult/data_.pickle"
scince_data = {
    'Const' : {
        'CisloChastic' : CisloChastic,
        'Vyazkost' : Vyazkost,
        'Time' : Time,
        'Temperature' : Temperature,
        'Radiuse' : Radiuse,
        'Dlina_PAV' : Dlina_PAV,
        'Plotnost' : Plotnost,
        'H_max' : H_max,
        'NamagnicEdiniciMassi' : NamagnicEdiniciMassi,
        'Koncintraciya_obyomnaya' : Koncintraciya_obyomnaya,
        'GraniciVselennoy' : GraniciVselennoy
    },
    'Varibles' : {
        'KolvoIteraciy' : KolvoIteraciy,
        'N' : 0,
        'H' : [],
        'Chasichki' : [],
        'Result' : []
    }
}

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
#%%
# <|:|><|:|><|:|><|:|><|:|>__Osnovnaya_Proga__!<|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|>
@jit(fastmath = True, nopython = True, parallel = True)
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

@jit(fastmath = True, nopython = True, parallel = True)
def PrimoyMoment(mI, mJ, n, magN):
    rIJ = (mI[RadVek] - mJ[RadVek]) + n
    magR = math.sqrt(rIJ[0] ** 2 + rIJ[1] ** 2 + rIJ[2] ** 2)

    B_I = (
        U0
        / (4 * np.pi)
        * (Dot(mI[NaprUgl], rIJ) * 3 * rIJ / (magR) ** 5 - mI[NaprUgl] / (magR) ** 3)
    )
    mI[VekMomentov] += Cross(mJ[NaprUgl], B_I)

    return mI

@jit(fastmath = True, nopython = True, parallel = True)
def VneshPole(mI, N):
    B = np.array([1, 0, 0]) * H(N) * U0
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
        mI[VekSil] += A * 3 * U0 * M ** 2 / (4 * np.pi * a ** 4) * e * pom

    return mI


# https://www.desmos.com/calculator/bhjmf8p0pf
@jit(fastmath = True, nopython = True, parallel = True)
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
        C[VekMomentov]
        / (8.0 * np.pi * C[ParamCastic][R_Chastici] ** 3 * Vyazkost)
        * Time
        + pom
    )  # np.linalg.norm(DeltaAlfa) #
    buf = math.sqrt(DeltaAlfa[0] ** 2 + DeltaAlfa[1] ** 2 + DeltaAlfa[2] ** 2)
    C[NaprUgl] = RotatinVec(C[NaprUgl], DeltaAlfa, buf)
    C[VekVrash] = (C[NaprUgl] - C[VekVrash]) / Time

    C = PorvrkaGrani(C)
    C[VekSil] = np.zeros(3)
    C[VekMomentov] = np.zeros(3)

    return C

@jit(fastmath = True, nopython = True, parallel = True)
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

@jit(fastmath = True, nopython = True, parallel = True)
def StahostSmeshLineynoe(Radiuse):
    difuz = kT / (6.0 * np.pi * Radiuse * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time) ** 0.5 * gauss(0, 1))

@jit(fastmath = True, nopython = True, parallel = True)
def StahostSmeshUglovoe(Radiuse):
    difuz = kT / (8.0 * np.pi * Radiuse ** 3 * Vyazkost)

    return RandNormVec() * ((2 * difuz * Time) ** 0.5 * gauss(0, 1))


# <|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|><|:|>

@jit(fastmath = True, nopython = True, parallel = True)
def H(N):
    # a = N+1
    # Ампер/метр
    return H_max * math.cos(np.pi / 2500 * N)

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
@jit(fastmath = True, nopython = True, parallel = True)
def Cross(a, b):
    return np.array(
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

    return np.array([a1 / sq, a2 / sq, a3 / sq])

@jit(fastmath = True, nopython = True, parallel = True)
def Culculete(pom):
    return np.sqrt(np.sum(np.square(np.copy(pom.reshape(len(pom), 7 * 3)[:, NaprUgl * 3:NaprUgl * 3 + 3])), axis = 1)) * MagMom

def Koordi(pom):
    return np.copy(pom.reshape(len(pom), 7 * 3)[:, RadVek * 3:RadVek * 3 + 3])


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
    return np.array([p0, p1, p2])


def fu():
    print("\n     += Поехали =+")


# Funkciya generiruet massiv dinamichiskih peremennih chastic
def createrChastic(CisloChastic):
    pom = [[]]
    pom[-1].append(2 * GraniciVselennoy * np.random.rand(3) - GraniciVselennoy)
    pom[-1].append(RandNormVec() * MagMom)

    pom[-1].append([0, 0, 0])
    pom[-1].append([0, 0, 0])

    pom[-1].append([0, 0, 0])
    pom[-1].append([0, 0, 0])

    pom[-1].append([Radiuse, Massa, 0])
    for i in range(1, CisloChastic):
        print(
            "\rСоздано %d из %d частиц"
#             % (len(pom), CisloChastic),
            end="",
        )
        koord = 2 * GraniciVselennoy * np.random.rand(3) - GraniciVselennoy
        new_koord = True
        while new_koord:
            for x in pom:
                if np.absolute(np.sqrt(np.sum(np.square(x[0]))) - np.sqrt(np.sum(np.square(koord)))) > 2 * (Radiuse + Dlina_PAV):
                    new_koord = False
                else:
                    new_koord = True
                    koord = 2 * GraniciVselennoy * np.random.rand(3) - GraniciVselennoy
        
        pom.append([])            
        pom[-1].append(koord)
        pom[-1].append(RandNormVec() * MagMom)

        pom[-1].append([0, 0, 0])
        pom[-1].append([0, 0, 0])

        pom[-1].append([0, 0, 0])
        pom[-1].append([0, 0, 0])

        pom[-1].append([Radiuse, Massa, 0])

    print(np.max(Koordi(np.array(pom))))
    return np.array(pom)
#%%
# =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>=
new_experiment = True
if os.path.exists(pickelPath):
    f = open(pickelPath, "rb")
    buffer = pickle.load(f)
    f.close()
    if buffer['Const'] == scince_data['Const']:
        new_experiment = False
        for k, v in buffer['Varibles'].items():
            scince_data['Varibles'][k] = v

if new_experiment:
    print('Запускаем новый опыт')
    Chasichki = createrChastic(CisloChastic)
    scince_data['Varibles']['Chasichki'].append(Chasichki)
else:
    print('Продолжаем старый опыт')
    Chasichki = scince_data['Varibles']['Chasichki']
    KolvoIteraciy = scince_data['Varibles']['KolvoIteraciy']

start_time = time.time()
Iteraciy = KolvoIteraciy - scince_data['Varibles']['N']
if Iteraciy > 0:
    print('\n Начало опыта')
    scince_data['Varibles']['Chasichki'].append(Chasichki)
    scince_data['Varibles']['Result'].append(Culculete(Chasichki))
    scince_data['Varibles']['H'].append(H(0) * U0)
    timeInterput = -time.time()
    for N in range(1, Iteraciy):
        scince_data['Varibles']['N'] += 1
        N = scince_data['Varibles']['N']

        Chasichki = MathKernel(Chasichki, N)
        Chasichki = GeneralLoop(Chasichki)
        scince_data['Varibles']['Chasichki'].append(Chasichki)
        scince_data['Varibles']['Result'].append(Culculete(Chasichki))
        scince_data['Varibles']['H'].append(H(N) * U0)
        if time.time() - timeInterput > 600:
            print(
                "\rВыполнено %d из %d итераций \t\tМагнитное поле=%eH"
#                 % (N + 1, KolvoIteraciy, H(N)),
                end="",
            )
            f = open(pickelPath, 'wb+')
            pickle.dump(scince_data, f)
            f.close()
            timeInterput = time.time()

f = open(pickelPath, 'wb+')
pickle.dump(scince_data, f)
f.close()

print("\nВремя выполнения составило {}".format(time.time() - start_time))
# =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>=
#%%
print(GraniciVselennoy)
PredelSumm = 1

f = open(pickelPath, "rb")
buffer = pickle.load(f)
f.close()

koordi = buffer['Varibles']['Chasichki'][0]
koordi = Koordi(koordi)
koordi = koordi * GraniciVselennoy / np.max(koordi)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

x, y, z = np.copy(koordi[:, :1]), np.copy(koordi[:, 1:2]), np.copy(koordi[:, 2:3])
ax.scatter(x, y, z, marker = 'o')

plt.show()
fig.savefig("test_rasterization1.svg", dpi=350)

fig, ax = plt.subplots()

s = np.arange(0.0, len(buffer['Varibles']['Result']), 1)
ax.plot(s, buffer['Varibles']['Result'])

s = np.arange(0.0, len(buffer['Varibles']['H']), 1)
ax.plot(s, buffer['Varibles']['H'])

plt.ylim(np.min(buffer['Varibles']['H']), np.max(buffer['Varibles']['H']))
ax.grid()
fig.savefig("test_rasterization2.svg")
plt.show()

# 2366.5597999095917
import numpy as np
import cupy as cp

a = cp.linspace(-3, -1, 10)
a = a.reshape((10, 1))
b = cp.linspace(2, 5, 4)
# b = b.reshape((10, 3))
b * a