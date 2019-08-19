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

def MagMom(rad):
    """Функция возвращающая намагниченность частици заданного радиуса
    
    Arguments:
        rad {float} -- Радиус частицы
    """
    Obyom = 4 / 3 * xp.pi * rad**3
    return NamagnicEdiniciObyoma * Obyom

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

Obyom_ = 4 / 3 * xp.pi * (Radiuse + Dlina_PAV)**3  # Метров^3
Obyom = 4 / 3 * xp.pi * Radiuse**3  # Метров^3

Massa = Obyom * Plotnost  # килограмм

NamagnicEdiniciObyoma = 4.78e5 #@param {type: "number"}

# Метров
Koncintraciya_obyomnaya = 0.10 #@param {type: "number"}
GraniciVselennoy = math.pow(Obyom_ * CisloChastic / Koncintraciya_obyomnaya, 1/3)

#@markdown ---

n = xp.array(
    [
        GraniciVselennoy * 2,
        GraniciVselennoy * 2,
        GraniciVselennoy * 2,
    ]
)
infinity = xp.array(
    [
        10e10,
        10e10,
        10e10,
    ]
)
# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
#@jit(fastmath = True, nopython = True, parallel = True)
def H(N):
    """Функция возвращающая вентор магнитной напряженности Н [A/м]
    
    Arguments:
        N {[integer]} -- Номер итерации
    """
    return xp.array([
        Hx_amplitud * xp.cos(2 * xp.pi * Frequency * (N * delta_T) + Faza),
        Hy_amplitud,
        0,
    ], dtype = xp.float64)


#@jit(fastmath = True, parallel = True) # , nopython = True
def _VneshPole(N, moment):
    """Функция расчёта силы действующая на частицу со стороны внешнего поля
    
    Возвращает мвссив моментов сил
    
    Arguments:
        N {integer} -- Номер итерации
        moment {array[float64]} -- Массив векторов намагниченности частиц
    """
    B = H(N) * U0 # здесь стоит мину, потомучто я ленивая жопка и мне влом переписывать векторное произведение :З
    return Cross(moment, B)

VneshPole = np.vectorize(_VneshPole, otypes=[float], signature='(),(n)->(n)')


# https://www.desmos.com/calculator/ddxmffkqrj
#@jit(fastmath = True, nopython = True, parallel = True)
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
        return -distVek * dist * A * 3e-7 * M**2 / (Rq2**4) * xp.exp(-B * (dist / Rq2 - 1))
    else:
        return xp.array([0, 0, 0,], dtype = xp.float64)

SteerOttalk = np.vectorize(_SteerOttalk, otypes=[float], signature='(m),(l),(k)->(k)')


def createrChastic(koordi, namag, skor, uglSkor):
    """Функция заполняющая матрицы для системы частиц
    
    Arguments:
        koordi {array} -- Матрица векторов радиус координат
        namag {array} -- Матрица векторов намагниченности
        skor {array} -- Матрица векторов скоростей
        uglSkor {array} -- Матрица векторов угловых скоростей
    """
    for i in range(len(koordi)):
        # Задаю магнитный момент
        rend_k = xp.random.rand(3) - 0.5
        namag[i] = rend_k / xp.linalg.norm(rend_k) * MagMom(Radiuse)

        # Задаю наиболее вероятную скорость для частиц
        rend_k = xp.random.rand(3) - 0.5
        skor[i] = rend_k / xp.linalg.norm(rend_k) * xp.sqrt(2 * kT / Massa)

        # Задаю наиболее вероятну угловую скорость частиц
        rend_k = xp.random.rand(3) - 0.5
        uglSkor[i] = rend_k / xp.linalg.norm(rend_k) * xp.sqrt(8 * kT / (2 / 5 *Massa * Radiuse**2))

        # Устанавливаю координаты, так чтобы они небыли слишком близко
        rend_k = xp.random.rand(3) * GraniciVselennoy
        isTouch = True
        _isTouch = True
        j = 0
        while isTouch:
            koordi[j]
            dist = xp.sqrt(xp.sum(xp.square(xp.copy(rend_k - koordi[j]))))
            if dist < ((Radiuse + Dlina_PAV) * 2):
                _isTouch = True
            else:
                _isTouch = False

            j += 1
            if j > i:
                j = 0
                if _isTouch:
                    isTouch = True
                    rend_k = xp.random.rand(3) * GraniciVselennoy
                else:
                    isTouch = False
                    koordi[i] = rend_k

    return koordi, namag, skor, uglSkor


#@jit(fastmath = True, nopython = True, parallel = True)
def Culculete(pom):
    """Функция рассчитывающая среднее значение модуля для матрицы векторов
    
    Arguments:
        pom {array} -- Матрицы векторов
    """
    return xp.sum(xp.sqrt(xp.sum(xp.square(xp.copy(pom)), axis = 1)))


#@jit(fastmath = True, nopython = True, parallel = True)
def _Sila(n, matrix_K, matrix_U, uglVek, radVek):
    """Функция расчёта силы взаимодействия двух маг диполей
    
    Arguments:
        n {array} -- Вектор смещение. Используется для переноса системы частиц в соответствующем направлении
        matrix_K {array} --Матрица координат частиц
        matrix_U {array} -- Матрица векторов намагниченности частиц
        uglVek {array} -- Вектор намагниченности данной частицы
        radVek {array} -- Координаты данной частицы
    """
    distVek = xp.copy((matrix_K + n) - radVek)
    dist = xp.sqrt(xp.sum(xp.square(distVek)))
    distVek /= dist
    return (
        3 * U0 / (4 * xp.pi * dist**4)
        * (
            (uglVek @ distVek) * matrix_U
            + (matrix_U @ distVek) * uglVek
            + (uglVek @ matrix_U) * distVek
            - 5 * (uglVek @ distVek) * (matrix_U @ distVek) * distVek
        )
    )

Sila = np.vectorize(_Sila, otypes=[float], signature='(a),(s),(d),(m),(k)->(k)')


#@jit(fastmath = True, parallel = True) # , nopython = True
def _Moment(n, matrix_K, matrix_U, uglVek, radVek):
    """Функция расчёта силы взаимодействия двух маг диполей
    
    Arguments:
        n {array} -- Вектор смещение. Используется для переноса системы частиц в соответствующем направлении
        matrix_K {array} --Матрица координат частиц
        matrix_U {array} -- Матрица векторов намагниченности частиц
        uglVek {array} -- Вектор намагниченности данной частицы
        radVek {array} -- Координаты данной частицы
    """
    distVek = xp.copy((matrix_K + n) - radVek)
    dist = xp.sqrt(xp.sum(xp.square(distVek)))
    distVek /= dist
    B_mI = (
        U0 / (4 * xp.pi)
         * (3 * distVek * (uglVek @ distVek) - uglVek) / (dist**3)
    )
    return Cross(matrix_U, B_mI)

Moment = np.vectorize(_Moment, otypes=[float], signature='(a),(s),(d),(m),(k)->(k)')


#@jit(fastmath = True, nopython = True, parallel = True)
def _Cross(A, B):
    """Функция расчёта векторно произведения бвух трёхмерных векторов
    
    Arguments:
        A {array} -- Вектор над которым будут производить век. умножение
        B {array} -- Вектор который будет производить век. умножение
    """
    return xp.array([
        A[1] * B[2] - A[2] * B[1],
        A[2] * B[0] - A[0] * B[2],
        A[0] * B[1] - A[1] * B[0],
    ], dtype = xp.float64)

Cross = np.vectorize(_Cross, otypes=[float], signature='(m),(k)->(k)')


#@jit(fastmath = True, nopython = True, parallel = True)
def _PorvrkaGrani(koord):
    """Функция проверки "не вышлали координата за пределы ячейки".
    Она вернёт частицу во внетрь ячейки, даже если её отфигачела на километры.
    
    Arguments:
        koord {array} -- Вектор координат частиц
    """
    return koord - GraniciVselennoy * int(koord / GraniciVselennoy)

PorvrkaGrani = np.vectorize(_PorvrkaGrani, otypes=[float], signature='()->()')


#@jit(fastmath = True, parallel = True) # , nopython = True
def MathKernel(MatrixKoordinat, MatrixNamagnicennosti, MatrixSili, MatrixMoenta, N, CisloProekciy = 4):
    MatrixMoenta += VneshPole(N, MatrixNamagnicennosti)
    lenses = len(MatrixKoordinat)
    for i in range(lenses):
        onePartickle_r = xp.copy(MatrixKoordinat[i])
        MatrixKoordinat[i] = xp.copy(infinity)
        onePartickle_u = xp.copy(MatrixNamagnicennosti[i])
        MatrixSili += SteerOttalk(MatrixKoordinat, onePartickle_u, onePartickle_r)
        for X in range(-CisloProekciy, CisloProekciy + 1):
            buffer1 = int(math.sqrt(CisloProekciy**2 - X**2))
            for Y in range(-buffer1, buffer1 + 1):
                buffer2 = int(math.sqrt(CisloProekciy**2 - X**2 - Y**2))
                for Z in range(-buffer2, buffer2 + 1):
                    n[0] = X * GraniciVselennoy * 2
                    n[1] = Y * GraniciVselennoy * 2
                    n[2] = Z * GraniciVselennoy * 2
                    MatrixMoenta += Moment(n, MatrixKoordinat, MatrixNamagnicennosti, onePartickle_u, onePartickle_r)
                    MatrixSili += Sila(n, MatrixKoordinat, MatrixNamagnicennosti, onePartickle_u, onePartickle_r)

        MatrixKoordinat[i] = xp.copy(onePartickle_r)
    return MatrixKoordinat, MatrixNamagnicennosti, MatrixSili, MatrixMoenta


# # https://www.desmos.com/calculator/bhjmf8p0pf
## @jit(fastmath = True, nopython = True, parallel = True)
# def Kinematika(C):
#     pom = StahostSmeshLineynoe(C[ParamCastic][R_Chastici])
#     C[VecSkor] = C[RadVek]
#     C[RadVek] = (
#         C[RadVek]
#         + C[VekSil] / (6.0 * xp.pi * C[ParamCastic][R_Chastici] * Vyazkost) * delta_T
#         + pom
#     )
#     C[VecSkor] = (C[RadVek] - C[VecSkor]) / delta_T

#     pom = StahostSmeshUglovoe(C[ParamCastic][R_Chastici])
#     C[VekVrash] = C[NaprUgl]
#     DeltaAlfa = (
#         C[VekMomentov]
#         / (8.0 * xp.pi * C[ParamCastic][R_Chastici]**3 * Vyazkost)
#         * delta_T
#         + pom
#     )  # xp.linalg.norm(DeltaAlfa) #
#     buf = math.sqrt(DeltaAlfa[0]**2 + DeltaAlfa[1]**2 + DeltaAlfa[2]**2)
#     C[NaprUgl] = RotatinVec(C[NaprUgl], DeltaAlfa, buf)
#     C[VekVrash] = (C[NaprUgl] - C[VekVrash]) / delta_T

#     C = PorvrkaGrani(C)
#     C[VekSil] = xp.zeros(3)
#     C[VekMomentov] = xp.zeros(3)

#     return C


## @jit(fastmath = True, nopython = True, parallel = True)
# def StahostSmeshLineynoe(Radiuse):
#     difuz = kT / (6.0 * xp.pi * Radiuse * Vyazkost)

#     return RandNormVec() * ((2 * difuz * delta_T)**0.5 * gauss(0, 1))


## @jit(fastmath = True, nopython = True, parallel = True)
# def StahostSmeshUglovoe(Radiuse):
#     difuz = kT / (8.0 * xp.pi * Radiuse**3 * Vyazkost)

#     return RandNormVec() * ((2 * difuz * delta_T)**0.5 * gauss(0, 1))


## @jit(fastmath = True, nopython = True, parallel = True)
# def GeneralLoop(mass):
#     for x in range(len(mass)):
#         mass[x] = Kinematika(mass[x])

#     return mass


## @jit(fastmath = True, nopython = True, parallel = True)
# def RotatinVec(vec, axis, ugol):
#     nK = math.sqrt(axis[0]**2 + axis[2]**2 + axis[1]**2)
#     axis[0], axis[1], axis[2] = axis[0] / nK, axis[1] / nK, axis[2] / nK
#     a = math.cos(ugol / 2.0)
#     sinKoef = math.sin(ugol / 2.0)
#     b, c, d = -axis[0] * sinKoef, -axis[1] * sinKoef, -axis[2] * sinKoef
#     aa, bb, cc, dd = a * a, b * b, c * c, d * d
#     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#     q1, q2, q3 = aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)
#     q4, q5, q6 = 2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)
#     q7, q8, q9 = 2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc
#     p0 = q1 * vec[0] + q2 * vec[1] + q3 * vec[2]
#     p1 = q4 * vec[0] + q5 * vec[1] + q6 * vec[2]
#     p2 = q7 * vec[0] + q8 * vec[1] + q9 * vec[2]
#     return xp.array([p0, p1, p2])
