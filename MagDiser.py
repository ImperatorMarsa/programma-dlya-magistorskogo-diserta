# -*- coding: utf-8 -*-
"""MagDiser.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r7cBJODBB_Lji0B75G-kdAc-xDEOa-GZ
"""

#%%
import math
import time
import pickle
import os.path
import numpy as np
import MathLib as m
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

Obyom_ = 4 / 3 * xp.pi * (Radiuse + Dlina_PAV) ** 3  # Метров^3
Obyom = 4 / 3 * xp.pi * Radiuse ** 3  # Метров^3

Massa = Obyom * Plotnost  # килограмм

NamagnicEdiniciObyoma = 4.78e5 #@param {type: "number"}

# Метров
Koncintraciya_obyomnaya = 0.10 #@param {type: "number"}
GraniciVselennoy = math.pow(Obyom_ * CisloChastic / Koncintraciya_obyomnaya, 1/3)

m.kT = kT
m.U0 = U0
m.Faza = Faza
m.Obyom = Obyom
m.Massa = Massa
m.Radiuse = Radiuse
m.delta_T = delta_T
m.Vyazkost = Vyazkost
m.Plotnost = Plotnost
m.Frequency = Frequency
m.Dlina_PAV = Dlina_PAV
m.Temperature = Temperature
m.Hx_amplitud = Hx_amplitud
m.Hy_amplitud = Hy_amplitud
m.CisloChastic = CisloChastic
m.KolvoIteraciy = KolvoIteraciy
m.GraniciVselennoy = GraniciVselennoy
m.PostoyanayaBolcmana = PostoyanayaBolcmana
m.NamagnicEdiniciObyoma = NamagnicEdiniciObyoma
m.Koncintraciya_obyomnaya = Koncintraciya_obyomnaya
#@markdown ---

MatrixKoordinat = xp.zeros((CisloChastic, 3))
MatrixNamagnicennosti = xp.zeros((CisloChastic, 3))
MatrixSkorosti = xp.zeros((CisloChastic, 3))
MatrixUglSkorosti = xp.zeros((CisloChastic, 3))
MatrixSili = xp.zeros((CisloChastic, 3))
MatrixMoenta = xp.zeros((CisloChastic, 3))

# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
pickelPath = "C:\\Users\\sitnikov\\Documents\\Python Scripts\\data_.pickle"
scince_data = {
    'Const' : {
        'Faza' : m.Faza,
        'delta_T' : m.delta_T,
        'Radiuse' : m.Radiuse,
        'Plotnost' : m.Plotnost,
        'Vyazkost' : m.Vyazkost,
        'Dlina_PAV' : m.Dlina_PAV,
        'Frequency' : m.Frequency,
        'Hx_amplitud' : m.Hx_amplitud,
        'Hy_amplitud' : m.Hy_amplitud,
        'Temperature' : m.Temperature,
        'CisloChastic' : m.CisloChastic,
        'NamagnicEdiniciObyoma' : m.NamagnicEdiniciObyoma,
        'Koncintraciya_obyomnaya' : m.Koncintraciya_obyomnaya,
    },
    'Varibles' : {
        'N' : 0,
        'H' : [],
        'MatrixSili' : MatrixSili,
        'MatrixMoenta' : MatrixMoenta,
        'KolvoIteraciy' : KolvoIteraciy,
        'MatrixSkorosti' : MatrixSkorosti,
        'MatrixKoordinat' : MatrixKoordinat,
        'MatrixUglSkorosti' : MatrixUglSkorosti,
        'MatrixNamagnicennosti' : MatrixNamagnicennosti,
    }
}

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
    MatrixKoordinat, MatrixNamagnicennosti, MatrixSkorosti, MatrixUglSkorosti = m.createrChastic(
        MatrixKoordinat,
        MatrixNamagnicennosti,
        MatrixSkorosti,
        MatrixUglSkorosti
    )
    scince_data['Varibles']['MatrixKoordinat'] = MatrixKoordinat
    scince_data['Varibles']['MatrixNamagnicennosti'] = MatrixNamagnicennosti
else:
    print('Продолжаем старый опыт')
    KolvoIteraciy = scince_data['Varibles']['KolvoIteraciy']
    MatrixKoordinat = scince_data['Varibles']['MatrixKoordinat']
    MatrixNamagnicennosti = scince_data['Varibles']['MatrixNamagnicennosti']

start_time = time.time()
Iteraciy = KolvoIteraciy - scince_data['Varibles']['N']
if Iteraciy > 0:
    print('\n Начало опыта')
    scince_data['Varibles']['MatrixKoordinat'] = MatrixKoordinat
    scince_data['Varibles']['MatrixNamagnicennosti'] = MatrixNamagnicennosti
    scince_data['Varibles']['MatrixSkorosti'] = MatrixSkorosti
    scince_data['Varibles']['MatrixUglSkorosti'] = MatrixUglSkorosti
    scince_data['Varibles']['Result'].append(m.Culculete(MatrixNamagnicennosti))
    scince_data['Varibles']['H'].append(m.H(0) * U0)
    TABLE_DATA = (
        ('Название параметра', 'Значеие', 'Единицы измерения'),
        ('Число частиц', CisloChastic, 'шт.'),
        ('Вязкость', Vyazkost, 'Па * с'),
        ('Шаг времени', delta_T, 'с'),
        ('Температура', Temperature, 'К'),
        ('Радиус', Radiuse, 'м'),
        ('Длина молекул ПАВ', Dlina_PAV, 'м'),
        ('Плотность', Plotnost, 'кг / м^3'),
        ('Максимум значения Н, по оси Х', Hx_amplitud, 'А / м'),
        ('Максимум значения Н, по оси Y', Hy_amplitud, 'А / м'),
        ('Намаг. ед. массы', NamagnicEdiniciObyoma, 'А / (м * кг)'),
        ('Объёмная концентрация', Koncintraciya_obyomnaya, ''),
        ('Размеры ячейки', GraniciVselennoy, 'м'),
        ('Количество итераций', KolvoIteraciy, ''),
    )
    title = 'Параметры эксперимента'
    table_instance = DoubleTable(TABLE_DATA, title)
    print(table_instance.table, '\n')

    timeInterput = -time.time()
    for N in range(1, Iteraciy):
        scince_data['Varibles']['N'] += 1
        N = scince_data['Varibles']['N']

        MatrixKoordinat, MatrixNamagnicennosti, MatrixSkorosti, MatrixUglSkorosti, MatrixSili, MatrixMoenta = m.MathKernel(
            MatrixKoordinat,
            MatrixNamagnicennosti,
            MatrixSkorosti,
            MatrixUglSkorosti,
            MatrixSili,
            MatrixMoenta,
            N
        )
        MatrixKoordinat, MatrixNamagnicennosti, MatrixSkorosti, MatrixUglSkorosti, MatrixSili, MatrixMoenta = m.GeneralLoop(
            MatrixKoordinat,
            MatrixNamagnicennosti,
            MatrixSkorosti,
            MatrixUglSkorosti,
            MatrixSili,
            MatrixMoenta,
            N
        )
        scince_data['Varibles']['MatrixKoordinat'] = MatrixKoordinat
        scince_data['Varibles']['MatrixNamagnicennosti'] = MatrixNamagnicennosti
        scince_data['Varibles']['MatrixSkorosti'] = MatrixSkorosti
        scince_data['Varibles']['MatrixUglSkorosti'] = MatrixUglSkorosti
        scince_data['Varibles']['Result'].append(m.Culculete(MatrixNamagnicennosti))
        scince_data['Varibles']['H'].append(m.H(N) * U0)
        if time.time() - timeInterput > 600:
            print(
                "\rВыполнено %d из %d итераций \t\tМагнитное поле=%eH"
                % (N + 1, KolvoIteraciy, m.H(N)),
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
