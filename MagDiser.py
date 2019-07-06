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
from numba import jit
from random import random, gauss
from terminaltables import DoubleTable

# import cupy as cp

import sys
sys.path.insert(0, './lib')
import MathLib as m

xp = np

# <+>!=<+>!=<+>!=___Fundamental'nie Poctoyannie___0<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=
#@title Настройка
#@markdown Настройка переменных системы
#@markdown ---

m.CisloChastic = 234 #@param {type: "integer"}
m.KolvoIteraciy = 3 #@param {type: "integer"}

# Paskal*sekunda
m.Vyazkost = 2.15e-3 #@param {type: "number"}

# sekund
m.Time = 1e-10 #@param {type: "number"}

# 1.38e-23
m.PostoyanayaBolcmana = 1.38e-23 
m.Temperature = 273.16 #@param {type: "number"}
m.kT = Temperature * PostoyanayaBolcmana

m.U0 = 4e-7 * xp.pi # Genri/метр
# Метров
m.Radiuse = 6.66e-9 #@param {type: "number"}

m.H_max = 7.3e3 #@param {type: "number"}

m.Plotnost = 5000 # килограмм/метр^3

m.Obyom = 4 / 3 * xp.pi * Radiuse ** 3  # Метров^3

m.Massa = Obyom * Plotnost  # килограмм

m.Dlina_PAV = 2e-9 #@param {type: "number"}

m.NamagnicEdiniciMassi = 4.78e5 #@param {type: "number"}
m.MagMom = NamagnicEdiniciMassi * Obyom # Ампер*метр^2((namagnichenost' nasisheniya=4.78*10^5 Ампер/метр))

# Метров
m.Koncintraciya_obyomnaya = 0.10 #@param {type: "number"}
m.GraniciVselennoy = math.pow(Obyom * CisloChastic / Koncintraciya_obyomnaya, 1/3)

#@markdown ---
# <+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=<+>!=

pickelPath = "C:\\Users\\sitnikov\\Documents\\Python Scripts\\data_.pickle"
scince_data = {
    'Const' : {
        'CisloChastic' : m.CisloChastic,
        'Vyazkost' : m.Vyazkost,
        'Time' : m.Time,
        'Temperature' : m.Temperature,
        'Radiuse' : m.Radiuse,
        'Dlina_PAV' : m.Dlina_PAV,
        'Plotnost' : m.Plotnost,
        'H_max' : m.H_max,
        'NamagnicEdiniciMassi' : m.NamagnicEdiniciMassi,
        'Koncintraciya_obyomnaya' : m.Koncintraciya_obyomnaya,
        'GraniciVselennoy' : m.GraniciVselennoy
    },
    'Varibles' : {
        'KolvoIteraciy' : m.KolvoIteraciy,
        'N' : 0,
        'H' : [],
        'Chasichki' : [],
        'Result' : []
    }
}

#%%
# =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>=
# new_experiment = True
# if os.path.exists(pickelPath):
#     f = open(pickelPath, "rb")
#     buffer = pickle.load(f)
#     f.close()
#     if buffer['Const'] == scince_data['Const']:
#         new_experiment = False
#         for k, v in buffer['Varibles'].items():
#             scince_data['Varibles'][k] = v

# if new_experiment:
#     print('Запускаем новый опыт')
#     Chasichki = createrChastic(CisloChastic)
#     scince_data['Varibles']['Chasichki'].append(Chasichki)
# else:
#     print('Продолжаем старый опыт')
#     Chasichki = scince_data['Varibles']['Chasichki']
#     KolvoIteraciy = scince_data['Varibles']['KolvoIteraciy']

# start_time = time.time()
# Iteraciy = KolvoIteraciy - scince_data['Varibles']['N']
# if Iteraciy > 0:
#     print('\n Начало опыта')
#     scince_data['Varibles']['Chasichki'].append(Chasichki)
#     scince_data['Varibles']['Result'].append(Culculete(Chasichki))
#     scince_data['Varibles']['H'].append(H(0) * U0)
#     TABLE_DATA = (
#         ('Название параметра', 'Значеие', 'Единицы измерения'),
#         ('Число частиц', CisloChastic, 'шт.'),
#         ('Вязкость', Vyazkost, 'Па * с'),
#         ('Шаг времени', Time, 'с'),
#         ('Температура', Temperature, 'К'),
#         ('Радиус', Radiuse, 'м'),
#         ('Длина молекул ПАВ', Dlina_PAV, 'м'),
#         ('Плотность', Plotnost, 'кг / м^3'),
#         ('Максимум значения Н', H_max, 'А / м'),
#         ('Намаг. ед. массы', NamagnicEdiniciMassi, 'А / (м * кг)'),
#         ('Объёмная концентрация', Koncintraciya_obyomnaya, ''),
#         ('Размеры ячейки', GraniciVselennoy, 'м'),
#         ('Количество итераций', KolvoIteraciy, ''),
#     )
#     title = 'Параметры эксперимента'
#     table_instance = DoubleTable(TABLE_DATA, title)
#     print(table_instance.table, '\n')

#     timeInterput = -time.time()
#     for N in range(1, Iteraciy):
#         scince_data['Varibles']['N'] += 1
#         N = scince_data['Varibles']['N']

#         Chasichki = MathKernel(Chasichki, N)
#         Chasichki = GeneralLoop(Chasichki)
#         scince_data['Varibles']['Chasichki'].append(Chasichki)
#         scince_data['Varibles']['Result'].append(Culculete(Chasichki))
#         scince_data['Varibles']['H'].append(H(N) * U0)
#         if time.time() - timeInterput > 600:
#             print(
#                 "\rВыполнено %d из %d итераций \t\tМагнитное поле=%eH"
#                 % (N + 1, KolvoIteraciy, H(N)),
#                 end="",
#             )
#             f = open(pickelPath, 'wb+')
#             pickle.dump(scince_data, f)
#             f.close()
#             timeInterput = time.time()

# f = open(pickelPath, 'wb+')
# pickle.dump(scince_data, f)
# f.close()

# print("\nВремя выполнения составило {}".format(time.time() - start_time))
# =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>= =<< >>=
