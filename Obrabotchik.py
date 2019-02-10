import pickle

with open('C:/SciData/data_Premoe.pickle', 'rb') as f:
    mas=pickle.load(f)

a=[]
for x in mas:
    a.append([0, 0, 0])
    for y in x:
        a[-1][0]+=y[1][0]
        a[-1][1]+=y[1][1]
        a[-1][2]+=y[1][2]

q=open("Pony.txt", "w")
for x in a: q.writelines("%E\t%E\t%E\n"%(x[0], x[1], x[2]))
q.close()