import matplotlib.pyplot as plt

fp = open('lab2_500k.txt', "r")
mean = []
x = []
for line in iter(fp):
    content = line.split()
    for i in range(len(content)):
        if(content[i] == 'mean'):
            mean.append(float(content[i+2]))

count = 0
for  i in range(len(mean)):
    x.append(count)
    count += 1000
fp.close()
plt.plot(x, mean)
plt.xlabel('episode')
plt.ylabel('score')
plt.show()

plt.cla()
fp = open('lab2_4tuple.txt', "r")
mean = []
x = []
for line in iter(fp):
    content = line.split()
    for i in range(len(content)):
        if(content[i] == 'mean'):
            mean.append(float(content[i+2]))

count = 0
for  i in range(len(mean)):
    x.append(count)
    count += 1000
fp.close()
plt.plot(x, mean)
plt.xlabel('episode')
plt.ylabel('score')
plt.show()
