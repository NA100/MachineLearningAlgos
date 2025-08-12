import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)
y = np.sin(x)
z = np.cos(x)
print(x)
print(y)
print(z)

#sine wave
plt.plot(x,y)
plt.show()

#adding title
plt.plot(x,y)
plt.xlabel('angle')
plt.ylabel('sine value')
plt.title('sine wave')
plt.show()

#parabola
x = np.linspace(-10, 10, 20)
y = x**2
plt.plot(x,y)
plt.show()

#plots in red plus sign
plt.plot(x,y,'r+')
plt.show()

x = np.linspace(-5, 5, 50)
plt.plot(x,np.sin(x), 'g-')
plt.plot(x,np.cos(x), 'r--') #dotted lines
plt.show() #shows both

#Bar plot
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
lagnuages = ['Eng', 'Fre']
people = [100,50]
ax.bar(languages, people)
plt.show()

#Pie chart
fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
lagnuages = ['Eng', 'Fre']
people = [100,50]
ax.pie(people,labels=languages, autopct='1f%%')
plt.show()

#Scatter plot
x = np.linspace(0,10,30)
y = np.sin(x)
z = np.cos(x)
fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1])
ax.scatter(x,y,color='g')
ax.scatter(x,z,color='b')
plt.show()

#3d scatter plot