import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

alpha=0.001
G=27*10**9
#fun1 and fun2 are the upper and lower boundaries of the cross-secton
def evaluate_fun1(x):
	temp=x/math.sqrt(3)
	return temp

def evaluate_fun2(x):
	temp=-x/math.sqrt(3)
	return temp

def relu(x):
	b=x>0
	return b*x

x_start=0
x_end=1
x_steps=20
x_change=((x_end-x_start)/x_steps)
x_list=np.linspace(x_start,x_end,x_steps+1)

y_list=[]
y_start=0
y_end=0

for i in x_list:
	y_start=min(y_start,evaluate_fun2(i))
	y_end=max(y_end,evaluate_fun1(i))

y_steps=20
y_change=((y_end-y_start)/y_steps)

y_list=np.linspace(y_start,y_end,y_steps+1)

points_interior=[]

for i in x_list:
	for j in y_list:
		if((evaluate_fun1(i)>j)&(evaluate_fun2(i)<j)):
			points_interior.append([i,j])

def f1(x,y):
	f=(y-evaluate_fun1(x))*(y-evaluate_fun2(x))*(x-x_start)*(x-x_end)
	return f

def f2(x,y):
	f=(y-evaluate_fun1(x))*np.sqrt(abs(y-evaluate_fun2(x)))*(x-x_start)*(x-x_end)
	return f

def f3(x,y):
	f=np.sqrt(abs(y-evaluate_fun1(x)))*np.sqrt(abs(y-evaluate_fun2(x)))*((x-x_start)*(x_end-x))
	return f

def eval_grad(f,x,y):
	h=0.00001

	grad_x=(f(x+h,y)-f(x,y))/h
	grad_y=(f(x,y+h)-f(x,y))/h
	return np.array([grad_x,grad_y])

#initialized weights of f1 and f2 in prandlt stress function
c1=6437709
c2=7069016
c3=62301

def integrate(f):
	sum_f=0
	for i in points_interior:
		x,y=i
		sum_f+=f(x,y)
	return sum_f*x_change*y_change

def eval_pe(c1,c2,c3):
	h=0.001
	pe=-c1*integrate(f1)-c2*integrate(f2)-c3*integrate(f3)

	for i in points_interior:
		x,y=i
		a=eval_grad(f1,x,y)
		b=eval_grad(f2,x,y)
		c=eval_grad(f3,x,y)
		grad=c1*a+c2*b+c3*c
		grad_sq=np.square(grad)
		sum_all=np.sum(grad_sq)
		pe+=(sum_all*x_change*y_change)/(4*G*alpha)

	return pe

learning_rate=1000000
n_epochs=500
h=0.001

momentum=0.9

g1=0.0
g2=0.0
g3=0.0

for epoch in range(n_epochs):
	g1=momentum*g1+(eval_pe(c1+h,c2,c3)-eval_pe(c1,c2,c3))/h
	g2=momentum*g1+(eval_pe(c1,c2+h,c3)-eval_pe(c1,c2,c3))/h
	g3=momentum*g1+(eval_pe(c1,c2,c3+h)-eval_pe(c1,c2,c3))/h
	c1-=learning_rate*g1
	c2-=learning_rate*g2
	c3-=learning_rate*g3
	print(eval_pe(c1,c2,c3),epoch)

print(c1,c2,c3)

def evaluate_phi(x_list,y_list):
	r=len(x_list)
	c=len(y_list)
	z=np.zeros((r,c))
	a=0
	b=0
	for x in x_list:
		a=0
		for y in y_list:
			z[a,b]=c1*f1(x,y)+c2*f2(x,y)+c3*f3(x,y)
			a+=1
		b+=1
	return z

z=evaluate_phi(x_list,y_list)

fig,ax=plt.subplots(1,1)
cp=ax.contourf(x_list,y_list,z,cmap="viridis")
fig.colorbar(cp)
#marking cross-section boundary in red
plt.axvline(x=x_end-0.01,color="red")
plt.axvline(x=x_start+0.01,color="red")
plt.show()

def phi(x,y):
	return c1*f1(x,y)+c2*f2(x,y)+c3*f3(x,y)

T=2*integrate(phi)

Torsional_stiffness=T/alpha

print(T,Torsional_stiffness)