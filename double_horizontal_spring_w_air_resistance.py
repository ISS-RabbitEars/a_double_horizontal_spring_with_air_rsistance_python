import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(xvo, ti, p):
	x1, v1, x2, v2 = xvo
	m1, k1, xeq1, m2, k2, xeq2, rho, cd, ar = p

	print(ti)

	return [v1, a1.subs({M1:m1, K1:k1, K2:k2, XEQ1:xeq1, XEQ2:xeq2, X1:x1, X2:x2, RHO:rho, CD:cd, Ar:ar, X1Dot:v1}),\
		v2, a2.subs({M2:m2, K2:k2, XEQ2:xeq2, X1:x1, X2:x2, RHO:rho, CD:cd, Ar:ar, X2Dot:v2})]



M1, K1, XEQ1, M2, K2, XEQ2, t = sp.symbols('M1 K1 XEQ1 M2 K2 XEQ2 t')
RHO, CD, Ar = sp.symbols('RHO CD Ar')
X1, X2 = dynamicsymbols('X1 X2')

X1Dot = X1.diff(t, 1)
X2Dot = X2.diff(t, 1)

T = M1 * X1Dot**2 + M2 * X2Dot**2
V = K1 * (X1 - XEQ1)**2 + K2 * (X2 - X1 - XEQ2)**2
T *= sp.Rational(1, 2)
V *= sp.Rational(1, 2)

L = T - V

dLdX1 = L.diff(X1, 1)
dLdX1Dot = L.diff(X1Dot, 1)
ddtdLdX1Dot = dLdX1Dot.diff(t, 1)

dLdX2 = L.diff(X2, 1)
dLdX2Dot = L.diff(X2Dot, 1)
ddtdLdX2Dot = dLdX2Dot.diff(t, 1)

Fc = sp.Rational(1, 2) * RHO * CD * Ar
F1 = Fc * sp.sign(X1Dot) * X1Dot**2
F2 = Fc * sp.sign(X2Dot) * X2Dot**2

dL1 = ddtdLdX1Dot - dLdX1 + F1
dL2 = ddtdLdX2Dot - dLdX2 + F2

aa1 = sp.solve(dL1, X1.diff(t, 2))
aa2 = sp.solve(dL2, X2.diff(t, 2))

a1 = aa1[0]
a2 = aa2[0]

#----------------------------------

m1 = 1
k1 = 1 
xo1 = 3
vo1 = 0
xeq1 = 2.25
m2 = 1
k2 = 1
xo2 = 6 
vo2 = 0
xeq2 = 2.25 
rho = 1.225
cd = 0.47
rad = 0.25
ar = np.pi * rad**2

p = m1, k1, xeq1, m2, k2, xeq2, rho, cd, ar
xv_o = xo1, vo1, xo2, vo2 

tf = 240
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

xv = odeint(integrate, xv_o, ta, args=(p,))

x1 = xv[:,0]
v1 = xv[:,1]
x2 = xv[:,2]
v2 = xv[:,3]

ke = np.zeros(nframes)
pe = np.zeros(nframes)
for i in range(nframes):
	ke[i] = T.subs({M1:m1, M2:m2, X1Dot:v1[i], X2Dot:v2[i]})
	pe[i] = V.subs({K1:k1, K2:k2, XEQ1:xeq1, XEQ2:xeq2, X1:x1[i], X2:x2[i]})
E = ke + pe

fig, a=plt.subplots()

yline = 0
xmax = max(x2)+2*rad
if min(x1) < 0:
	xmin = min(x1)-2*rad
else:
	xmin = -2*rad
ymax = yline + 2*rad
ymin = yline - 2*rad
dx12 = np.zeros(nframes)
for i in range(nframes):
	dx12[i] = x2[i] - x1[i]
nl1 = int(np.ceil((max(x1)+rad)/(2*rad)))
nl2 = int(np.ceil(max(dx12)/(2*rad)))
xl1 = np.zeros((nl1,nframes))
yl1 = np.zeros((nl1,nframes))
xl2 = np.zeros((nl2,nframes))
yl2 = np.zeros((nl2,nframes))
for i in range(nframes):
	l1 = (x1[i]/nl1)
	l2 = (x2[i]-x1[i]-2*rad)/nl2
	xl1[0][i] = x1[i] - rad - 0.5*l1
	xl2[0][i] = x2[i] - rad - 0.5*l2
	for j in range(1,nl1):
		xl1[j][i] = xl1[j-1][i] - l1
	for j in range(1,nl2):
		xl2[j][i] = xl2[j-1][i] - l2
	for j in range(nl1):
		yl1[j][i] = yline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l1)**2))
	for j in range(nl2):
		yl2[j][i] = yline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l2)**2))

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x1[frame],yline),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	circle=plt.Circle((x2[frame],yline),radius=rad,fc='xkcd:light purple')
	plt.gca().add_patch(circle)
	plt.plot([x1[frame]-rad,xl1[0][frame]],[yline,yl1[0][frame]],'xkcd:cerulean')
	plt.plot([xl1[nl1-1][frame],-rad],[yl1[nl1-1][frame],yline],'xkcd:cerulean')
	for i in range(nl1-1):
		plt.plot([xl1[i][frame],xl1[i+1][frame]],[yl1[i][frame],yl1[i+1][frame]],'xkcd:cerulean')
	plt.plot([x2[frame]-rad,xl2[0][frame]],[yline,yl2[0][frame]],'xkcd:cerulean')
	plt.plot([xl2[nl2-1][frame],x1[frame]+rad],[yl2[nl2-1][frame],yline],'xkcd:cerulean')
	for i in range(nl2-1):
		plt.plot([xl2[i][frame],xl2[i+1][frame]],[yl2[i][frame],yl2[i+1][frame]],'xkcd:cerulean')
	plt.title("A Double Horizontal Spring With Air Resistance")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('double_horizontal_spring_w_air_resistance.mp4', writer=writervideo)
plt.show()

