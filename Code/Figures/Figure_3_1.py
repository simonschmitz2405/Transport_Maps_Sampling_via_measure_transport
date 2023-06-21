import numpy as np
import matplotlib.pyplot as plt  
import scipy.stats as sts 
from scipy import integrate
from matplotlib.patches import Polygon


def f(x):
    return 0.5*sts.norm.pdf(x,-1,1.3) + 0.5*sts.norm.pdf(x,2,0.6)

def h(x):
    return 0.2*sts.norm.pdf(x,-2,0.6) + 0.8*sts.norm.pdf(x,3,3)


x = np.linspace(-5,12,100)
y1 = f(x)
y2 = h(x)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(x,y1,'r',linewidth=1,color='0.5')
ax2.plot(x,y2,'r',linewidth=1,color='0.5555')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

a = -2.5
b = 1.0


ixx1 = f(x)
verts = [(-5,0),*zip(x,ixx1),(12,0)]
poly = Polygon(verts,facecolor='0.9',edgecolor = '0.5')
ax1.add_patch(poly)

ixx2 = h(x)
verts = [(-5,0),*zip(x,ixx2),(12,0)]
poly = Polygon(verts,facecolor='0.9',edgecolor = '0.5')
ax2.add_patch(poly)



# Color space A
ix = np.linspace(a,b)
iy = h(ix)
verts = [(a,0),*zip(ix,iy),(b,0)]
poly = Polygon(verts,facecolor='#8cb63c',edgecolor = '0.5')
ax2.add_patch(poly)

ix = np.linspace(a,b)
iy = f(ix)
verts = [(a,0),*zip(ix,iy),(b,0)]
poly = Polygon(verts,facecolor='#8cb63c',edgecolor = '0.5')
ax1.add_patch(poly)

fig.text(0.9,0.05,'$x$')
fig.text(0.45,0.05,'$x$')
fig.text(0.1,0.9,'$y$')
fig.text(0.55,0.9,'$y$')

ax1.text(0.5 * (a + b), -0.02, r"$T^{-1}(A)$",
        horizontalalignment='center', fontsize=10, color='black')
ax2.text(0.5 * (a + b), -0.01, r'A',
        horizontalalignment='center', fontsize=10, color='black')

ax1.spines[['top','right']].set_visible(False)
ax2.spines[['top','right']].set_visible(False)
ax1.set_yticks([])
ax2.set_yticks([])
ax1.set_xticks([])
ax2.set_xticks([])
plt.show()



