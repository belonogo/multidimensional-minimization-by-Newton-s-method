import numpy as np
import matplotlib.pyplot as plt


def f(x1, x2):
    return 3*x1**2+x1*x2+x2**2-5.5*x1-6.5*x2


def gradf(x1, x2):
    return np.array([[6*x1+x2-5.5], [x1+2*x2-6.5]])


def Newton_CG(X0, g0, d0, A, eps, iterator=1):
    lnumerator = -(g0.transpose()*d0)
    ldenominator = d0.transpose()*A*d0
    lambda0 = lnumerator.diagonal().sum() / (ldenominator.diagonal().sum() + np.fliplr(ldenominator).diagonal().sum())
    X1 = X0 + lambda0*d0
    g1 = gradf(X1[0][0], X1[1][0])
    bnumerator = g1.transpose()*A*d0
    bdenominator = d0.transpose()*A*d0
    beta0 = bnumerator.diagonal().sum() / (bdenominator.diagonal().sum() + np.fliplr(bdenominator).diagonal().sum())
    d1 = -g1 + beta0*d0
    if ((abs(X1[0][0]-X0[0][0]) <= eps) and (abs(X1[1][0]-X0[1][0]) <= eps)):
        print("Количество итераций для достижения заданной точности:", iterator)
        print("Точка минимума:", X1.transpose())
        return
    else:
        iterator+=1
        Newton_CG(X1, g1, d1, A, eps, iterator)


eps = 10**(-6)
A = np.array([[6, 1], [1, 2]])
X0 = np.array([[0], [0]])
g0 = gradf(X0[0][0], X0[1][0])
d0 = -g0

Newton_CG(X0, g0, d0, A, eps)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(0, 6, 100))
z = 3*x**2+x*y+y**2-5.5*x-6.5*y
ax.plot_surface(x, y, z)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('f(x1,x2)')
plt.show()

fig, ax = plt.subplots(1, 1)
ax.contour(x, y, z, levels=13)
fig.set_figwidth(8)
fig.set_figheight(8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('f(x1,x2)')
plt.show
