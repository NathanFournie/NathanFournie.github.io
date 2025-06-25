import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, simplify
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

def input_param_curve():
    print("Entrez la fonction paramétrée r(t).")
    print("Exemple en 2D : (cos(t), sin(t))")
    print("Exemple en 3D : (cos(t), sin(t), t)")
    expr = input("r(t) = ")
    return expr

def parse_param_curve(expr, t):
    expr = expr.strip()
    if expr.startswith('(') and expr.endswith(')'):
        expr = expr[1:-1]
    components = expr.split(',')
    components = [simplify(c.strip()) for c in components]
    return components

def main():
    t = symbols('t')
    expr = input_param_curve()
    funcs = parse_param_curve(expr, t)
    
    dim = len(funcs)
    if dim not in [2, 3]:
        print("Seules les courbes 2D ou 3D sont supportées.")
        return
    
    tmin = float(input("Intervalle : t_min = "))
    tmax = float(input("Intervalle : t_max = "))
    
    f_numeric = [lambdify(t, f, "numpy") for f in funcs]
    ts = np.linspace(tmin, tmax, 2000)
    points = np.array([f(ts) for f in f_numeric])
    
   #il y a un petit problème avec les couleurs, la fin de la courbe ne
   #correspond pas avec le debut (pour les courbes fermées) en terme de couleur
   #ce qui brise le dégradé, faut donc jouer avec le paramétre t pour que la 
   #courbe cesse sur une même plage de couleur.
    norm = Normalize(vmin=0, vmax=tmax - tmin)
    red_green = LinearSegmentedColormap.from_list("RedGreen", ["blue", "purple", "red"])
    t_mod = (ts[:-1] - tmin) % (tmax - tmin)
    colors = red_green(norm(t_mod))


    
    fig = plt.figure(figsize=(6,6), dpi=600)

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        segments = np.array([
            [[points[0][i], points[1][i], points[2][i]],
             [points[0][i+1], points[1][i+1], points[2][i+1]]]
            for i in range(len(ts) - 1)
        ])
        lc = Line3DCollection(segments, colors=colors, linewidths=1.5)
        ax.add_collection3d(lc)
        ax.set_xlim(points[0].min(), points[0].max())
        ax.set_ylim(points[1].min(), points[1].max())
        ax.set_zlim(points[2].min(), points[2].max())

        changer_vue = input("Changer angle de vue ? (oui/non) : ").lower()
        if changer_vue == 'oui':
            elev = float(input("Élévation (exemple 30) : "))
            azim = float(input("Azimut (exemple 45) : "))
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=15, azim=0)
    else:
        ax = fig.add_subplot(111)
        segments = np.array([
            [[points[0][i], points[1][i]],
             [points[0][i+1], points[1][i+1]]]
            for i in range(len(ts) - 1)
        ])
        lc = LineCollection(segments, colors=colors, linewidths=1.5)
        ax.add_collection(lc)
        x_min, x_max = points[0].min(), points[0].max()
        y_min, y_max = points[1].min(), points[1].max()
        x_centre = (x_min + x_max)/2
        y_centre = (y_min + y_max)/2
        demi_range = max(x_max - x_min, y_max - y_min)/2
        ax.set_xlim(x_centre - demi_range, x_centre + demi_range)
        ax.set_ylim(y_centre - demi_range, y_centre + demi_range)
        ax.set_aspect('equal')

    
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    if dim == 3:
        ax.zaxis.set_ticks([])
    ax.grid(False)

    # Cercle osculateur
    cercle_oscul = input("Afficher le cercle osculateur ? (oui/non) : ").lower()
    if cercle_oscul == 'oui':
        t0 = float(input("En quel point t0 ? "))
        r_t0 = np.array([f_numeric[i](t0) for i in range(dim)], dtype=float)
        r_prime = [diff(f, t) for f in funcs]
        r_prime_prime = [diff(f_prime, t) for f_prime in r_prime]
        r_prime_num = np.array([lambdify(t, f, "numpy")(t0) for f in r_prime], dtype=float)
        r_prime_prime_num = np.array([lambdify(t, f, "numpy")(t0) for f in r_prime_prime], dtype=float)
        norm_r_prime = np.linalg.norm(r_prime_num)
        T = r_prime_num / norm_r_prime
        
        if dim == 3:
            cross = np.linalg.norm(np.cross(r_prime_num, r_prime_prime_num))
            kappa = cross / norm_r_prime**3
        else:
            kappa = abs(r_prime_num[0]*r_prime_prime_num[1] - r_prime_num[1]*r_prime_prime_num[0]) / norm_r_prime**3
        
        if kappa == 0:
            print("Courbure nulle en ce point, pas de cercle osculateur.")
        else:
            R = 1 / kappa
            if dim == 2:
                N = np.array([-T[1], T[0]])
            else:
                B = np.cross(r_prime_num, r_prime_prime_num)
                B /= np.linalg.norm(B)
                N = np.cross(B, T)
            C = r_t0 + R * N
            
            angle = np.linspace(0, 2*np.pi, 300)
            if dim == 2:
                x_cercle = C[0] + R * np.cos(angle)
                y_cercle = C[1] + R * np.sin(angle)
                ax.plot(x_cercle, y_cercle, color='blue', linestyle='dashed')
                ax.plot(r_t0[0], r_t0[1], 'o', color='blue')
            else:
                cercle_pts = np.array([C + R*(np.cos(a)*N + np.sin(a)*B) for a in angle])
                ax.plot(cercle_pts[:,0], cercle_pts[:,1], cercle_pts[:,2], color='maroon', linestyle='dashed')
                ax.scatter(r_t0[0], r_t0[1], r_t0[2], color='maroon')

    # Plan osculateur
    plan_oscul = input("Afficher le plan osculateur ? (oui/non) : ").lower()
    if plan_oscul == 'oui':
        t0 = float(input("En quel point t0 ? "))
        r_t0 = np.array([f_numeric[i](t0) for i in range(dim)], dtype=float)
        r_prime = [diff(f, t) for f in funcs]
        r_prime_prime = [diff(f_prime, t) for f_prime in r_prime]
        r_prime_num = np.array([lambdify(t, f, "numpy")(t0) for f in r_prime], dtype=float)
        r_prime_prime_num = np.array([lambdify(t, f, "numpy")(t0) for f in r_prime_prime], dtype=float)
        norm_r_prime = np.linalg.norm(r_prime_num)
        T = r_prime_num / norm_r_prime
        
        if dim != 3:
            print("Plan osculateur uniquement pour courbes en 3D.")
        else:
            B = np.cross(r_prime_num, r_prime_prime_num)
            norm_B = np.linalg.norm(B)
            if norm_B == 0:
                print("Torsion nulle en ce point, plan osculateur non défini.")
            else:
                B /= norm_B
                N = np.cross(B, T)
                R = 2
                u = np.linspace(-R, R, 10)
                v = np.linspace(-R, R, 10)
                U, V = np.meshgrid(u, v)
                P = r_t0[:, None, None] + U[None,:,:]*T[:, None, None] + V[None,:,:]*N[:, None, None]
                ax.plot_surface(P[0], P[1], P[2], color='green', alpha=0.15)

    plt.show()


main()
