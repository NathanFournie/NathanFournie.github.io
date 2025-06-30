import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import math

# Paramètres
res = 100
R = 2.0
x = np.linspace(-R, R, res)
y = np.linspace(-R, R, res)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

def partial_sum(a_n, z, N):
    val = 0
    for n in range(min(N, len(a_n))):
        val += a_n[n] * (z ** n)
    return val

def lire_expression_et_generer_coeffs(max_N=50):
    print("Entrez l'expression de a_n en fonction de n (ex: 1/math.factorial(n), 1/(n**2+1), 0.5**n):")
    expr = input()
    coeffs = []
    for n in range(max_N + 1):

        safe_dict = {'n': n, 'math': math, 'np': np, 'complex': complex}
        try:
            val = eval(expr, {"__builtins__": None}, safe_dict)
        except Exception as e:
            print(f"Erreur d'évaluation pour n={n}: {e}")
            return None
        coeffs.append(val)
    return coeffs

while True:
    a_n = lire_expression_et_generer_coeffs(50)
    if a_n is not None:
        break
    print("Réessaie avec une expression valide.")

N_values = list(range(1, len(a_n) + 1))

tmp_dir = "tmp_gif_images"
os.makedirs(tmp_dir, exist_ok=True)

filenames = []

for idx, N in enumerate(N_values):
    S = np.zeros_like(Z, dtype=np.complex128)
    for i in range(res):
        for j in range(res):
            z = Z[i, j]
            S[i, j] = partial_sum(a_n, z, N)

    plt.figure()
    plt.imshow(np.abs(S), extent=(-R, R, -R, R), origin='lower', cmap='plasma', vmin=0, vmax=10)
    plt.title(f"|S_N(z)| pour N = {N}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.colorbar(label="|S_N(z)|")

    filename = f"{tmp_dir}/frame_{idx}.png"
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)

gif_path = "animation_S_N.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.8) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF créé ici : {gif_path}")

for filename in filenames:
    os.remove(filename)
os.rmdir(tmp_dir)
