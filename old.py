import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from skimage import measure
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#numero de pontos 
n = 100
n = int(np.sqrt(n))


def f_grade(dom_x,dom_y,dom_z,points,normals,h, kd_tree,K):    
    shape_original = dom_x.shape
    n = points.shape[0]
    
    #achatando os pontos da grade num array (M,3)
    pontos_grade = np.vstack([dom_x.ravel(), dom_y.ravel(), dom_z.ravel()]).T
    M = pontos_grade.shape[0]

    #construção das matrizes, sem misterio
    b = np.zeros(4 * n)
    b[1::4] = normals[:, 0]
    b[2::4] = normals[:, 1]
    b[3::4] = normals[:, 2]

    D = np.zeros((4 * n, 5))
    D[0::4, 0] = 1                          # 1
    D[0::4, 1] = points[:, 0]               # x
    D[0::4, 2] = points[:, 1]               # y
    D[0::4, 3] = points[:, 2]               # z
    D[0::4, 4] = np.sum(points**2, axis=1)  # x² + y² + z²

    D[1::4, 1] = 1
    D[1::4, 4] = 2 * points[:, 0]
    D[2::4, 2] = 1
    D[2::4, 4] = 2 * points[:, 1]
    D[3::4, 3] = 1
    D[3::4, 4] = 2 * points[:, 2]

    #lista com as distancias entre os k vizinhos mais proximos
    k_dist = kd_tree.query(pontos_grade,k=K)[0]

    #maior distancia entre os k vizinhos
    h_din = np.maximum(k_dist[:,-1] * h, 1e-5)

    #new_axis adiciona um novo eixo na posição especificada. em pontos grade, a matriz original que tinha n linhas e m colunas agora tem n linhas, 1 coluna e m de 'profundidade
    #em points, o array original tinha n linhas e m colunas e passou a ter 1 linha, n colunas e m de 'profundidade.a
    #por fim, para fazer a subtração, nós usamos essa dimensão extra para 'empilhar' a informção relevante.a
    #por exemplo, no caso de points, se imaginarmos em 3d a matriz com o new_axis é uma 'folha' (1 de altura, n*m de area). 
    #o que fazemos então é empilhar varias folhas ate bater com a dimensão necessaria
    dif = pontos_grade[:, np.newaxis, :] - points[np.newaxis, :, :]
    distancias = np.linalg.norm(dif, axis=2)

    #calculo explicito do argumento da função phi (||x-x0||/h)
    razao = distancias / h_din[:,np.newaxis]
    #aplicação do kernel. altera o array onde razao < 1 para *. se for > 1 deixa o peso como zero
    pesos = np.where(razao < 1, (1.0 - razao**2)**4, 0.0)
    


    amostras_ativas = np.sum(pesos > 1e-8, axis=1)

    beta = 1e6 * h_din**2
    
    W_diag = np.zeros((M, 4 * n))
    W_diag[:, 0::4] = pesos
    W_diag[:, 1::4] = beta[:,np.newaxis] * pesos
    W_diag[:, 2::4] = beta[:,np.newaxis] * pesos
    W_diag[:, 3::4] = beta[:,np.newaxis] * pesos
    
    D_weighted = D[np.newaxis, :, :] * W_diag[:, :, np.newaxis]
    A = np.matmul(D.T, D_weighted)
    A += 1e-6 * np.eye(5) 

    b_hat = (W_diag * b) @ D

    #linha alinhar as dimensões das matrizes A e b_hat
    b_hat_fixed = b_hat[:,:,np.newaxis]
    #squeeze(-1) retira a ultima dimensão adicionada na linha anterior
    u_batch = np.linalg.solve(A, b_hat_fixed).squeeze(-1)

    P_poly = np.zeros((M, 5))
    P_poly[:, 0] = 1
    P_poly[:, 1] = pontos_grade[:, 0]
    P_poly[:, 2] = pontos_grade[:, 1]
    P_poly[:, 3] = pontos_grade[:, 2]
    P_poly[:, 4] = np.sum(pontos_grade**2, axis=1)

    dom_w_flat = np.sum(P_poly * u_batch, axis=1)

    dom_w_flat[amostras_ativas < 4] = 100.0

    return dom_w_flat.reshape(shape_original)

def RMSQ(dom_x, dom_y, dom_z, dom_w, func):
    dom_w_exact = func(dom_x,dom_y,dom_z)
    valid_mask = dom_w < 99.0

    w_apss = dom_w[valid_mask]
    w_exact = dom_w_exact[valid_mask]

    square_error = (w_apss - w_exact)**2

    rmsq = np.sqrt(np.mean(square_error))

    return rmsq

def error_scalar_field(dom_x, dom_y,dom_z, func):
    real_dom_z = func(dom_x, dom_y)

    error_field = np.abs(dom_z - real_dom_z)

    error_field[dom_z > 99.0] = np.nan

    return error_field

def plot_surface(ax, scalar_field, color):

    verts, faces, _, _ = measure.marching_cubes(scalar_field, level=0.0)
    
    spacing = (max_range - min_range) / (samples - 1)
    verts = min_range + verts * spacing
    
    mesh = Poly3DCollection(verts[faces], alpha=0.8, edgecolor='none')
    mesh.set_facecolor(color)
    mesh.set_edgecolor('black')
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)
        
    ax.set_xlim(min_range, max_range)
    ax.set_ylim(min_range, max_range)
    ax.set_zlim(min_range, max_range)
    ax.set_box_aspect([1, 1, 1])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


tipo = input()

match(tipo):
    #esfera
    case 'sp':
        r = 1
        grad = lambda p: [2*p[0], 2*p[1], 2*p[2]]
        func = lambda x,y,z: (x**2 + y**2 + z**2 - r**2)
        points = []
        for phi in np.linspace(0,np.pi,n):
            for theta in np.linspace(0,2*np.pi,n):
                points.append([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])
        points = np.array(points)
        min_range,max_range = -r,r
    #toro
    case 'to':
        R = 1.5
        r = 1
        grad = lambda p: [(2*(np.sqrt(p[0]**2 + p[1]**2)) - R**2)*p[0]/np.sqrt(p[0]**2 + p[1]**2), 
                            (2*(np.sqrt(p[0]**2 + p[1]**2)) - R**2)*p[1]/np.sqrt(p[0]**2 + p[1]**2),
                            2*p[2]]
        func = lambda x,y,z: ((np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2) 
        points = []
        for theta in np.linspace(0,2*np.pi, n):
            for phi in np.linspace(0,2*np.pi, n):
                points.append([(R + r*np.sin(theta))*np.cos(phi), (R + r*np.sin(theta))*np.sin(phi), r*np.cos(theta)])
        points = np.array(points)
        min_range, max_range = -(R+r),(R+r)
    #cone
    case 'co':
        R = 1
        grad = lambda p: [2*p[0], 2*p[1], -2*p[2]]
        func = lambda x,y,z: (x**2 + y**2 - z**2)
        points = []
        for theta in np.linspace(0,2*np.pi, n):
            for r in np.linspace(-R, R, n):
                points.append([r*np.cos(theta), r*np.sin(theta), r])
        points = np.array(points)
        min_range, max_range = -R, R
    

grads = np.array([grad(p) for p in points])
norms = np.linalg.norm(grads,axis=1,keepdims=True)
norms = np.where(norms == 0.0, 1.0, norms)
normals = grads/norms


samples = 300
samples = int(np.sqrt(samples))

kd_tree = cKDTree(points)

dom_x = np.linspace(min_range, max_range, samples); dom_y = np.linspace(min_range, max_range, samples); dom_z = np.linspace(min_range, max_range, samples)

dom_x, dom_y, dom_z = np.meshgrid(dom_x, dom_y, dom_z)

dom_w = f_grade(dom_x, dom_y, dom_z, points, normals, h=1.8, kd_tree=kd_tree, K=5)

print(f"rmsq = {RMSQ(dom_x,dom_y, dom_z, dom_w, func)}")

dom_w_exact = func(dom_x, dom_y, dom_z)

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(1, 2, 1, projection='3d') 
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.set_title("Superfície Exata Original")
ax2.set_title("Reconstrução APSS")

plot_surface(ax1, dom_w_exact, 'seagreen') 
plot_surface(ax2, dom_w, 'crimson')       

ax2.scatter(points[:,0], points[:,1], points[:,2], alpha=1, color='blue', s=20)

plt.tight_layout()
plt.show()

