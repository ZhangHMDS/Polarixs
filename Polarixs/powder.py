import numpy as np

c = 137.035999084
Ha = 27.2113863

############################
#  Inversed Gram Matrixes  #
############################
G_inv = {
    2: 1/3 * np.array([[1]]),
    4: 1/30 * np.array([[ 4, -1, -1],
                        [-1,  4, -1],
                        [-1, -1,  4]]),
    6: 1/210 * np.array([[16, -5, -5, -5,  2,  2, -5,  2,  2,  2,  2, -5,  2,  2, -5],
                         [-5, 16, -5,  2, -5,  2,  2,  2, -5, -5,  2,  2,  2, -5,  2],
                         [-5, -5, 16,  2,  2, -5,  2, -5,  2,  2, -5,  2, -5,  2,  2],
                         [-5,  2,  2, 16, -5, -5, -5,  2,  2,  2, -5,  2,  2, -5,  2],
                         [ 2, -5,  2, -5, 16, -5,  2, -5,  2, -5,  2,  2,  2,  2, -5],
                         [ 2,  2, -5, -5, -5, 16,  2,  2, -5,  2,  2, -5, -5,  2,  2],
                         [-5,  2,  2, -5,  2,  2, 16, -5, -5, -5,  2,  2, -5,  2,  2],
                         [ 2,  2, -5,  2, -5,  2, -5, 16, -5,  2, -5,  2,  2,  2, -5],
                         [ 2, -5,  2,  2,  2, -5, -5, -5, 16,  2,  2, -5,  2, -5,  2],
                         [ 2, -5,  2,  2, -5,  2, -5,  2,  2, 16, -5, -5, -5,  2,  2],
                         [ 2,  2, -5, -5,  2,  2,  2, -5,  2, -5, 16, -5,  2, -5,  2],
                         [-5,  2,  2,  2,  2, -5,  2,  2, -5, -5, -5, 16,  2,  2, -5],
                         [ 2,  2, -5,  2,  2, -5, -5,  2,  2, -5,  2,  2, 16, -5, -5],
                         [ 2, -5,  2, -5,  2,  2,  2,  2, -5,  2, -5,  2, -5, 16, -5],
                         [-5,  2,  2,  2, -5,  2,  2, -5,  2,  2,  2, -5, -5, -5, 16]])
}

########################
#  Delta Combinations  #
########################
Delta_index = {
    2: ['i,i->'],
    4: ['ii,jj->', 'ij,ij->', 'ij,ji->'],
    6: ['iij,jkk->', 'iij,kjk->', 'iij,kkj->',
        'iji,jkk->', 'iji,kjk->', 'iji,kkj->',
        'ijj,ikk->', 'ijk,ijk->', 'ijk,ikj->',
        'ijj,kik->', 'ijk,jik->', 'ijk,kij->',
        'ijj,kki->', 'ijk,jki->', 'ijk,kji->']
}

#########################
#  Polarization Tensor  #
#########################
def polartensor_even(Operator, alpha, phii, psii, phio, psio):

    alpha, phii, psii, phio, psio = np.deg2rad([alpha, phii, psii, phio, psio])
    
    #### incident
    epsi = np.array([np.cos(phii), 0, np.sin(phii) * np.exp(1j*psii)])
    ki = np.array([0, 1, 0])

    #### outgoing
    epso = np.array([np.cos(phio) * np.cos(alpha), - np.cos(phio) * np.sin(alpha), np.sin(phio) * np.exp(1j*psio)]).conj()
    ko = np.array([np.sin(alpha), np.cos(alpha), 0]).conj()

    #### build tensor
    tensor = None
    
    if Operator == "E1":
        tensor = epsi
    if Operator == "E2":
        tensor = np.tensordot(epsi, ki, axes=0)
    if Operator == "M1":
        tensor = np.cross(ki, epsi)

    if Operator == "E1E1":
        ti = epsi
        to = epso
    if Operator == "E1E2":
        ti = epsi
        to = np.tensordot(epso, ko, axes=0)
    if Operator == "E2E1":
        ti = np.tensordot(epsi, ki, axes=0)
        to = epso
    if Operator == "E1M1":
        ti = epsi
        to = np.cross(ko, epso)
    if Operator == "M1E1":
        ti = np.cross(ki, epsi)
        to = epso
    if Operator == "M1M1":
        ti = np.cross(ki, epsi)
        to = np.cross(ko, epso)

    if tensor is None:
        tensor = np.tensordot(ti, to.conj(), axes=0)

    return tensor

#########
#  XAS  #
#########
def xas(w_inc, tensor, Operator, Gamma=2, status=True, phii=0, psii=0):
    I = np.zeros_like(w_inc)

    polartensor = polartensor_even(Operator, 0, phii, psii, 0, 0)

    polardim = polartensor.ndim
    delta = Delta_index[2*polardim]
    G = G_inv[2*polardim]

    p_array = np.array([np.einsum(d, polartensor.conjugate(), polartensor) for d in delta])
    
    for energy, trans in tensor.values():
        t_array = np.array([np.einsum(d, trans.conjugate(), trans) for d in delta])

        intf = np.einsum('i,ij,j->', p_array, G, t_array).real

        I += intf * Gamma / (np.pi * ((w_inc - energy)**2 + Gamma**2))
        
    if Operator == "E2":
        I = I * (w_inc / (c * Ha))**2
        
    #### pre factor
    I = (4 * np.pi**2 / c) * (I * Ha / w_inc)
    
    return I
    
##########
#  RIXS  #
##########
from collections import defaultdict

def rixs_inc(wi, gf_map, polardim, p_array, Gamma_n):
    data = []

    G = G_inv[2*polardim]
    delta = Delta_index[2*polardim]

    for (g, f), entries in gf_map.items():
        SumT = np.zeros((3,) * polardim, dtype=complex)
        Delta = []

        for n, w_gn, w_nf, t_gnf in entries:
            SumT += Ha * t_gnf / (wi - w_gn + Gamma_n * 1j)    # Corrected to Ha energy
            Delta.append((w_gn - w_nf).real)

        t_array = np.array([np.einsum(d, SumT.conjugate(), SumT) for d in delta])

        I = np.einsum('i,ij,j->', p_array, G, t_array)

        if Delta:
            if np.var(Delta) >= 1e-8:
                print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}")

            data.append([np.mean(Delta), I.real])

    return np.vstack(data)

def rixs_inc_conv(args):
    (wi, w_los, gf_map, Operator, polardim, p_array, Gamma_n, Gamma_f, energy_emission) = args

    inf_result = rixs_inc(wi, gf_map, polardim, p_array, Gamma_n)

    Delta = inf_result[:, 0]
    Intensity = inf_result[:, 1]
    
    I_col = np.zeros(len(w_los))

    if energy_emission:
        w_los = wi - w_los

    for i, loss in enumerate(w_los): 
        conv = Intensity * (Gamma_f / np.pi) / ((Delta - loss)**2 + Gamma_f**2)
        I_col[i] = ((wi - loss) / wi) * np.sum(conv)

    #### Intensity correction of E2
        if Operator == "E1E2":
            I_col[i] = I_col[i] * ((wi - loss) / (c * Ha))**2
    if Operator == "E2E1":
        I_col = I_col * (wi / (c * Ha))**2

    return I_col

def rixs(w_inc, w_los, tensor, Operator, Gamma_n=2, Gamma_f=2, energy_emission=False, status=True,
         alpha=90, phii=0, psii=0, phio=None, psio=0):

    if phio is None:
        polartensor = polartensor_even(Operator, alpha, phii, psii, 0, psio)    # Default sigma detection for phio
    else:
        polartensor = polartensor_even(Operator, alpha, phii, psii, phio, psio)

    polardim = polartensor.ndim
    delta = Delta_index[2*polardim]
    
    p_array = np.array([np.einsum(d, polartensor.conjugate(), polartensor) for d in delta])

    if phio is None:    # Additional average with pi detection
        polartensor = polartensor_even(Operator, alpha, phii, psii, 90, psio)
        p_array += np.array([np.einsum(d, polartensor.conjugate(), polartensor) for d in delta])
        p_array = 0.5 * p_array

    I = np.zeros((len(w_los), len(w_inc)))

    gf_map = defaultdict(list)
    for (g, n, f), (w_gn, w_nf, t_gnf) in tensor.items():
        gf_map[(g, f)].append((n, w_gn, w_nf, t_gnf))
    
    for i, wi in enumerate(w_inc):
        args = (wi, w_los, gf_map, Operator, polardim, p_array, Gamma_n, Gamma_f, energy_emission)   
        I[:, i] = rixs_inc_conv(args)
        
        if status:
            print(f"\rProcessing: {i / len(w_inc) * 100:.2f}% ", end='', flush=True)

    if status:
        print("\rFinished!          ", flush=True) 

    #### pre factor
    I = I / c**4
    
    return I

from tqdm.contrib.concurrent import process_map

def rixs_pal(w_inc, w_los, tensor, Operator, Gamma_n=2, Gamma_f=2, energy_emission=False,
    alpha=90, phii=0, psii=0, phio=None, psio=0, 
    status=True, max_workers=None, chunksize=1
):

    if phio is None:
        polartensor = polartensor_even(Operator, alpha, phii, psii, 0, 0)    # Default sigma detection for phio
    else:
        polartensor = polartensor_even(Operator, alpha, phii, psii, phio, psio)

    polardim = polartensor.ndim
    delta = Delta_index[2*polardim]
    
    p_array = np.array([np.einsum(d, polartensor.conjugate(), polartensor) for d in delta])

    if phio is None:    # Additional average with pi detection
        polartensor = polartensor_even(Operator, alpha, phii, psii, 90, 0)
        p_array += np.array([np.einsum(d, polartensor.conjugate(), polartensor) for d in delta])
        p_array = 0.5 * p_array    

    if not tensor:
        return np.zeros((len(w_los), len(w_inc)))

    gf_map = defaultdict(list)
    for (g, n, f), (w_gn, w_nf, t_gnf) in tensor.items():
        gf_map[(g, f)].append((n, w_gn, w_nf, t_gnf))

    args = [
        (wi, w_los, gf_map, Operator, polardim, p_array, Gamma_n, Gamma_f, energy_emission)
        for wi in w_inc
    ]

    results = process_map(
        rixs_inc_conv,
        args,
        max_workers=max_workers,
        chunksize=chunksize,
        disable=not status
    )

    I = np.column_stack(results)

    #### pre factor
    I = I / c**4

    return I
