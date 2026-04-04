import numpy as np

c = 137.035999084
Ha = 27.2113863

############################
#  XAS Direct Convolution  #
############################
def xas_conv(w_inc, T, Gamma=2, modulus_square=False):
    I = np.zeros_like(w_inc)

    for energy, trans in T.values():
        if modulus_square:
            I += np.mean(np.abs(trans)**2) * Gamma / (np.pi * ((w_inc - energy)**2 + Gamma**2))
        else: 
            I += np.mean(np.abs(trans)) * Gamma / (np.pi * ((w_inc - energy)**2 + Gamma**2))
            
    return I

##########################
#  Generate RIXS Tensor  #
##########################
def build_tensor(Tgn, Tnf):
    tensor = {}

    for (g, n), (w_gn, t_gn) in Tgn.items():

        for (f, n2), (w_nf, t_nf) in Tnf.items():

            if n2 != n:
                continue

            t_gnf = np.tensordot(t_gn, t_nf.conj(), axes=0) # Notice, There is no conjugation for the emission.

            tensor[(g, n, f)] = (w_gn, w_nf, t_gnf)

    return tensor

#############################
#  RIXS Direct Convolution  #
#############################
from collections import defaultdict

def rixs_intf(wi, tensor, Gamma_n, modulus_square=False):
    data = []

    gf_map = defaultdict(list)
    for (g, n, f), (w_gn, w_nf, t_gnf) in tensor.items():
        gf_map[(g, f)].append((n, w_gn, w_nf, t_gnf))

    for (g, f), entries in gf_map.items():
        I = 0
        Delta = []

        for n, w_gn, w_nf, t_gnf in entries:
            if modulus_square:    
                I = I + np.mean(np.abs(t_gnf)**2) / ((wi - w_gn)**2 + Gamma_n**2)
            else:
                I = I + np.mean(np.abs(t_gnf)) / ((wi - w_gn)**2 + Gamma_n**2)
            Delta.append((w_gn - w_nf))
            
        if Delta:
            var = np.var(Delta)
            if var >= 1e-8:
                print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}, var={var} eV")
            data.append([np.average(Delta), I])
            
    return np.vstack(data)

def rixs_conv_wi(args):
    (wi, w_los, tensor, Gamma_n, Gamma_f, modulus_square) = args

    inf_result = rixs_intf(wi, tensor, Gamma_n, modulus_square)

    Delta = inf_result[:, 0]
    Intensity = inf_result[:, 1]

    I_col = np.zeros(len(w_los)) 
    
    for i, loss in enumerate(w_los): 
        conv = Intensity * (Gamma_f / np.pi) / ((Delta - loss)**2 + Gamma_f**2)
        I_col[i] = ((wi - loss) / wi) * np.sum(conv)

    return I_col

def rixs_conv(w_inc, w_los, tensor, Gamma_n=2, Gamma_f=2, modulus_square=False, status=True):

    I = np.zeros((len(w_los), len(w_inc)))

    for i, wi in enumerate(w_inc):

        args = (wi, w_los, tensor, Gamma_n, Gamma_f, modulus_square)

        I[:, i] = rixs_conv_wi(args)

        if status:
            print(f"\rProcessing: {i / len(w_inc) * 100:.2f}% ", end='', flush=True)

    if status:
        print("\rFinished!          ", flush=True)

    return I

from tqdm.contrib.concurrent import process_map

def rixs_conv_pal(
    w_inc, w_los, tensor, Gamma_n=2, Gamma_f=2, modulus_square=False,
    status=True, max_workers=None, chunksize=1
):

    if not tensor:
        return np.zeros((len(w_los), len(w_inc)))

    args = [
        (wi, w_los, tensor, Gamma_n, Gamma_f, modulus_square)
        for wi in w_inc
    ]

    results = process_map(
        rixs_conv_wi,
        args,
        max_workers=max_workers,
        chunksize=chunksize,
        disable=not status
    )

    I = np.column_stack(results)

    return I
