import numpy as np

def xas_conv(w_inc, T, Gamma=2):
    I = np.zeros_like(w_inc)
    
    for trans in T:
        I += trans[1] * Gamma / (np.pi * ((w_inc - trans[0])**2 + Gamma**2))
    
    return I

def rixs_trans(Tgn, Tnf):
    intensity = {}

    for [w_gn, int_gn, in1, in2] in Tgn: # loop g states
        
        matching_nf = [row for row in Tnf if row[-1] == in2] # match n states 
        
        for [w_nf, int_nf, out1, out2] in matching_nf:   
            I = np.abs(int_gn * int_nf)
            intensity[(int(in1), int(in2), int(out1))] = (w_gn, w_nf, I)
            
    return intensity

def rixs_intf(wi, tensor, Gamma_n):
    data = []
    
    default = (0.0, 0.0, np.zeros((3,3), dtype=complex))
    max_g = max(g for (g, n, f) in tensor)
    max_n = max(n for (g, n, f) in tensor)
    max_f = max(f for (g, n, f) in tensor)

    for g in range(1, max_g+1):
        for f in range(1, max_f+1):
            I = 0
            Delta = []
            
            for n in range(1, max_n+1):
                try:
                    w_gn, w_nf, t_gnf = tensor[(g, n, f)]
                except KeyError:
                    continue
                    
                I = I + t_gnf / ((wi - w_gn)**2 + Gamma_n**2)
                Delta.append((w_gn - w_nf))
            
            if Delta:
                if np.var(Delta) >= 1e-10:
                    print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}")
                data.append([np.average(Delta), I])
            
    return np.vstack(data)

def rixs_conv(w_inc, w_los, Tgn, Tnf, Gamma_n=2, Gamma_f=2):
    I = np.zeros((len(w_los), len(w_inc))) 

    print("\rConverting Data...", end='', flush=True)
    
    trans = rixs_trans(Tgn, Tnf)

    for i, wi in enumerate(w_inc):
        inf_result = rixs_intf(wi, trans, Gamma_n)
            
        Delta = inf_result[:, 0]
        Intensity = inf_result[:, 1]
        
        for j, loss in enumerate(w_los): 
            conv = Intensity * (Gamma_f / np.pi) / ((Delta - loss)**2 + Gamma_f**2)
            I[j, i] = ((wi - loss) / wi) * np.sum(conv)
        
        print(f"\rProcessing: {i / len(w_inc) * 100:.2f}% ", end='', flush=True)
    print("\rFinished!          ", flush=True) 
    
    return I
