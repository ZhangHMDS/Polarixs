import numpy as np

def pw_dd_tensor(Tgn, Tnf, filterin=np.diag([1, 1, 1]), filterout=np.diag([1, 1, 1])):
    tensor = {}

    for [w_gn, Xgn, Ygn, Zgn, in1, in2] in Tgn: # loop g states
        
        matching_nf = [row for row in Tnf if row[-1].real == in2.real] # match n states
        
        for [w_nf, Xnf, Ynf, Znf, out1, out2] in matching_nf: 
            t_gn = filterin @ np.array([Xgn, Ygn, Zgn])
            t_nf = filterout @ np.array([Xnf, Ynf, Znf]).conj()
            t_gnf = np.outer(t_gn, t_nf) 
                        
            tensor[(int(in1.real), int(in2.real), int(out1.real))] = (w_gn, w_nf, t_gnf)
            
    return tensor

def pw_dd_approx(wi, tensor, Gamma_n):
    data = []
    
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
                    
                I = I + np.trace(t_gnf.conjugate().T @ t_gnf) / (((wi - w_gn)**2 + Gamma_n**2) * 9)
                Delta.append((w_gn - w_nf).real)
            
            if Delta:
                if np.var(Delta) >= 1e-8:
                    print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}")
                data.append([np.average(Delta), I.real])
            
    return np.vstack(data)

def pw_dd_ang_intf(wi, tensor, Gamma_n, theta, phii = 0, phio = None):
    data = []
    
    max_g = max(g for (g, n, f) in tensor)
    max_n = max(n for (g, n, f) in tensor)
    max_f = max(f for (g, n, f) in tensor)
    
    if phio is None:
        dep = np.cos(2*theta*np.pi/180)**2 * np.cos(phii*np.pi/180)**2 + np.sin(phii*np.pi/180)**2
    else:
        dep = (np.cos(2*theta*np.pi/180) * np.cos(phii*np.pi/180) * np.cos(phio*np.pi/180) + np.sin(phii*np.pi/180) * np.sin(phio*np.pi/180))**2

    for g in range(1, max_g+1):
        for f in range(1, max_f+1):
            SumT = np.zeros([3, 3], dtype=complex)
            Delta = []
            
            for n in range(1, max_n+1):
                try:
                    w_gn, w_nf, t_gnf = tensor[(g, n, f)]
                except KeyError:
                    continue
                    
                SumT = SumT + t_gnf / (wi - w_gn + Gamma_n * 1j)
                Delta.append((w_gn - w_nf).real)

            T1 = np.einsum('ij,ij->', SumT.conjugate(), SumT)
            T2 = np.einsum('ii,jj->', SumT.conjugate(), SumT)
            T3 = np.einsum('ij,ji->', SumT.conjugate(), SumT)
            A = (4*T1 - T2 - T3) / 30
            B = (-2*T1 + 3*T2 + 3*T3) / 30
            I = A + 0.5 * B * dep
                       
            if Delta:
                if np.var(Delta) >= 1e-10:
                    print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}")
                data.append([np.average(Delta), I.real])
            
    return np.vstack(data)

def pw_dd_conv(w_inc, w_los, Tgn, Tnf, Gamma_n=2, Gamma_f=2, AngDep=True, theta=45, phii=0, phio=None, filterin=np.diag([1, 1, 1]), filterout=np.diag([1, 1, 1])):
    I = np.zeros((len(w_los), len(w_inc))) 
    
    print("\rConverting Data...", end='', flush=True)
    tensor = pw_dd_tensor(Tgn, Tnf, filterin, filterout)

    for i, wi in enumerate(w_inc):
        if AngDep:
            inf_result = pw_dd_ang_intf(wi, tensor, Gamma_n, theta, phii, phio)
        else:
            inf_result = pw_dd_approx(wi, tensor, Gamma_n)
            
        Delta = inf_result[:, 0]
        Intensity = inf_result[:, 1]
        
        for j, loss in enumerate(w_los): 
            conv = Intensity * (Gamma_f / np.pi) / ((Delta - loss)**2 + Gamma_f**2)
            I[j, i] = ((wi - loss) / wi) * np.sum(conv)
        
        print(f"\rProcessing: {i / len(w_inc) * 100:.2f}% ", end='', flush=True)
    print("\rFinished!          ", flush=True) 
    
    return I
