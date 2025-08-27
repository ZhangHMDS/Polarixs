import numpy as np

def pw_qd_tensor(Tgn, Tnf, filterinl=np.diag([1, 1, 1]), filterinr=np.diag([1, 1, 1]), filterout=np.diag([1, 1, 1])):
    tensor = {}

    for [w_gn, XXgn, XYgn, XZgn, YYgn, YZgn, ZZgn, in1, in2] in Tgn: # loop g states
        
        matching_nf = [row for row in Tnf if row[-1].real == in2.real] # match n states
        
        for [w_nf, Xnf, Ynf, Znf, out1, out2] in matching_nf: 
            t_gn = filterinl @ np.array([[XXgn, XYgn, XZgn], [XYgn, YYgn, YZgn], [XZgn, YZgn, ZZgn]]) @ filterinr
            t_nf = filterout @ np.array([Xnf, Ynf, Znf]).conj()
            t_gnf = t_gn[:, :, None] * t_nf[None, None, :]
                        
            tensor[(int(in1.real), int(in2.real), int(out1.real))] = (w_gn, w_nf, t_gnf)
            
    return tensor

def pw_qd_approx(wi, tensor, Gamma_n):
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
                    
                I = I + np.sum(np.abs(t_gnf)**2) / (((wi - w_gn)**2 + Gamma_n**2) * 27)
                Delta.append((w_gn - w_nf).real)
            
            if Delta:
                if np.var(Delta) >= 1e-8:
                    print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}")
                data.append([np.average(Delta), I.real])
            
    return np.vstack(data)

def pw_qd_ang_intf(wi, tensor, Gamma_n, theta, phii = 0, phio = None):
    data = []
    
    max_g = max(g for (g, n, f) in tensor)
    max_n = max(n for (g, n, f) in tensor)
    max_f = max(f for (g, n, f) in tensor)
    
    if phio is None:
        dep = np.sin(2*theta*np.pi/180)**2 * np.sin(phii*np.pi/180)**2 + 1
    else:
        dep = 2 * (np.cos(2*theta*np.pi/180) * np.cos(phii*np.pi/180) * np.cos(phio*np.pi/180) + np.sin(phii*np.pi/180) * np.sin(phio*np.pi/180))**2 + 2 * np.sin(2*theta*np.pi/180)**2 * np.cos(phio*np.pi/180)**2

    for g in range(1, max_g+1):
        for f in range(1, max_f+1):
            SumT = np.zeros([3, 3, 3], dtype=complex)
            Delta = []
            
            for n in range(1, max_n+1):
                try:
                    w_gn, w_nf, t_gnf = tensor[(g, n, f)]
                except KeyError:
                    continue
                    
                SumT = SumT + t_gnf / (wi - w_gn + Gamma_n * 1j)
                Delta.append((w_gn - w_nf).real)

            T1 = np.einsum('ijk,ijk->', SumT.conjugate(), SumT)
            T2 = np.einsum('ijk,ikj->', SumT.conjugate(), SumT)
            T3 = np.einsum('ijj,ikk->', SumT.conjugate(), SumT)
            A = (16*T1 - 10*T2 - 10*T3) / 210
            B = (-10*T1 + 15*T2 + 15*T3) / 210
            I = A + B * dep
                       
            if Delta:
                if np.var(Delta) >= 1e-8:
                    print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}")
                data.append([np.average(Delta), I.real])
            
    return np.vstack(data)

def pw_qd_conv(w_inc, w_los, Tgn, Tnf, Gamma_n=2, Gamma_f=2, AngDep=True, theta=45, phii=0, phio=None, filterinl=np.diag([1, 1, 1]), filterinr=np.diag([1, 1, 1]), filterout=np.diag([1, 1, 1])):
    I = np.zeros((len(w_los), len(w_inc))) 
    
    print("\rConverting Data...", end='', flush=True)
    tensor = pw_qd_tensor(Tgn, Tnf, filterinl, filterinr, filterout)

    for i, wi in enumerate(w_inc):
        if AngDep:
            inf_result = pw_qd_ang_intf(wi, tensor, Gamma_n, theta, phii, phio)
        else:
            inf_result = pw_qd_approx(wi, tensor, Gamma_n)
            
        Delta = inf_result[:, 0]
        Intensity = inf_result[:, 1]
        
        for j, loss in enumerate(w_los): 
            conv = Intensity * (Gamma_f / np.pi) / ((Delta - loss)**2 + Gamma_f**2)
            I[j, i] = ((wi - loss) / wi) * np.sum(conv)
        
        print(f"\rProcessing: {i / len(w_inc) * 100:.2f}% ", end='', flush=True)
    print("\rFinished!          ", flush=True) 
    
    return I