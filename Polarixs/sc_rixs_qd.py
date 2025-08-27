import numpy as np

def sc_qd_tensor(Tgn, Tnf, R, filterinl=np.diag([1, 1, 1]), filterinr=np.diag([1, 1, 1]), filterout=np.diag([1, 1, 1])):
    tensor = {}

    for [w_gn, XXgn, XYgn, XZgn, YYgn, YZgn, ZZgn, in1, in2] in Tgn: # loop g states
        
        matching_nf = [row for row in Tnf if row[-1].real == in2.real] # match n states
        
        for [w_nf, Xnf, Ynf, Znf, out1, out2] in matching_nf: 
            t_gn = R @ filterinl @ np.array([[XXgn, XYgn, XZgn], [XYgn, YYgn, YZgn], [XZgn, YZgn, ZZgn]]) @ filterinr @ R.T
            t_nf = R @ filterout @ np.array([Xnf, Ynf, Znf]).conj()
            t_gnf = t_gn[:, :, None] * t_nf[None, None, :]
                        
            tensor[(int(in1.real), int(in2.real), int(out1.real))] = (w_gn, w_nf, t_gnf)
            
    return tensor

def sc_qd_ang_intf(wi, tensor, Gamma_n, ei, eo, k):
    data = []
    
    max_g = max(g for (g, n, f) in tensor)
    max_n = max(n for (g, n, f) in tensor)
    max_f = max(f for (g, n, f) in tensor)

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

            I = np.sum(np.abs(V * SumT)**2)
                       
            if Delta:
                if np.var(Delta) >= 1e-8:
                    print(f"Error: Not Matched DE_g,f! State Index: g={g}, f={f}")
                data.append([np.average(Delta), I.real])
            
    return np.vstack(data)

def sc_qd_conv(w_inc, w_los, Tgn, Tnf, R, Gamma_n=2, Gamma_f=2, theta=45, phii=0, phio=0, filterinl=np.diag([1, 1, 1]), filterinr=np.diag([1, 1, 1]), filterout=np.diag([1, 1, 1])):
    I = np.zeros((len(w_los), len(w_inc))) 
    
    print("\rConverting Data...", end='', flush=True)
    tensor = sc_qd_tensor(Tgn, Tnf, R, filterinl, filterinr, filterout)

    ei = np.array([np.sin(phii*np.pi/180), 0, np.cos(phii*np.pi/180)])
    eo = np.array([np.sin(phio*np.pi/180), -np.sin(2*theta*np.pi/180)*np.cos(phio*np.pi/180), -np.cos(2*theta*np.pi/180)*np.cos(phio*np.pi/180)])
    k = np.array([0, -np.cos(2*theta*np.pi/180), np.sin(2*theta*np.pi/180)])
    
    V = ei[:, None, None] * eo[None, :, None] * ei[None, None, :]

    for i, wi in enumerate(w_inc):
        inf_result = sc_qd_ang_intf(wi, tensor, Gamma_n, V)
            
        Delta = inf_result[:, 0]
        Intensity = inf_result[:, 1]
        
        for j, loss in enumerate(w_los): 
            conv = Intensity * (Gamma_f / np.pi) / ((Delta - loss)**2 + Gamma_f**2)
            I[j, i] = ((wi - loss) / wi) * np.sum(conv)
        
        print(f"\rProcessing: {i / len(w_inc) * 100:.2f}% ", end='', flush=True)
    print("\rFinished!          ", flush=True) 
    
    return I
