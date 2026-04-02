import h5py
import numpy as np

c = 137.035999084
Ha = 27.2113863

def tensor(filename, SOC=True, Operator="E1", Subset=0, GStates=[], Threshold=0): 
    
    with h5py.File(filename, "r") as f:
        if SOC:
            E = f['SOS_ENERGIES'][:] 
            r = f['SOS_EDIPMOM_REAL'][:] + 1j * f['SOS_EDIPMOM_IMAG'][:]
            L = 1j * f['SOS_EDIPMOM_REAL'][:] - f['SOS_EDIPMOM_IMAG'][:]
            S = f['SOS_EDIPMOM_REAL'][:] + 1j * f['SOS_EDIPMOM_IMAG'][:]

        else:
            E = f['SFS_ENERGIES'][:] 
            r = f['SFS_EDIPMOM'][:]
            L = f['SFS_ANGMOM'][:]
            S = np.zeros_like(L)

        r = np.transpose(r, (1, 2, 0))
        L = np.transpose(L, (1, 2, 0))
        S = np.transpose(S, (1, 2, 0))

        Nstates = np.shape(E)[0]
        Ng = Nstates
        nf = 0
        
        if Subset != 0:
            Ng = Subset
            nf = Subset
            
        E1 = False
        M1 = False
        SP = False
        if Operator == "E1":
            E1 = True
        elif Operator == "M1":
            M1 = True
        elif Operator == "SP":
            SP = True
        
        data = {}
        
        for i in range(0, Ng):
            if GStates and i+1 not in GStates:
                continue
            for j in range(nf, Nstates):
                if Subset == 0 and j == i:
                    continue
                    
                g_index = int(i + 1)
                f_index = int(j + 1 - Subset)
                energy = E[j] - E[i]
                
                if E1:
                    tensor = r[i, j] * energy * 1j
                if M1:
                    tensor = (L[i, j] + 2 * S[i, j]) * 1j / 2
                if SP:    # This operator is done by intermediate state insertion over CAS space
                    rxS = np.zeros(3, dtype=complex)
                    for n in range(Nstates):
                        r_fn = r[i, n]
                        S_ng = S[n, j]
                        rxS += np.cross(r_fn, S_ng)
                    tensor = - rxS * energy**2 / (2 * c**2)
                
                if np.sum(np.abs(tensor)) > Threshold:
                    data[(g_index, f_index)] = (energy * Ha, tensor) 

        return np.array(data)