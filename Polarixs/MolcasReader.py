import re
import numpy as np

#########################
#                       #
#  State Energy Reader  #
#                       #
#########################
def Molcas_eigenE(filename, SOC=False):
    data = []

    if SOC:
        start_line = 'SO State' 
    else:
        start_line = 'SF State' 
        
    with open(filename, 'r') as f:
        lines = f.readlines()

    table_start = False
    skip_line = False

    for line in lines:
        if start_line in line:
            table_start = True
            skip_line = True
            continue
        if table_start:
            if skip_line:
                skip_line = False
                continue

            if line.strip() == "" or not re.match(r'^\s*\d+', line):
                break

            match = re.match(r'^\s*(\d+)\s+([-0-9.Ee]+)\s+([-0-9.Ee]+)\s+([-0-9.Ee]+)', line)
            if match:
                state = int(match.group(1))
                rel_emin = float(match.group(2))
                rel_ev = float(match.group(3))
                rel_cm1 = float(match.group(4))
                data.append([state, rel_emin, rel_ev, rel_cm1])
                
    if not data:
        print("Energy Data Reading Failed!")
        
    return data

################################
#                              #
#  DIprint and QIprint Reader  #
#                              #
################################
def Molcas_trans_int(filename, SOC=False, Quadrupole=False, Velocity=False):
    data = []
    
    if SOC:
        if Quadrupole:
            start_line = 'Second-order contribution to the transition strengths (SO states):'
        elif Velocity:
            start_line = 'Velocity transition strengths (SO states):'
        else:
            start_line = 'Dipole transition strengths (SO states):'
    else:
        if Quadrupole:
            start_line = 'Second-order contribution to the transition strengths (spin-free states):'
        elif Velocity:
            start_line = 'Velocity transition strengths (spin-free states):'
        else:
            start_line = 'Dipole transition strengths (spin-free states):'
            
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    table_start = False
    dash_count = 0

    for i, line in enumerate(lines):
        if start_line in line:
            table_start = True
            continue

        if table_start:
            if re.match(r'^\s*-+\s*$', line):
                dash_count += 1
                continue

            if dash_count == 2:
                if re.match(r'^\s*-+\s*$', line):
                    break
                match = re.match(r'^\s*(\d+)\s+(\d+)\s+([0-9.Ee+-]+)', line)
                if match:
                    from_state = int(match.group(1))
                    to_state = int(match.group(2))
                    osc_strength = float(match.group(3))
                    data.append([from_state, to_state, osc_strength])
                    
    if not data:
        print("Transition Data Reading Failed!")
        
    return data

def Molcas_read_int(filename, SOC=False, Quadrupole=False, Velocity=False, Subset=0, GStates = []): 
    eigenvalues = np.array(Molcas_eigenE(filename, SOC))
    transitions = np.array(Molcas_trans_int(filename, SOC, Quadrupole, Velocity))
    
    if GStates:
        transitions = transitions[np.isin(transitions[:,0].real, GStates)]
        data = np.zeros((np.shape(transitions)[0], 4))
    else:
        data = np.zeros((np.shape(transitions)[0], 4))
        
    for i, line in enumerate(transitions):
        data[i][0] = eigenvalues[eigenvalues[:,0].real == line[1]][0][2] - eigenvalues[eigenvalues[:,0].real == line[0]][0][2]
        data[i][1] = line[2]
        data[i][2] = line[0]
        data[i][3] = line[1] - Subset
        
    return data

##########################
#                        #
#  TDRI and TDRC Reader  #
#                        #
##########################
def Molcas_trans_vec(filename, SOC=False):
    data = []

    with open(filename, 'r') as file:
        lines = file.readlines()
    
    table_start = False
    dash_count = 0
    
    if SOC:
        start_line = 'Complex transition dipole vectors (SO states):' 

        for i, line in enumerate(lines):
            if start_line in line:
                table_start = True
                continue

            if table_start:
                if re.match(r'^\s*-+\s*$', line):
                    dash_count += 1
                    continue

                if dash_count == 2:
                    if re.match(r'^\s*-+\s*$', line):
                        break
                    match = re.match(r'^\s*(\d+)\s+(\d+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)', line)
                    if match:
                        from_state = int(match.group(1))
                        to_state = int(match.group(2))
                        ReX = float(match.group(3))
                        ImX = float(match.group(4))
                        ReY = float(match.group(5))
                        ImY = float(match.group(6))
                        ReZ = float(match.group(7))
                        ImZ = float(match.group(8))
                        data.append([from_state, to_state, complex(ReX, ImX), complex(ReY, ImY), complex(ReZ, ImZ)])
                        
    else:
        start_line = 'Dipole transition vectors (spin-free states):'

        for i, line in enumerate(lines):
            if start_line in line:
                table_start = True
                continue

            if table_start:
                if re.match(r'^\s*-+\s*$', line):
                    dash_count += 1
                    continue
                    
                if dash_count == 2:
                    if re.match(r'^\s*-+\s*$', line):
                        break
                    match = re.match(r'^\s*(\d+)\s+(\d+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)', line)
                    if match:
                        from_state = int(match.group(1))
                        to_state = int(match.group(2))
                        X = float(match.group(3))
                        Y = float(match.group(4))
                        Z = float(match.group(5))
                        data.append([from_state, to_state, X, Y, Z])

    if not data:
        print("Transition Data Reading Failed!")
        
    return data

def Molcas_read_vec(filename, SOC=False, Subset=0, GStates = []): 
    eigenvalues = np.array(Molcas_eigenE(filename, SOC))
    transitions = np.array(Molcas_trans_vec(filename, SOC))
    
    if GStates:
        transitions = transitions[np.isin(transitions[:,0].real, GStates)]
        data = np.zeros((np.shape(transitions)[0], 6), dtype=complex)
    else:
        data = np.zeros((np.shape(transitions)[0], 6), dtype=complex)
        
    for i, line in enumerate(transitions):
        data[i][0] = eigenvalues[eigenvalues[:,0].real == line[1]][0][2] - eigenvalues[eigenvalues[:,0].real == line[0]][0][2]
        data[i][1] = line[2]
        data[i][2] = line[3]
        data[i][3] = line[4]
        data[i][4] = line[0]
        data[i][5] = line[1] - Subset
        
    return data

##########################
#                        #
#  MEES and MESO Reader  #
#                        #
##########################
def Molcas_trans_me(filename, SOC=False, mltpl=None, comp=None):
    
    if SOC:
        block_start = "++ Matrix elements over SO states"
    else:
        block_start = "++ Matrix elements"
        
    matrix_start = f"PROPERTY: MLTPL  {mltpl}   COMPONENT:   {comp}"
    
    block_capture = False
    matrix_capture = False
    lines = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == block_start:
                block_capture = True
                continue
            if line.strip() == "--":
                block_capture = False
                continue
            if block_capture and line.strip().startswith(matrix_start):
                matrix_capture = True
                continue
            if matrix_capture and line.strip().startswith("PROPERTY"):
                break
            if matrix_capture:
                lines.append(line)

    data = "\n".join(lines)

    blocks = []
    
    if SOC:
        number_pattern = r"\(\s*([-+]?\d*\.\d+)\s*,\s*([-+]?\d*\.\d+)\s*\)"
        
        data = data.replace("**********", "  0.000000")
        
        for block in data.split("STATE"): 
            rows = []
            for line in block.splitlines():
                if not line.strip():
                    continue
                if re.match(r"^\s*\d+", line):  
                    nums = re.findall(number_pattern, line)
                    if nums:
                        rows.append([complex(float(x), float(y)) for x, y in nums])
            if rows:
                blocks.append(np.array(rows))

    else:
        number_pattern = r"[-+]?\d*\.\d+E[+-]?\d+"
        
        for block in data.split("STATE"): 
            rows = []
            for line in block.splitlines():
                if not line.strip():
                    continue
                if re.match(r"^\s*\d+", line):  
                    nums = re.findall(number_pattern, line)
                    if nums:
                        rows.append([float(x) for x in nums])
            if rows:
                blocks.append(np.array(rows))

    matrix = np.hstack(blocks)

    if matrix.size == 0 :
        print("Transition Data Reading Failed!")
        
    return matrix

def Molcas_read_ten(filename, SOC=False, Mltpl=2, Subset=0, GStates=[], Threshold=0): 
    
    data = []
    
    eigenvalues = np.array(Molcas_eigenE(filename, SOC))
    
    if Mltpl == 1:
        tx = np.array(Molcas_trans_me(filename, SOC, Mltpl, 1))
        ty = np.array(Molcas_trans_me(filename, SOC, Mltpl, 2))
        tz = np.array(Molcas_trans_me(filename, SOC, Mltpl, 3))
        
        Nstates = np.shape(tx)[0]
        Ng = Nstates
        nf = 0
        
        if Subset != 0:
            Ng = Subset
            nf = Subset
        
        for i in range(0, Ng):
            if GStates and i+1 not in GStates:
                continue
            for j in range(nf, Nstates):
                line = np.zeros(6, dtype=complex)
                line[0] = eigenvalues[j][2] - eigenvalues[i][2]
                line[1] = tx[i, j]
                line[2] = ty[i, j]
                line[3] = tz[i, j]
                line[4] = i + 1
                line[5] = j + 1 - Subset
                
                if np.sum(np.abs(line[1:4]) > Threshold):
                    data.append(line)   
        
    if Mltpl == 2:
        txx = np.array(Molcas_trans_me(filename, SOC, Mltpl, 1))
        txy = np.array(Molcas_trans_me(filename, SOC, Mltpl, 2))
        txz = np.array(Molcas_trans_me(filename, SOC, Mltpl, 3))
        tyy = np.array(Molcas_trans_me(filename, SOC, Mltpl, 4))
        tyz = np.array(Molcas_trans_me(filename, SOC, Mltpl, 5))
        tzz = np.array(Molcas_trans_me(filename, SOC, Mltpl, 6))
            
        Nstates = np.shape(txx)[0]
        Ng = Nstates
        nf = 0
        
        if Subset != 0:
            Ng = Subset
            nf = Subset
              
        for i in range(0, Ng):
            if GStates and i+1 not in GStates:
                continue
            for j in range(nf, Nstates):
                line = np.zeros(9, dtype=complex)
                line[0] = eigenvalues[j][2] - eigenvalues[i][2]
                line[1] = txx[i, j]
                line[2] = txy[i, j]
                line[3] = txz[i, j]
                line[4] = tyy[i, j]
                line[5] = tyz[i, j]
                line[6] = tzz[i, j]
                line[7] = i + 1
                line[8] = j + 1 - Subset
                
                if np.sum(np.abs(line[1:7]) > Threshold):
                    data.append(line)   
                    
    return np.array(data)