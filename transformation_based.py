# gate的表示法:
#    編號 43210
# ex.    '10012' 代表qbit編號1、4作control，對qbit編號0施加NOT運算

def output2gates_basic(n, Y):
    gates = []
    g = lambda n, c: ''.join('2' if i == n else a for i, a in enumerate(c[::-1]))[::-1]
    for X in range(1 << n):
        if X == Y[X]:
            continue
        new_gates = [g(i, bin(Y[X])[2:].zfill(n)) for i in range(n) if X & (1 << i) > 0 and Y[X] & (1 << i) == 0] \
            + [g(i, bin(X)[2:].zfill(n)) for i in range(n) if X & (1 << i) == 0 and Y[X] & (1 << i) > 0]
        gates += new_gates
        for gate in new_gates:
            ctrl = int(gate.replace('2', '0'), 2)
            for x in range(X, 1 << n):
                if ctrl & Y[x] == ctrl:  # apply NOT
                    Y[x] ^= int(gate.replace('1', '0').replace('2', '1'), 2)
    return gates[::-1]

# ex.
# > output2gates_basic(3, [1, 6, 5, 3, 4, 0, 7, 2])
# ['121', '120', '211', '112', '210', '120', '201', '021', '002']

def output2gates_bidirectional(n, Y):
    gates_forward, gates_backward = [], []
    g = lambda n, c: ''.join('2' if i == n else a for i, a in enumerate(c[::-1]))[::-1]
    for X in range(1 << n):
        if X == Y[X]:
            continue
        new_gates = []
        if bin(X ^ Y[X]).count('1') <= bin(Y.index(X) ^ X).count('1'):  # 從後面往前加gate
            new_gates += [g(i, bin(Y[X])[2:].zfill(n)) for i in range(n) if X & (1 << i) > 0 and Y[X] & (1 << i) == 0]
            new_gates += [g(i, bin(X)[2:].zfill(n)) for i in range(n) if X & (1 << i) == 0 and Y[X] & (1 << i) > 0]
            gates_backward += new_gates
            for gate in new_gates:
                ctrl = int(gate.replace('2', '0'), 2)
                for x in range(X, 1 << n):
                    if ctrl & Y[x] == ctrl:  # apply NOT
                        Y[x] ^= int(gate.replace('1', '0').replace('2', '1'), 2)
        else:  # 從前面往後加gate
            new_gates += [g(i, bin(Y.index(X))[2:].zfill(n)) for i in range(n) if Y.index(X) & (1 << i) == 0 and X & (1 << i) > 0]
            new_gates += [g(i, bin(X)[2:].zfill(n)) for i in range(n) if Y.index(X) & (1 << i) > 0 and X & (1 << i) == 0]
            gates_forward += new_gates
            for gate in new_gates:
                ctrl = int(gate.replace('2', '0'), 2)
                Y_old = Y.copy()
                for x in range(X, 1 << n):
                    if ctrl & Y_old.index(x) == ctrl:  # apply NOT
                        Y[Y_old.index(x) ^ int(gate.replace('1', '0').replace('2', '1'), 2)] = x
    return gates_forward + gates_backward[::-1]

# ex.
# > output2gates_bidirectional(3, [1, 6, 5, 3, 4, 0, 7, 2])
# ['201', '120', '121', '112', '211', '210', '120', '002']
# > output2gates_bidirectional(4, [2, 4, 3, 10, 5, 15, 7, 13, 11, 14, 1, 0, 9, 6, 12, 8])
# ['0012', '0021', '2010', '1021', '2011', '1112', '1102', '1012', '1021', '1012', '1002', '2111', '1211', '2110', '1210', '0121', '0112', '0120', '0102', '0020']

def gates2output(n, gates):
    Y = [*range(1 << n)]
    for gate in gates:
        ctrl = int(gate.replace('2', '0'), 2)
        Y = [y ^ int(gate.replace('1', '0').replace('2', '1'), 2) if ctrl & y == ctrl else y for y in Y]
    return Y
