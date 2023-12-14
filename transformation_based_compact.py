# 0: no, 1: control, 2: not
#    編號 0  1  2  3  4
# ex.    [2, 1, 0, 0, 1] 代表qbit編號1、4作control，對qbit編號0施加NOT運算

def compact(gates):
    ctrl = gates[0][1:]
    not_pos, new_gates = [gates[0][0]], []
    for i in range(1, len(gates)):
        if gates[i][1:] == ctrl:
            not_pos.append(gates[i][0])
        else:
            for np in not_pos:
                ctrl[np] = 2
            new_gates.append(ctrl)
            ctrl = gates[i][1:]
            not_pos = [gates[i][0]]
    for np in not_pos:
        ctrl[np] = 2
    new_gates.append(ctrl)
    return new_gates

def output2gates_basic(n, Y):
    gates = []
    for X in range(1 << n):
        if X == Y[X]:
            continue
        new_gates = []
        for i in range(n):
            if X & (1 << i) > 0 and Y[X] & (1 << i) == 0:
                new_gates.append([i, *map(int, reversed(bin(Y[X])[2:].zfill(n)))])
        for i in range(n):
            if X & (1 << i) == 0 and Y[X] & (1 << i) > 0:
                new_gates.append([i, *map(int, reversed(bin(X)[2:].zfill(n)))])
        gates = [*new_gates[::-1], *gates]
        for gate in new_gates:
            ctrl = int(''.join(map(str, gate[:0:-1])), 2) & (((1 << n) - 1) ^ (1 << gate[0]))
            for x in range(X, 1 << n):
                if ctrl & Y[x] == ctrl:  # apply NOT
                    Y[x] ^= 1 << gate[0]
    return compact(gates)

# ex.
# > output2gates_basic(3, [1, 6, 5, 3, 4, 0, 7, 2])
# [[1, 2, 1], [0, 2, 1], [1, 1, 2], [2, 1, 1], [0, 1, 2], [0, 2, 1], [1, 2, 2], [2, 0, 0]]

def output2gates_bidirectional(n, Y):
    gates_forward, gates_backward = [], []
    for X in range(1 << n):
        if X == Y[X]:
            continue
        new_gates = []
        if bin(X ^ Y[X]).count('1') <= bin(Y.index(X) ^ X).count('1'):  # 從後面往前加gate
            for i in range(n):
                if X & (1 << i) > 0 and Y[X] & (1 << i) == 0:
                    new_gates.append([i, *map(int, reversed(bin(Y[X])[2:].zfill(n)))])
            for i in range(n):
                if X & (1 << i) == 0 and Y[X] & (1 << i) > 0:
                    new_gates.append([i, *map(int, reversed(bin(X)[2:].zfill(n)))])
            gates_backward += new_gates
            for gate in new_gates:
                ctrl = int(''.join(map(str, gate[:0:-1])), 2) & (((1 << n) - 1) ^ (1 << gate[0]))
                for x in range(X, 1 << n):
                    if ctrl & Y[x] == ctrl:  # apply NOT
                        Y[x] ^=  1 << gate[0]
        else:  # 從前面往後加gate
            for i in range(n):
                if Y.index(X) & (1 << i) == 0 and X & (1 << i) > 0:
                    new_gates.append([i, *map(int, reversed(bin(Y.index(X))[2:].zfill(n)))])
            for i in range(n):
                if Y.index(X) & (1 << i) > 0 and X & (1 << i) == 0:
                    new_gates.append([i, *map(int, reversed(bin(X)[2:].zfill(n)))])
            gates_forward += new_gates
            for gate in new_gates:
                ctrl = int(''.join(map(str, gate[:0:-1])), 2) & (((1 << n) - 1) ^ (1 << gate[0]))
                Y_old = Y.copy()
                for x in range(X, 1 << n):
                    if ctrl & Y_old.index(x) == ctrl:  # apply NOT
                        Y[Y_old.index(x) ^ (1 << gate[0])] = x
    return compact(gates_forward + gates_backward[::-1])

# ex.
# > output2gates_bidirectional(3, [1, 6, 5, 3, 4, 0, 7, 2])
# [[1, 0, 2], [0, 2, 1], [1, 2, 1], [2, 1, 1], [1, 1, 2], [0, 1, 2], [0, 2, 1], [2, 0, 0]]
# > output2gates_bidirectional(4, [2, 4, 3, 10, 5, 15, 7, 13, 11, 14, 1, 0, 9, 6, 12, 8])
# [[2, 1, 0, 0], [1, 2, 0, 0], [0, 1, 0, 2], [1, 2, 0, 1], [1, 1, 0, 2], [2, 1, 1, 1], [2, 0, 1, 1], [2, 1, 0, 1], [1, 2, 0, 1], [2, 1, 0, 1], [2, 0, 0, 1], [1, 1, 1, 2], [1, 1, 2, 1], [0, 1, 1, 2], [0, 1, 2, 1], [1, 2, 1, 0], [2, 1, 1, 0], [2, 2, 1, 0], [0, 2, 0, 0]]

def gates2output(n, gates):
    Y = [*range(1 << n)]
    for gate in gates:
        ctrl = int(''.join(map(str, gate[::-1])).replace('2', '0'), 2)
        Y = [y ^ int(''.join(map(str, gate[::-1])).replace('1', '0').replace('2', '1'), 2) if ctrl & y == ctrl else y for y in Y]
    return Y
