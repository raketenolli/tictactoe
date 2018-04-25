import numpy as np
import pickle

DEBUG = False
DEBUG_Q = False
n_games = 10000

gamma = 0.95

_win_combos = np.array([[0,1,2],
                        [3,4,5],
                        [6,7,8],
                        [0,3,6],
                        [1,4,7],
                        [2,5,8],
                        [0,4,8],
                        [2,4,6]])

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def initialize_Q_t():
    Q = {}
    t = {}
    ii = cartesian([[0,1,2]] * 9)
    for i in ii:
        ti = tuple(i)
        Q[ti] = [0.0] * 9
        t[ti] = [1] * 9
    return Q, t

def choose_random_space(s):
    es = np.nonzero(np.array(s)==0)[0]
    if len(es) == 0:
        return -1
    return np.random.choice(es)

def invert_key(s):
    sa = np.array(s)
    si = np.zeros(sa.shape, dtype=np.uint8)
    si[sa==1] = 2
    si[sa==2] = 1
    return tuple(si)

def choose_best_move(s, **kwargs):
    explore_rate = kwargs.get("explore_rate", 0.25)
    invert = kwargs.get("invert", False)
    s = tuple(s)
    sa = np.array(s)
    if len(np.nonzero(sa==0)[0]) == 0: return -1
    if np.random.rand() > explore_rate:
        if invert: k = invert_key(s)
        else: k = s
        Qa = np.array(Q[k])
        Qa[sa!=0] = -1.0
        max_Q = max(Qa)
        es = np.nonzero(Qa==max_Q)[0]
        return np.random.choice(es)
    else:
        return choose_random_space(s)

def is_game_over(s):
    for wc in _win_combos:
        if np.all(np.array(s)[wc]==1):
            return 1
        if np.all(np.array(s)[wc]==2):
            return 2
    return 0

# uncomment if Q learning should be restarted
Q, t = initialize_Q_t()

# # uncomment if Q and t values should be initialized with pickled values from last run
# with open("Q.pickle", "rb") as fh:
    # Q = pickle.load(fh)
# with open("t.pickle", "rb") as fh:
    # t = pickle.load(fh)

for g in range(n_games):

    # initialize empty board
    # 0 0 0
    # 0 0 0
    # 0 0 0
    # in row-major order
    b = [0,0,0,0,0,0,0,0,0]
    if DEBUG: print("new game")

    # play with learning AI as second mover on odd numbered games
    if (g % 2 == 0):
        o = choose_best_move(b, explore_rate=0.25, invert=False)
    
    game_over = False

    while not game_over:
        s = tuple(b.copy()) # retain current board state for future reference
        
        if DEBUG: print(s, Q[s])
        
        a = choose_best_move(b, explore_rate=0.25, invert=False)
        if a == -1:
            game_over = True
        else:
            b[a] = 1
            if DEBUG: print(np.array(b).reshape(3,3))
            
            game_result = is_game_over(b)
            
            if game_result == 1:
                Q[s][a] = Q[s][a] + 1 / t[s][a] * (1.0 - Q[s][a])
                t[s][a] += 1
                if DEBUG: print("1 won")
                if DEBUG: print(s, Q[s], t[s])
                game_over = True
                
            else:
                o = choose_best_move(b, explore_rate=0.25, invert=True)
                b[o] = 2
                if DEBUG: print(np.array(b).reshape(3,3))
                
                game_result = is_game_over(b)
                
                if game_result == 2:
                    Q[s][a] = Q[s][a] + 1 / t[s][a] * (-1.0 - Q[s][a])
                    t[s][a] += 1
                    if DEBUG: print("2 won")
                    if DEBUG: print(s, Q[s], t[s])
                    game_over = True
                    
                else:
                    s_prime = tuple(b.copy())
                    Q[s][a] = Q[s][a] + 1 / t[s][a] * (gamma*max(Q[s_prime]) - Q[s][a])
                    t[s][a] += 1

    if DEBUG_Q:
        for k, q in Q.items():
            if not (q==[0.0]*9): print(k, q)

print(Q[(0,0,0,0,0,0,0,0,0)])
print(t[(0,0,0,0,0,0,0,0,0)])

with open("Q.pickle", "wb") as fh:
    pickle.dump(Q, fh)
with open("t.pickle", "wb") as fh:
    pickle.dump(t, fh)
