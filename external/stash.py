# Note: Numba target needs to be set here and cannot be set after importing the library
NUMBA_TARGET = "parallel"   # Available targets: "cpu", "parallel", and "cuda"

@nb.jit(nopython=True, parallel=True)
def example_func(a, b):
    return a**2 + b**2

@nb.jit(nopython=True, parallel=True)
def compute_direction_numba(trk_list):
    # out = np.zeros(length_of_output)
    out = [None]*len(trk_list)
    for i in nb.prange(len(trk_list)):
        # independent and parallel loop
        # out[i] = compute_direction(trk_list[i])
        trk = np.transpose(np.asarray(trk_list[i]))
        # numb_points = trk.shape[0]
        direction_rescale = [None]*trk.shape[0]
        for j in range(trk.shape[0] - 1):
            direction = trk[j + 1, :] - trk[j, :]
            direction = direction / np.linalg.norm(direction)
            # direction_rescale.append([int(255 * abs(s)) for s in direction])
            direction_rescale.append([int(255 * abs(s)) for s in direction])

        # repeat color for last point
        direction_rescale.append([int(255 * abs(s)) for s in direction])
        out[i] = direction_rescale
    return out


@nb.jit(nopython=True, parallel=True)
def normalize_numba(x):
    ret = np.empty_like(x)
    for i in nb.prange(x.shape[0]):
        acc = 0.0
        for j in range(x.shape[1]):
            acc += x[i, j]**2
    norm = np.sqrt(acc)
    for j in range(x.shape[1]):
        ret[i, j] = x[i, j] / norm

    return ret


def numba3(vec_obj, vec_ps, cos_maxsep):
    nps = len(vec_ps)
    nobj = len(vec_obj)
    out = np.zeros(nobj, bool)
    numba3_helper(vec_obj, vec_ps, cos_maxsep, out, nps, nobj)
    return np.flatnonzero(out)

@nb.jit(nopython=True)
def numba3_helper(vec_obj, vec_ps, cos_maxsep, out, nps, nobj):
    for i in range(nobj):
        for j in range(nps):
            cos = (vec_obj[i,0]*vec_ps[j,0] +
                   vec_obj[i,1]*vec_ps[j,1] +
                   vec_obj[i,2]*vec_ps[j,2])
            if cos > cos_maxsep:
                out[i] = True
                break
    return out


@nb.guvectorize([nb.void(nb.double[:], nb.double[:], nb.double[:])],
    '(n),(n)->(n)', target=NUMBA_TARGET, nopython=True)
def _gprocess_point(a, b, out_b):
    """Substracts 'b' from 'a' and stores result in 'out_V'. Then takes cross product of 'c' and 'out_V' and stores result in 'out_p'.

    Parameters
    ----------
    a, b ,c: np.ndarray
        One-dimensional arrays to process.

    out_b : np.ndarray
        Output where to accumulate the results
    """
    # Substract
    a1, a2, a3 = a[0], a[1], a[2]
    b1, b2, b3 = b[0], b[1], b[2]
    V1 = a1 - b1
    V2 = a2 - b2
    V3 = a3 - b3

    # Length of V
    v_norm = math.sqrt(V1*V1 + V2*V2 + V3*V3)

    # Cross product
    if v_norm != 0:
        p1 = V1/v_norm
        p2 = V2/v_norm
        p3 = V3/v_norm

        # Store result in out_b
        #np.add(b, p, out=b)
        out_b[0] = 255*p1
        out_b[1] = 255*p2
        out_b[2] = 255*p3


@nb.jit()
def _gprocess_points(xyz, wire, dwire, out_b):
    """Processes 'xyz' coordinates and calculates b-field due to current in wire 'w', 'dw'. Stores outputs to 'out_V', 'out_p', and 'out_b'.

    Parameters
    ----------
    xyz : np.ndarray
        One-dimensional array to process.

    wire : np.ndarray
        Wire coordinates as 3-dimensional vectors.

    dwire : np.ndarray
        Wire length vectors

    out_b : np.ndarray
        Output where to accumulate the results
    """
    for i in range(len(wire)):
        w = wire[i]
        dw = dwire[i]
        _gprocess_point(xyz, w, dw, out_b)   # V = xyz - w; p = dw x V

# def calculate_points():
#
#     B = np.zeros_like(xyz, dtype=np.double)
#
#     return B

#condition_tut.py

import random, time
from threading import Condition, Thread

"""
'condition' variable will be used to represent the availability of a produced
item.
"""

condition = Condition()

box = []

def producer(box, nitems):
    for i in range(nitems):
        time.sleep(random.randrange(2, 5))  # Sleeps for some time.
        condition.acquire()
        num = random.randint(1, 10)
        box.append(num)  # Puts an item into box for consumption.
        condition.notify()  # Notifies the consumer about the availability.
        print("Produced:", num)
        condition.release()

def consumer(box, nitems):
    for i in range(nitems):
        condition.acquire()
        condition.wait()  # Blocks until an item is available for consumption.
        print("%s: Acquired: %s" % (time.ctime(), box.pop()))
        condition.release()

threads = []

"""
'nloops' is the number of times an item will be produced and
consumed.
"""

nloops = random.randrange(3, 6)
for func in [producer, consumer]:
    threads.append(Thread(target=func, args=(box, nloops)))
    threads[-1].start()  # Starts the thread.

for thread in threads:
    """Waits for the threads to complete before moving on
       with the main script.
    """
    thread.join()
print("All done.")