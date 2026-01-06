import sys
import time
import math
import array
from mpi4py import MPI

def get_system_data(n, start_index, count):
    mat = []
    rhs = [0.0] * count
    
    for i in range(count):
        global_idx = start_index + i
        row = [0.0] * n
        
        for j in range(n):
            dist = abs(global_idx - j)
            if dist <= 20:
                val = -(2.0 ** (-(2.0**dist)))
                row[j] = val
        
        row[global_idx] = 1.0
        rhs[i] = 1.0
        mat.append(row)
        
    return mat, rhs

def check_performance(duration, size):
    ref_table = {1: 1700, 2: 858, 4: 450, 8: 269, 16: 163}
    
    if size in ref_table:
        ref_time = ref_table[size]
        if duration < ref_time:
            print(f"speed check: faster than table reference ({ref_time}s).")
        else:
            print(f"speed check: slower than table reference ({ref_time}s).")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = 10000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    if rank == 0:
        print(f"node {rank} ready: n={n}, cores={size}")

    base_rows = n // size
    extra = n % size
    
    if rank < extra:
        local_rows = base_rows + 1
        offset = rank * local_rows
    else:
        local_rows = base_rows
        offset = rank * local_rows + extra

    t0 = time.perf_counter()
    
    local_a, local_b = get_system_data(n, offset, local_rows)
    local_x = [0.0] * local_rows
    
    comm.Barrier()
    t1 = time.perf_counter()
    
    if rank == 0:
        print(f"setup time: {t1 - t0:.4f}s")

    global_buffer = array.array('d', [0.0] * n)
    local_buffer = array.array('d', [0.0] * local_rows)

    counts = comm.allgather(local_rows)
    displacements = []
    current_disp = 0
    for c in counts:
        displacements.append(current_disp)
        current_disp += c

    max_iter = 1000
    tol = 1e-6

    local_buffer[:] = array.array('d', local_x)
    comm.Allgatherv([local_buffer, MPI.DOUBLE], [global_buffer, counts, displacements, MPI.DOUBLE])
    full_x = list(global_buffer)

    sum_sq_diff = 0.0
    for i in range(local_rows):
        val = local_b[i]
        row = local_a[i]
        for j in range(n):
            val -= row[j] * full_x[j]
        sum_sq_diff += val * val

    total_sq_diff = comm.allreduce(sum_sq_diff, op=MPI.SUM)
    initial_norm = math.sqrt(total_sq_diff)

    t_solve_start = time.perf_counter()
    new_x = [0.0] * local_rows

    for k in range(max_iter):
        for i in range(local_rows):
            global_idx = offset + i
            sigma = local_b[i]
            row = local_a[i]
            
            for j in range(n):
                if j != global_idx:
                    sigma -= row[j] * full_x[j]
            
            new_x[i] = sigma

        local_x[:] = new_x[:]

        local_buffer[:] = array.array('d', local_x)
        comm.Allgatherv([local_buffer, MPI.DOUBLE], [global_buffer, counts, displacements, MPI.DOUBLE])
        full_x = list(global_buffer)

        local_err = 0.0
        for i in range(local_rows):
            check_val = local_b[i]
            row = local_a[i]
            for j in range(n):
                check_val -= row[j] * full_x[j]
            local_err += check_val * check_val
            
        global_err = comm.allreduce(local_err, op=MPI.SUM)
        current_norm = math.sqrt(global_err)

        if current_norm < tol * initial_norm:
            break

    t_final = time.perf_counter()
    elapsed = t_final - t_solve_start
    
    if rank == 0:
        print(f"solver time: {elapsed:.4f}s")
        check_performance(elapsed, size)

if __name__ == "__main__":
    main()