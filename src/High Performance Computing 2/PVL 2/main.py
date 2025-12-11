from mpi4py import MPI

message = 0.0

def task_a(comm, rank):
    if rank == 0:
        print("--- Starting Task A (Logic Check) ---")

    comm.Barrier()
    
    if rank == 0:
        for i in range(50000):
            comm.send(message, dest=1, tag=10)
            comm.recv(source=1, tag=10)
        print(f"Task A: Rank 0 successfully completed 50000.")
        
    elif rank == 1:
        for i in range(50000):
            msg = comm.recv(source=0, tag=10)
            comm.send(msg, dest=0, tag=10)

    comm.Barrier()
    if rank == 0:
        print("-------------------------------------\n")


def task_b(comm, rank):
    if rank == 0:
        print("--- Starting Task B (Timing & Measurement) ---")

    comm.Barrier()
    t_start = MPI.Wtime()
    
    if rank == 0:
        for i in range(50000):
            comm.send(message, dest=1, tag=20)
            comm.recv(source=1, tag=20)
    elif rank == 1:
        for i in range(50000):
            msg = comm.recv(source=0, tag=20)
            comm.send(msg, dest=0, tag=20)

    t_end = MPI.Wtime()

    if rank == 0:
            duration = t_end - t_start
            total_msgs = 2 * 50000
            
            latency_sec = duration / total_msgs
            latency_ms = latency_sec * 1e3
            latency_micros = latency_sec * 1e6
            
            print(f"Task B: Duration for {total_msgs} messages: {duration:.6f} s")
            print(f"Task B: Single message latency: {latency_ms:.6f} milliseconds")
            print(f"Task B: Single message latency: {latency_micros:.6f} microseconds")
            print("-------------------------------------")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            raise ValueError("Error: Run this with exactly 2 ranks (e.g., mpiexec -n 2 ...)")
        return

    task_a(comm, rank)
    task_b(comm, rank)


if __name__ == "__main__":
    main()
