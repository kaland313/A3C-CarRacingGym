# from mpi4py import MPI
# import os
# import subprocess
# import sys
# import time
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
#
# def mpi_fork(n):
#     """Re-launches the current script with workers
#     Returns "parent" for original parent, "child" for MPI children
#     (from https://github.com/garymcintire/mpi_util/)
#     """
#     if n<=1:
#         return "child"
#     if os.getenv("IN_MPI") is None:
#         env = os.environ.copy()
#         env.update(
#             MKL_NUM_THREADS="1",
#             OMP_NUM_THREADS="1",
#             IN_MPI="1"
#         )
#         cmd = ["mpirun", "-np", str(n), sys.executable] + ['-u'] + sys.argv
#         print(cmd)
#         subprocess.check_call(cmd, env=env)
#         # subprocess.check_call(["/usr/bin/mpirun", "-np", str(n), '-mca', 'coll_tuned_bcast_algorithm', '0', sys.executable] +['-u']+ sys.argv, env=env)       # this mpirun is 1/3 the speed of the one above
#         return "parent"
#     else:
#         global nworkers, rank
#         nworkers = MPI.COMM_WORLD.Get_size()
#         rank = MPI.COMM_WORLD.Get_rank()
#         print('assigning the rank and nworkers', nworkers, rank)
#         return "child"
#
#
# if rank == 0:
#     mpi_fork(4)
#     while(True):
#         data = comm.recv(source=MPI.ANY_SOURCE, tag=13)
#         print("M: Sending data to ", data)
#         comm.send(True, dest=int(data), tag=14)
#         data2 = comm.recv(source=MPI.ANY_SOURCE, tag=15)
#         print("M: 2nd message received from: ", data2)
# else:
#     print("S", rank, ": Sending my rank to master")
#     comm.send(rank, dest=0, tag=13)
#     data = comm.recv(source=0, tag=14)
#     print("S", rank, ": Master reply received")
#     time.sleep(5)
#     comm.send(rank, dest=0, tag=15)

#
# class A:
#     def __init__(self, b):
#         self.b = b
#
# class opt:
#     def __init__(self,str):
#         self.str = str


# Referencing test
# opt1 = opt("1")
# print(opt1.str)
# x2 = A(opt1)
# print(x2.b.str)
# x2.b.str = "2"
# print(opt1.str)

# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
# if rank == 0:
#     data = {'a': 7, 'b': 3.14}
#     req = comm.isend(data, dest=1, tag=11)
#     print(req)
#     req.wait()
# elif rank == 1:
#     req = comm.irecv(source=0, tag=11)
#     data = req.wait()
#     print(data)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'key1' : [7, 2.72, 2+3j],
            'key2' : ( 'abc', 'xyz')}
    data = comm.bcast(data, root=0)
else:
    data = None
    data = comm.bcast(None, root=0)
    print(data)
