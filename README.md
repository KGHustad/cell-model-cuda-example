
## Building
```sh
build_dir=build
mkdir -p ${build_dir}
pushd ${build_dir}
cmake ../src
cmake --build .
```

## Usage
```
$ ./TP06 -h
Usage: cell_model_cuda_example [-h|--help] [-V|--version]
         [-NDOUBLE|--num_cells=DOUBLE] [-tINT|--num_timesteps=INT]
         [--dt=DOUBLE] [-iINT|--device_id=INT] [-sENUM|--scheme=ENUM]

  -h, --help               Print help and exit
  -V, --version            Print version and exit
  -N, --num_cells=DOUBLE   number of cells (independent instances of the ODE
                             system)  (default=`1E7')
  -t, --num_timesteps=INT  number of time steps  (default=`100')
      --dt=DOUBLE          time step (in ms) used to solve the cell model
                             (default=`0.001')
  -i, --device_id=INT      CUDA device to use  (default=`0')
  -s, --scheme=ENUM        numerical scheme used to solve the cell model
                             (possible values="FE", "RL", "GRL1"
                             default=`FE')
```

## Performance results
These performance results were obtained using a single NVIDIA A100-SXM4-80GB GPU from a DGX A100 system.
```
[kghustad@g002 23:38:18 build_a100q (master)*]$ nvidia-smi
Fri Jul 29 23:40:15 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.48.07    Driver Version: 515.48.07    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:4C:00.0 Off |                    0 |
| N/A   32C    P0    64W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Note that cell steps / second is a throughput metric defined as `number_of_cells * number_of_time_steps / time_elapsed`.

```
[kghustad@g002 23:37:54 build_a100q (master)*]$ ./TP06 -s FE
Solving with the FE scheme
Solved 100 timesteps with 10000000 cells in 0.584167 seconds.
Throughput: 1.71184e+09 cell steps / second.
Final V values are in the interval [-8.54229956918900655e+01, -8.54229956918900655e+01]
```
```
[kghustad@g002 23:38:10 build_a100q (master)*]$ ./TP06 -s RL
Solving with the RL scheme
Solved 100 timesteps with 10000000 cells in 0.960482 seconds.
Throughput: 1.04114e+09 cell steps / second.
Final V values are in the interval [-8.54229956921292626e+01, -8.54229956921292626e+01]
```
```
[kghustad@g002 23:38:14 build_a100q (master)*]$ ./TP06 -s GRL1
Solving with the GRL1 scheme
Solved 100 timesteps with 10000000 cells in 0.945808 seconds.
Throughput: 1.0573e+09 cell steps / second.
Final V values are in the interval [-8.54229956921292626e+01, -8.54229956921292626e+01]
```
