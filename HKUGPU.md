User account can be applied at  https://intranet.cs.hku.hk/gpufarm3_acct/. After your account is created, you may **login** the gateway node gpu3gate1.cs.hku.hk with SSH:
ssh <your_username>@gpu3gate1.cs.hku.hk 

After logging on the gateway node, a GPU session can be started with srun, e.g.,
srun --gres=gpu:1 --mail-type=ALL --pty bash 
The default SLURM queue (debug) allocates **RTX4090** GPUs. 4 CPU cores and 96GB system RAM is allocated with each GPU. 

To have a session with **2 GPUs**:
srun --nodes=1 --gres=gpu:2 --mail-type=ALL --pty bash
By default, each user account can request up to 4 GPUs concurrently. The limit can be raised on request.

Specifying the **longer time limit**
A job will be terminated when its time limited is reached. Use '-t' to specify a longer time limit than the default. For example, to have a time limit of 12 hours:
srun --nodes=1 --gres=gpu:2 -t 12:00:00 --mail-type=ALL --pty bash 

Running a Session with one **H100 or H800** GPU
To get a session with a H100/H800 GPU, use the q-hgpu-small partition by adding '-p q-hgpu-small' in srun or sbatch, e.g.,
srun -p q-hgpu-small --gres=gpu:1 --mail-type=ALL --pty bash
12 CPU cores and 240GB system RAM is allocated with each GPU. Either a H800 or H100 will be allocated depending on availability.

Running a Session with **2 H800** GPUs 
You may specify the GPU model (h100 or h800) you need as a command line option, e.g., 
srun -p q-hgpu-small --nodes=1 --gres=gpu:h800:2 --mail-type=ALL --pty bash