gpu-Ex
======================
### An example workload for testing GPU Clusters

This project runs a simple PyTorch matrix-multiplication benchmark on one
GPU using Slurm.

Files
-----

gpu_test.py
    Creates matrices on the GPU and performs repeated matrix multiplication.

run_gpu_test.sbatch
    Slurm batch script that requests one GPU and runs gpu_test.py.

Setup
-----

1. Connect to Unity.

2. Create the project directory:

       git clone https://github.com/ArchLabX/gpu-ex

4. Edit run_gpu_test.sbatch and replace:

       YOUR_ACCOUNT

   with the account associated with your allocation.

5. Inspect available software modules:

       module avail

6. If "pytorch" is not the correct module name, update the module load line
   in run_gpu_test.sbatch.

Submit the job
--------------

From the project directory:

       sbatch run_gpu_test.sbatch

Check the job
-------------

       squeue --me

After the job completes:

       cat logs/gpu-test-JOBID.out
       cat logs/gpu-test-JOBID.err

Replace JOBID with the job ID printed by sbatch.

Cancel the job
--------------

       scancel JOBID

Expected output
---------------

The output should include information similar to:

       CUDA available: True
       GPU: <GPU model>
       CUDA version used by PyTorch: <version>
       Total time: ...
       Average matrix multiplication time: ...
       Result checksum: ...

Troubleshooting
---------------

If CUDA available is False:

1. Confirm that the job requested a GPU using:

       #SBATCH --gres=gpu:1

2. Confirm that nvidia-smi succeeds inside the job.

3. Check the available PyTorch module:

       module avail

4. Confirm that PyTorch was installed with CUDA support.

If the job remains pending, inspect the reason with:

       scontrol show job JOBID

You can also inspect available GPUs and cluster usage using the Unity
Slurm helper commands documented by the Research Computing group.