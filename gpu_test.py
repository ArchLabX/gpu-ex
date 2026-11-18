#!/usr/bin/env python3

import argparse
import time
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check that the job requested a GPU "
            "and that the PyTorch/CUDA environment is loaded."
        )

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    print("GPU:", gpu_name)
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("Matrix size:", args.size)
    print("Iterations:", args.iterations)

    # Create matrices directly on the GPU.
    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)

    # Warm up the GPU.
    for _ in range(5):
        _ = torch.matmul(a, b)

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(args.iterations):
        c = torch.matmul(a, b)

    # CUDA operations are asynchronous, so synchronize before timing.
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    average_time = elapsed / args.iterations

    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Average matrix multiplication time: {average_time:.4f} seconds")
    print("Result checksum:", c[0, 0].item())


if __name__ == "__main__":
    main()