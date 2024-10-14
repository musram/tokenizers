import numpy as np
import struct
import os

# Define your HDR_MAGIC and dtypes here
HDR_MAGIC = b"PACKED"  # Example magic number
dtypes = {0: np.float32, 1: np.int64, 4: np.int32}  # Example dtype mapping


def read_memmap_file(path):
    with open(path, "rb") as f:
        # Read and validate the header
        magic = f.read(len(HDR_MAGIC))
        assert magic == HDR_MAGIC, "File doesn't match expected format."

        version = struct.unpack("<Q", f.read(8))
        assert version == (1,), "Unsupported version."

        (chunk_size,) = struct.unpack("<Q", f.read(8))
        (dtype_code,) = struct.unpack("<B", f.read(1))
        print(dtype_code)
        dtype = dtypes[dtype_code]

        # Print header information
        print(f"Chunk Size: {chunk_size}, Data Type: {dtype}")

        print(len(HDR_MAGIC) + 17)
        # Create a memory-mapped array
        mmap_array = np.memmap(
            path, mode="r", dtype=dtype, offset=len(HDR_MAGIC) + 18
        )  # Adjust offset as needed

        print(mmap_array.shape)

        # Iterate over mmap_array in multiples of 2048
        for i in range(0, len(mmap_array), 2048):
            end = min(i + 2048, len(mmap_array))
            print(f"Chunk {i // 2048 + 1}:")
            print(mmap_array[i:end].shape)
            assert mmap_array[i:end].shape == (2048,)
            print()  # Add a blank line between chunks for readability
            print(mmap_array[i:end])
        
        return mmap_array


# Usage
if __name__ == "__main__":
    filepath = "processed_processed_sample_10.memmap"  # Replace with your file path
    print(os.getcwd())
    data = read_memmap_file(filepath)

    # Display the data
    print(data)
