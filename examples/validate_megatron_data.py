import numpy as np
import struct
import os

# Define your HDR_MAGIC and dtypes here
HDR_MAGIC = b"MMIDIDX\x00\x00"  # Example magic number
dtypes = {0: np.float32, 1: np.int64, 4: np.int32}  # Example dtype mapping


def validate_mmap_data(sizes, pointers, doc_idx):
    # Check that sizes and pointers have the same length
    if len(sizes) != len(pointers):
        raise ValueError(f"Inconsistent lengths: sizes ({len(sizes)}) and pointers ({len(pointers)}) must match.")

    # Check that doc_idx has at least two entries (start and end)
    if len(doc_idx) < 2:
        raise ValueError(f"Document index must have at least two entries, but got {len(doc_idx)}.")

    # Check that the end of the last document index is within the bounds of the pointers
    if doc_idx[-1] != len(pointers):
        raise ValueError(f"Last document index ({doc_idx[-1]}) must equal the number of pointers ({len(pointers)}).")

    # Additional checks can be added based on the specific format and requirements
    print("Validation successful: Sizes, pointers, and document indices are consistent.")

def read_and_validate_mmap_file(path, skip_warmup=False):
    with open(path, 'rb') as stream:
        magic_test = stream.read(9)
        assert HDR_MAGIC == magic_test, (
            'Index file doesn\'t match expected format. '
            'Make sure that --dataset-impl is configured properly.'
        )
        version = struct.unpack('<Q', stream.read(8))
        assert (1,) == version

        dtype_code, = struct.unpack('<B', stream.read(1))
        dtype = dtypes[dtype_code]
        assert dtype_code == 4, "Only int32 is supported"

        dtype_size = dtype().itemsize
        assert dtype_size == 4, "Only int32 is supported"

        len_data = struct.unpack('<Q', stream.read(8))[0]
        doc_count = struct.unpack('<Q', stream.read(8))[0]

        print(f"len_data: {len_data}, doc_count: {doc_count}")

        offset = stream.tell()

    if not skip_warmup:
        print("    warming up index mmap file...")
        # Implement _warmup_mmap_file function or remove this if not needed

    bin_buffer_mmap = np.memmap(path, mode='r', order='C')
    bin_buffer = memoryview(bin_buffer_mmap)
    print("    reading sizes...")
    sizes = np.frombuffer(
        bin_buffer,
        dtype=np.int32,
        count=len_data,
        offset=offset
    )
    print(f"sizes: {sizes}")
    print("    reading pointers...")
    pointers = np.frombuffer(bin_buffer, dtype=np.int64, count=len_data,
                             offset=offset + sizes.nbytes)
    print(f"pointers: {pointers}")
    print("    reading document index...")
    doc_idx = np.frombuffer(bin_buffer, dtype=np.int64, count=doc_count,
                            offset=offset + sizes.nbytes + pointers.nbytes)
    print(f"doc_idx: {doc_idx}")
    # Validate the data read from the memory-mapped file
    validate_mmap_data(sizes, pointers, doc_idx)

    # Return the data structures if needed
    return sizes, pointers, doc_idx

# Example call
if __name__ == "__main__":
    print(os.getcwd())
    path = "data/output_folder/bin/processed_processed_sample_10.memmap" # Replace with your actual file path
    read_and_validate_mmap_file(path)
