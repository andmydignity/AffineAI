"""
High-Performance Fixed-Shape Padded Dataloader for CUDA Graphs & ASDAG Models
=============================================================================
Guarantees invariant static tensor shapes [batch_size, seq_len] across every batch,
enabling seamless zero-overhead CUDA Graph execution without recompilation or shape crashes.
Supports both variable-length sequence collections and continuous byte/token streams.
"""

from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch


class PaddedDataLoader:
    """
    Fixed-shape, high-throughput dataloader engineered for CUDA Graph execution.

    Guarantees every batch yielded has shape [batch_size, seq_len] with correct
    padding and autoregressive targets masked by `ignore_index`.

    Parameters:
        data: Continuous 1D stream (np.ndarray, torch.Tensor, np.memmap) or collection
              of variable-length sequences (list of strings, bytes, or int/tensor sequences).
        batch_size: Fixed batch dimension B.
        seq_len: Fixed sequence length dimension T.
        pad_id: Token or byte ID used for input padding (default: 0).
        ignore_index: Target label assigned to padded positions to ignore loss (default: -100).
        shuffle: Whether to shuffle sample indices at each epoch (default: False).
        drop_last: If True, drops the final batch if smaller than batch_size.
                   If False and pad_remainder is True, pads final batch up to batch_size.
        pad_remainder: Whether to pad the remainder batch to full batch_size (default: True).
        device: Target device for output tensors (e.g. 'cuda', 'cpu', or torch.device).
        pin_memory: Whether to stage batches in page-locked host memory for zero-copy DMA.
        inject_eos: Whether to automatically append EOS (<|endoftext|>) to each sample/document (default: True).
        eos_token: Token or marker string/bytes/int to inject (default: "<|endoftext|>").
        as_stream: If True, flattens sequence collections into a single continuous 1D stream joined by EOS (default: False).
    """
    def __init__(
        self,
        data: Any,
        batch_size: int,
        seq_len: int,
        pad_id: int = 0,
        ignore_index: int = -100,
        shuffle: bool = False,
        drop_last: bool = False,
        pad_remainder: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: Optional[bool] = None,
        inject_eos: bool = True,
        eos_token: Union[str, bytes, int] = "<|endoftext|>",
        as_stream: bool = False,
    ):
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.ignore_index = int(ignore_index)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad_remainder = pad_remainder
        self.device = torch.device(device) if device is not None else None
        self.inject_eos = inject_eos
        self.eos_token = eos_token
        self.as_stream = as_stream

        # Determine if data is a continuous 1D byte/token stream or sequence collection
        self.is_stream = False
        if isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 1:
            self.is_stream = True
            if isinstance(data, np.ndarray):
                self.data = data
                self.stream_len = len(data)
            else:
                self.data = data.cpu().numpy() if data.is_cuda else data
                self.stream_len = len(self.data)
        elif hasattr(data, "shape") and len(data.shape) == 1:
            # memmap or tensor-like
            self.is_stream = True
            self.data = data
            self.stream_len = len(data)
        else:
            # Collection of sequences
            processed = self._preprocess_sequences(data)
            if self.as_stream and len(processed) > 0:
                self.is_stream = True
                self.data = np.concatenate(processed)
                self.stream_len = len(self.data)
            else:
                self.is_stream = False
                self.data = processed

        # Pin memory setup for fast asynchronous host-to-device transfers
        is_cuda_target = self.device is not None and self.device.type == "cuda"
        if pin_memory is None:
            self.pin_memory = is_cuda_target and torch.cuda.is_available()
        else:
            self.pin_memory = bool(pin_memory)

        if self.pin_memory:
            self._pinned_x = torch.empty((self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True)
            self._pinned_y = torch.empty((self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True)
        else:
            self._pinned_x = None
            self._pinned_y = None

    def _preprocess_sequences(self, data: Sequence[Any]) -> List[np.ndarray]:
        """Converts heterogeneous sequence items into 1D uint8/int64 numpy arrays with optional EOS injection."""
        processed = []
        eos_bytes: Optional[bytes] = None
        eos_int: Optional[int] = None
        if self.inject_eos:
            if isinstance(self.eos_token, str):
                eos_bytes = self.eos_token.encode("utf-8")
            elif isinstance(self.eos_token, (bytes, bytearray)):
                eos_bytes = bytes(self.eos_token)
            elif isinstance(self.eos_token, int):
                eos_int = int(self.eos_token)
                eos_bytes = bytes([self.eos_token % 256])

        for item in data:
            if isinstance(item, str):
                s = item
                if self.inject_eos:
                    if isinstance(self.eos_token, str) and not s.endswith(self.eos_token):
                        s = s + self.eos_token
                    elif eos_bytes is not None:
                        encoded = s.encode("utf-8")
                        if not encoded.endswith(eos_bytes):
                            encoded = encoded + eos_bytes
                        processed.append(np.frombuffer(encoded, dtype=np.uint8).astype(np.int64))
                        continue
                processed.append(np.frombuffer(s.encode("utf-8"), dtype=np.uint8).astype(np.int64))
            elif isinstance(item, (bytes, bytearray)):
                b = bytes(item)
                if self.inject_eos and eos_bytes is not None and not b.endswith(eos_bytes):
                    b = b + eos_bytes
                processed.append(np.frombuffer(b, dtype=np.uint8).astype(np.int64))
            elif isinstance(item, (torch.Tensor, np.ndarray)):
                arr = item.detach().cpu().numpy().astype(np.int64) if isinstance(item, torch.Tensor) else item.astype(np.int64)
                if self.inject_eos:
                    if eos_int is not None:
                        if len(arr) == 0 or arr[-1] != eos_int:
                            arr = np.append(arr, eos_int)
                    elif eos_bytes is not None:
                        eos_arr = np.frombuffer(eos_bytes, dtype=np.uint8).astype(np.int64)
                        if len(arr) < len(eos_arr) or not np.array_equal(arr[-len(eos_arr):], eos_arr):
                            arr = np.concatenate([arr, eos_arr])
                processed.append(arr)
            elif isinstance(item, (list, tuple)):
                arr = np.array(item, dtype=np.int64)
                if self.inject_eos:
                    if eos_int is not None:
                        if len(arr) == 0 or arr[-1] != eos_int:
                            arr = np.append(arr, eos_int)
                    elif eos_bytes is not None:
                        eos_arr = np.frombuffer(eos_bytes, dtype=np.uint8).astype(np.int64)
                        if len(arr) < len(eos_arr) or not np.array_equal(arr[-len(eos_arr):], eos_arr):
                            arr = np.concatenate([arr, eos_arr])
                processed.append(arr)
            else:
                raise TypeError(f"Unsupported sequence item type: {type(item)}")
        return processed

    def __len__(self) -> int:
        if self.is_stream:
            # Number of non-overlapping chunks of length seq_len
            total_tokens = self.stream_len - 1  # 1 token offset for target
            if total_tokens <= 0:
                return 0
            num_chunks = total_tokens // self.seq_len
            if self.drop_last:
                return num_chunks // self.batch_size
            return math.ceil(num_chunks / self.batch_size)
        else:
            n_samples = len(self.data)
            if self.drop_last:
                return n_samples // self.batch_size
            return math.ceil(n_samples / self.batch_size)

    def _get_stream_batch(self, chunk_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts fixed-shape (B, T) batch from continuous 1D stream."""
        B = self.batch_size
        T = self.seq_len
        actual_count = len(chunk_indices)

        x_arr = np.full((B, T), self.pad_id, dtype=np.int64)
        y_arr = np.full((B, T), self.ignore_index, dtype=np.int64)

        for i, idx in enumerate(chunk_indices):
            start = idx * T
            end = start + T
            if end + 1 <= self.stream_len:
                chunk = self.data[start : end + 1]
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()
                x_arr[i] = chunk[:T]
                y_arr[i] = chunk[1 : T + 1]
            elif start < self.stream_len:
                chunk = self.data[start:]
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()
                valid_len = min(len(chunk) - 1, T)
                if valid_len > 0:
                    x_arr[i, :valid_len] = chunk[:valid_len]
                    y_arr[i, :valid_len] = chunk[1 : valid_len + 1]

        return self._to_target_tensor(x_arr, y_arr)

    def _get_sequence_batch(self, sample_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads variable length sequences into exact [batch_size, seq_len] tensor with ignore masks."""
        B = self.batch_size
        T = self.seq_len

        x_arr = np.full((B, T), self.pad_id, dtype=np.int64)
        y_arr = np.full((B, T), self.ignore_index, dtype=np.int64)

        for i, idx in enumerate(sample_indices):
            seq = self.data[idx]
            L = len(seq)
            if L <= 1:
                # Sequence too short to form (input, target) pair
                continue

            # Cap length to seq_len + 1 (for next-token target)
            usable_L = min(L, T + 1)
            inp_len = usable_L - 1
            x_arr[i, :inp_len] = seq[:inp_len]
            y_arr[i, :inp_len] = seq[1:usable_L]

        return self._to_target_tensor(x_arr, y_arr)

    def get_batch_by_indices(self, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs an invariant static [batch_size, seq_len] batch from specific sample or chunk indices.
        Supports Priority Replay buffer interleave while maintaining static tensor dimensions.
        """
        indices_list = list(indices)
        B = self.batch_size
        if len(indices_list) < B:
            fill_idx = indices_list[0] if indices_list else 0
            indices_list = indices_list + [fill_idx] * (B - len(indices_list))
        elif len(indices_list) > B:
            indices_list = indices_list[:B]

        if self.is_stream:
            return self._get_stream_batch(indices_list)
        else:
            return self._get_sequence_batch(indices_list)


    def _to_target_tensor(self, x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transfers CPU numpy batch to target device using pinned buffers if available."""
        if self._pinned_x is not None and self._pinned_y is not None:
            self._pinned_x.copy_(torch.from_numpy(x_arr))
            self._pinned_y.copy_(torch.from_numpy(y_arr))
            if self.device is not None:
                x = self._pinned_x.to(self.device, non_blocking=True)
                y = self._pinned_y.to(self.device, non_blocking=True)
            else:
                x = self._pinned_x
                y = self._pinned_y
        else:
            x_t = torch.from_numpy(x_arr)
            y_t = torch.from_numpy(y_arr)
            if self.device is not None:
                x = x_t.to(self.device, non_blocking=True)
                y = y_t.to(self.device, non_blocking=True)
            else:
                x = x_t
                y = y_t
        return x, y

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.is_stream:
            total_chunks = (self.stream_len - 1) // self.seq_len
            indices = list(range(total_chunks))
            if self.shuffle:
                np.random.shuffle(indices)
        else:
            indices = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            if len(batch_indices) < self.batch_size:
                if self.drop_last:
                    break
                if not self.pad_remainder:
                    # Yielding dynamic batch size will break CUDA graph replay
                    pass

            if self.is_stream:
                yield self._get_stream_batch(batch_indices)
            else:
                yield self._get_sequence_batch(batch_indices)


# Aliases for clear intent in user code
FixedShapeDataLoader = PaddedDataLoader
CUDAGraphDataLoader = PaddedDataLoader


class HFStreamDataLoader:
    """
    Live-stream dataloader for HuggingFace streaming iterables (and any
    single-pass iterator), yielding the same static [batch_size, seq_len]
    (x, y) contract as PaddedDataLoader for CUDA-Graph-safe training.

    Reads lazily through a rolling byte buffer: no len(), no random access,
    no upfront materialization. Priority replay is unsupported (sequential
    only); pass use_priority_replay=False in the trainer.
    """

    def __init__(
        self,
        source: Any,
        batch_size: int,
        seq_len: int,
        pad_id: int = 0,
        ignore_index: int = -100,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: Optional[bool] = None,
        inject_eos: bool = True,
        eos_token: Union[str, bytes, int] = "<|endoftext|>",
        text_field: str = "text",
        restart: bool = True,
    ):
        self.source = source
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.ignore_index = int(ignore_index)
        self.inject_eos = inject_eos
        self.eos_token = eos_token
        self.text_field = text_field
        self.restart = restart
        self.device = torch.device(device) if device is not None else None
        self.is_stream = True
        self.stream_len = None
        self.tokens_seen = 0
        self._buf = bytearray()
        self._src_iter = None
        if isinstance(eos_token, str):
            self._eos_bytes = eos_token.encode("utf-8")
        elif isinstance(eos_token, (bytes, bytearray)):
            self._eos_bytes = bytes(eos_token)
        else:
            self._eos_bytes = bytes([int(eos_token) % 256])

        is_cuda_target = self.device is not None and self.device.type == "cuda"
        if pin_memory is None:
            self.pin_memory = is_cuda_target and torch.cuda.is_available()
        else:
            self.pin_memory = bool(pin_memory)
        if self.pin_memory:
            self._pinned_x = torch.empty((self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True)
            self._pinned_y = torch.empty((self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True)
        else:
            self._pinned_x = None
            self._pinned_y = None

    def __len__(self) -> int:
        raise TypeError("HFStreamDataLoader has no length (unbounded stream); iterate with a token budget instead.")

    def get_batch_by_indices(self, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("HFStreamDataLoader is sequential-only: random-access batches and priority replay are unsupported.")

    def _encode_item(self, item: Any) -> bytes:
        if isinstance(item, dict):
            if self.text_field not in item:
                raise KeyError(f"text_field={self.text_field!r} missing from stream item keys {sorted(item.keys())}")
            item = item[self.text_field]
        if isinstance(item, str):
            raw = item.encode("utf-8")
        elif isinstance(item, (bytes, bytearray)):
            raw = bytes(item)
        elif isinstance(item, torch.Tensor):
            raw = bytes((item.detach().cpu().numpy().astype(np.int64) % 256).tolist())
        elif isinstance(item, np.ndarray):
            raw = bytes((item.astype(np.int64) % 256).tolist())
        elif isinstance(item, (list, tuple)):
            raw = bytes((np.array(item, dtype=np.int64) % 256).tolist())
        else:
            raise TypeError(f"Unsupported stream item type: {type(item)}")
        if self.inject_eos and raw and not raw.endswith(self._eos_bytes):
            raw = raw + self._eos_bytes
        return raw

    def _pull(self) -> bool:
        if self._src_iter is None:
            try:
                self._src_iter = iter(self.source)
            except TypeError:
                return False
        try:
            chunk = self._encode_item(next(self._src_iter))
            if chunk:
                self._buf += chunk
            return True
        except StopIteration:
            if not self.restart:
                return False
            try:
                self._src_iter = iter(self.source)
            except TypeError:
                return False
            try:
                chunk = self._encode_item(next(self._src_iter))
                if chunk:
                    self._buf += chunk
                return True
            except StopIteration:
                return False

    def _to_target_tensor(self, x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._pinned_x is not None and self._pinned_y is not None:
            self._pinned_x.copy_(torch.from_numpy(x_arr))
            self._pinned_y.copy_(torch.from_numpy(y_arr))
            if self.device is not None:
                return (
                    self._pinned_x.to(self.device, non_blocking=True),
                    self._pinned_y.to(self.device, non_blocking=True),
                )
            return self._pinned_x, self._pinned_y
        x_t = torch.from_numpy(x_arr)
        y_t = torch.from_numpy(y_arr)
        if self.device is not None:
            return x_t.to(self.device, non_blocking=True), y_t.to(self.device, non_blocking=True)
        return x_t, y_t

    def _take_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        B = self.batch_size
        T = self.seq_len
        need = B * (T + 1)
        while len(self._buf) < need:
            if not self._pull():
                break
        if not self._buf:
            return None
        x_arr = np.full((B, T), self.pad_id, dtype=np.int64)
        y_arr = np.full((B, T), self.ignore_index, dtype=np.int64)
        rows = min(len(self._buf) // (T + 1), B)
        if rows == 0:
            tail = bytes(self._buf)
            del self._buf[:]
            self.tokens_seen += len(tail)
            k = min(len(tail), T)
            x_arr[0, :k] = np.frombuffer(tail[:k], dtype=np.uint8).astype(np.int64)
            if len(tail) >= 2:
                y_arr[0, :k - 1] = np.frombuffer(tail[1:k], dtype=np.uint8).astype(np.int64)
            return self._to_target_tensor(x_arr, y_arr)
        consume = rows * (T + 1)
        flat = bytes(self._buf[:consume])
        del self._buf[:consume]
        self.tokens_seen += consume
        for i in range(rows):
            seg = flat[i * (T + 1):(i + 1) * (T + 1)]
            x_arr[i] = np.frombuffer(seg[:T], dtype=np.uint8).astype(np.int64)
            y_arr[i] = np.frombuffer(seg[1:T + 1], dtype=np.uint8).astype(np.int64)
        return self._to_target_tensor(x_arr, y_arr)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            batch = self._take_batch()
            if batch is None:
                return
            yield batch
