using System.Buffers;

namespace Yolo;

internal sealed class MemoryPoolArrayBuffer<T> : IMemoryOwner<T>
{
	public Memory<T> Memory
	{
		get
		{
			ObjectDisposedException.ThrowIf(_buffer == null, this);
			return new Memory<T>(_buffer, 0, _length);
		}
	}

	public MemoryPoolArrayBuffer(int length)
	{
		_length = length;
		_buffer = ArrayPool<T>.Shared.Rent(length);
	}

	public void Dispose()
	{
		if (_buffer != null)
			ArrayPool<T>.Shared.Return(_buffer);
	}

	private readonly int _length;
	private readonly T[]? _buffer;
}