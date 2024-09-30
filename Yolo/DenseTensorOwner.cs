using System.Buffers;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yolo;

internal sealed class DenseTensorOwner<T> : IDisposable
{
	public static DenseTensorOwner<T> Allocate(TensorShape shape)
	{
		MemoryPoolArrayBuffer<T> buffer = new(shape.Length);
		return new DenseTensorOwner<T>(buffer, shape.Dimensions);
	}

	public DenseTensor<T> Tensor
	{
		get
		{
			ObjectDisposedException.ThrowIf(_tensor == null, this);
			return _tensor;
		}
	}

	public DenseTensorOwner(IMemoryOwner<T> memoryOwner, ReadOnlySpan<int> dimensions)
	{
		_memoryOwner = memoryOwner;
		_tensor = new DenseTensor<T>(memoryOwner.Memory, dimensions);
	}

	public void Dispose()
	{
		_memoryOwner.Dispose();
	}

	private readonly IMemoryOwner<T> _memoryOwner;
	private readonly DenseTensor<T>? _tensor;
}