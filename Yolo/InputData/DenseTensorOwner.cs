using System.Buffers;
using Microsoft.ML.OnnxRuntime.Tensors;
using Yolo.Metadata;

namespace Yolo.InputData;

internal sealed class DenseTensorOwner<T> : IDisposable
{
	public static DenseTensorOwner<T> Allocate(TensorShape shape)
	{
		var memoryOwner = MemoryPool<T>.Shared.Rent(shape.Length);
		DenseTensor<T> tensor = new(memoryOwner.Memory[..shape.Length], shape.Dimensions);
		return new DenseTensorOwner<T>(tensor, memoryOwner);
	}

	public DenseTensor<T> Tensor { get; }

	private DenseTensorOwner(DenseTensor<T> tensor, IMemoryOwner<T> memoryOwner)
	{
		Tensor = tensor;
		_memoryOwner = memoryOwner;
	}

	public void Dispose()
	{
		_memoryOwner.Dispose();
	}

	private readonly IMemoryOwner<T> _memoryOwner;
}