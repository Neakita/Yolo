using Microsoft.ML.OnnxRuntime.Tensors;

namespace TensorWeaver.Metadata;

internal readonly struct TensorShape
{
	public int Length { get; }
	public int[] Dimensions { get; }
	public long[] Dimensions64 { get; }

	public TensorShape(int[] dimensions)
	{
		Length = GetSizeForShape(dimensions);
		Dimensions = dimensions;
		Dimensions64 = dimensions.Select(x => (long)x).ToArray();
	}

	internal DenseTensor<float> AllocateTensor()
	{
		return new DenseTensor<float>(new float[Length], Dimensions);
	}

	private static int GetSizeForShape(ReadOnlySpan<int> shape)
	{
		var product = 1;
		foreach (var dimension in shape)
		{
			if (dimension < 0)
				throw new ArgumentOutOfRangeException($"Shape must not have negative elements: {dimension}");
			product = checked(product * dimension);
		}
		return product;
	}
}