using System.Numerics.Tensors;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8ClassificationProcessor : OutputProcessor<Classification>
{
	public Classification Process(RawOutput output)
	{
		var span = output.Tensors[0].Buffer.Span;
		var index = TensorPrimitives.IndexOfMax(span);
		return new Classification((ushort)index, span[index]);
	}
}