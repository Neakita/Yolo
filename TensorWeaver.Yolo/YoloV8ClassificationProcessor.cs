using System.Numerics.Tensors;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8ClassificationProcessor : OutputProcessor<Classification>
{
	public Classification Process(RawOutput output)
	{
		var confidences = output.Tensors[0].Buffer.Span;
		var classId = TensorPrimitives.IndexOfMax(confidences);
		var confidence = confidences[classId];
		return new Classification((ushort)classId, confidence);
	}
}