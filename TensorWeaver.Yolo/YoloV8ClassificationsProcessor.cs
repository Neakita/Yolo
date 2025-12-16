using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8ClassificationsProcessor : OutputProcessor<Classification[]>
{
	public Classification[] Process(RawOutput output)
	{
		var span = output.Tensors[0].Buffer.Span;
		var classifications = new Classification[span.Length];
		for (int i = 0; i < span.Length; i++)
		{
			var confidence = span[i];
			Classification classification = new((ushort)i, confidence);
			classifications[i] = classification;
		}
		classifications.Sort(ReversePredictionConfidenceComparer.Instance);
		return classifications;
	}
}