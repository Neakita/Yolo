using CommunityToolkit.Diagnostics;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class V8ClassificationProcessor : OutputProcessor<List<Classification>>
{
	public float MinimumConfidence
	{
		get;
		set
		{
			Guard.IsInRange(value, 0, 1);
			field = value;
		}
	}

	public List<Classification> Process(RawOutput output)
	{
		var span = output.Tensors[0].Buffer.Span;
		var classifications = new List<Classification>();
		for (ushort classIndex = 0; classIndex < span.Length; classIndex++)
		{
			var confidence = span[classIndex];
			if (confidence < MinimumConfidence)
				continue;
			Classification classification = new(classIndex, confidence);
			classifications.Add(classification);
		}
		classifications.Sort(ReversePredictionConfidenceComparer.Instance);
		return classifications;
	}
}