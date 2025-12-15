using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8ClassificationsProcessor : OutputProcessor<List<Classification>>
{
	public float MinimumConfidence
	{
		get;
		set
		{
			if (value is <= 0 or >= 1)
				throw new ArgumentOutOfRangeException(nameof(MinimumConfidence), value, $"Value for {MinimumConfidence} should be exclusively between 0 and 1, but was {value}");
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