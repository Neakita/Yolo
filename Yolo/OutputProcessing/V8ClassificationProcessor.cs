using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8ClassificationProcessor : OutputProcessor<Classification>
{
	public float MinimumConfidence
	{
		get => _minimumConfidence;
		set
		{
			Guard.IsInRange(value, 0, 1);
			_minimumConfidence = value;
		}
	}

	public IReadOnlyList<Classification> Process(RawOutput output)
	{
		var span = output.Output0.Buffer.Span;
		PooledList<Classification> buffer = new();
		for (ushort classIndex = 0; classIndex < span.Length; classIndex++)
		{
			var confidence = span[classIndex];
			if (confidence < MinimumConfidence)
				continue;
			Classification classification = new(classIndex, confidence);
			buffer.Add(classification);
		}
		buffer.Sort(ReversePredictionConfidenceComparer.Instance);
		return buffer;
	}

	private float _minimumConfidence;
}