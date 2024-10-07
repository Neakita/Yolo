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
		PrepareBuffer();
		for (ushort classIndex = 0; classIndex < span.Length; classIndex++)
		{
			var confidence = span[classIndex];
			if (confidence < MinimumConfidence)
				continue;
			Classification classification = new(classIndex, confidence);
			_buffer.Add(classification);
		}
		_buffer.Sort(ReversePredictionConfidenceComparer.Instance);
		return _buffer;
	}

	private readonly PooledList<Classification> _buffer = new();
	private float _minimumConfidence;

	private void PrepareBuffer()
	{
		_buffer.Clear();
	}
}