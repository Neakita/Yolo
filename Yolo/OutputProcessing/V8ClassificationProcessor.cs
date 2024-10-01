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

	public IEnumerable<Classification> Process(RawOutput output)
	{
		var outputVersion = output.Version;
		var span = output.Output0.Buffer.Span;
		using PooledList<Classification> classifications = new();
		for (ushort i = 0; i < span.Length; i++)
		{
			var confidence = span[i];
			if (confidence < MinimumConfidence)
				continue;
			classifications.Add(new Classification(i, confidence));
		}
		classifications.Sort(ReversePredictionConfidenceComparer.Instance);
		foreach (var classification in classifications)
		{
			Guard.IsEqualTo(output.Version, outputVersion);
			yield return classification;
		}
	}

	private float _minimumConfidence;
}