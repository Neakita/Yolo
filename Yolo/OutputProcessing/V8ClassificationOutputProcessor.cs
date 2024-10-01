using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8ClassificationOutputProcessor : OutputProcessor<Classification>
{
	public override IEnumerable<Classification> Process(RawOutput output)
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
}