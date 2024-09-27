using System.Collections.Immutable;

namespace Yolo.OutputProcessing;

public sealed class V8ClassificationOutputProcessor : OutputProcessor<Classification>
{
	public override ImmutableArray<Classification> Process(RawOutput output)
	{
		var span = output.Output0.Buffer.Span;
		var builder = ImmutableArray.CreateBuilder<Classification>(span.Length);
		for (ushort i = 0; i < span.Length; i++)
		{
			var confidence = span[i];
			if (confidence < MinimumConfidence)
				continue;
			builder.Add(new Classification(i, confidence));
		}
		builder.Sort(ReversePredictionConfidenceComparer.Instance);
		return builder.DrainToImmutable();
	}
}