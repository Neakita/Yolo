using System.Collections.Immutable;

namespace Yolo;

public sealed class ClassificationOutputProcessor : OutputProcessor<Classification>
{
	public override ImmutableArray<Classification> Process(RawOutput output)
	{
		var span = output.Output0.Buffer.Span;
		var builder = ImmutableArray.CreateBuilder<Classification>(span.Length);
		for (ushort i = 0; i < span.Length; i++)
			builder.Add(new Classification(i, span[i]));
		builder.Sort(ReversePredictionConfidenceComparer.Instance);
		return builder.DrainToImmutable();
	}
}