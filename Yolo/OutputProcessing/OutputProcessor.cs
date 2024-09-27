using System.Collections.Immutable;

namespace Yolo.OutputProcessing;

public abstract class OutputProcessor<T>
{
	public float MinimumConfidence { get; set; } = 0.3f;

	public abstract ImmutableArray<T> Process(RawOutput output);
}