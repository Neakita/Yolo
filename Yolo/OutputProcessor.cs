using System.Collections.Immutable;

namespace Yolo;

public abstract class OutputProcessor<T>
{
	public abstract ImmutableArray<T> Process(RawOutput output);
}