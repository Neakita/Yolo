using System.Collections.ObjectModel;

namespace Yolo.OutputProcessing;

public interface OutputProcessor<T>
{
	float MinimumConfidence { get; set; }
	ReadOnlyCollection<T> Process(RawOutput output);
}