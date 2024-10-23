using System.Collections.ObjectModel;
using Yolo.OutputData;

namespace Yolo.OutputProcessing;

public interface OutputProcessor<T>
{
	float MinimumConfidence { get; set; }
	ReadOnlyCollection<T> Process(RawOutput output);
}