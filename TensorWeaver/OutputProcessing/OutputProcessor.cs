using System.Collections.ObjectModel;
using TensorWeaver.OutputData;

namespace TensorWeaver.OutputProcessing;

public interface OutputProcessor<T>
{
	float MinimumConfidence { get; set; }
	ReadOnlyCollection<T> Process(RawOutput output);
}