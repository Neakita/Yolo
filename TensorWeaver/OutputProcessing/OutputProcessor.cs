using TensorWeaver.OutputData;

namespace TensorWeaver.OutputProcessing;

public interface OutputProcessor<out T>
{
	float MinimumConfidence { get; set; }
	T Process(RawOutput output);
}