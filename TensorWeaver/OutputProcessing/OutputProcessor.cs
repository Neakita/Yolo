using TensorWeaver.OutputData;

namespace TensorWeaver.OutputProcessing;

public interface OutputProcessor<out T>
{
	T Process(RawOutput output);
}