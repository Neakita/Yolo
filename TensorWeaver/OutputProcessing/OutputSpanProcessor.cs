using TensorWeaver.OutputData;

namespace TensorWeaver.OutputProcessing;

public interface OutputSpanProcessor<T>
{
	void Process(RawOutput output, Span<T> target);
}