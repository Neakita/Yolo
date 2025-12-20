using TensorWeaver.OutputData;

namespace TensorWeaver.OutputProcessing;

public interface OutputSpanProcessor<T>
{
	int Process(RawOutput output, Span<T> target);
}