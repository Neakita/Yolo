using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloClassificationsProcessor : OutputProcessor<Classification[]>
{
	public int ClassificationsLimit { get; set; } = 5;

	public YoloClassificationsProcessor()
	{
		_spanProcessor = new YoloClassificationsSelectingSpanProcessor();
	}

	public YoloClassificationsProcessor(OutputSpanProcessor<Classification> spanProcessor)
	{
		_spanProcessor = spanProcessor;
	}

	public Classification[] Process(RawOutput output)
	{
		var span = output.Tensors[0].Buffer.Span;
		var classificationsCount = span.Length;
		var classificationsCountToReturn = Math.Min(ClassificationsLimit, classificationsCount);
		var classifications = new Classification[classificationsCountToReturn];
		_spanProcessor.Process(output, classifications);
		return classifications;
	}

	private readonly OutputSpanProcessor<Classification> _spanProcessor;
}