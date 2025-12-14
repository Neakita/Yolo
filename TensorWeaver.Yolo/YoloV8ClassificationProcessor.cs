using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8ClassificationProcessor : OutputProcessor<Classification>
{
	public Classification Process(RawOutput output)
	{
		var span = output.Tensors[0].Buffer.Span;
		var classification = new Classification(0, span[0]); 
		for (byte classIndex = 1; classIndex < span.Length; classIndex++)
		{
			var confidence = span[classIndex];
			if (confidence > classification.Confidence)
				classification = new Classification(classIndex, confidence);
		}
		return classification;
	}
}