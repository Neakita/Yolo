using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloClassificationsSpanProcessor : OutputSpanProcessor<Classification>
{
	public int Process(RawOutput output, Span<Classification> target)
	{
		var span = output.Tensors[0].Buffer.Span;
		var classificationsCount = span.Length;
		Span<float> confidences = stackalloc float[classificationsCount];
		Span<int> classIds = stackalloc int[classificationsCount];
		span.CopyTo(confidences);
		FillWithIndexes(classIds);
		confidences.Sort(classIds);
		confidences = confidences[^target.Length..];
		classIds = classIds[^target.Length..];
		confidences.Reverse();
		classIds.Reverse();
		for (int i = 0; i < target.Length; i++)
		{
			var classId = (ushort)classIds[i];
			var confidence = confidences[i];
			var classification = new Classification(classId, confidence);
			target[i] = classification;
		}
		return target.Length;
	}

	private static void FillWithIndexes(Span<int> span)
	{
		for (int i = 0; i < span.Length; i++)
			span[i] = i;
	}
}