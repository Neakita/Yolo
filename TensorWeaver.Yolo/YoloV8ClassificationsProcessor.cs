using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8ClassificationsProcessor : OutputProcessor<Classification[]>
{
	public int ClassificationsLimit { get; set; } = 5;

	public Classification[] Process(RawOutput output)
	{
		var span = output.Tensors[0].Buffer.Span;
		var classificationsCount = span.Length;
		Span<float> confidences = stackalloc float[classificationsCount];
		span.CopyTo(confidences);
		Span<int> classIds = stackalloc int[classificationsCount];
		FillWithIndexes(classIds);
		confidences.Sort(classIds, static (x, y) => y.CompareTo(x));
		var classificationsCountToReturn = Math.Min(ClassificationsLimit, classificationsCount);
		var classifications = new Classification[classificationsCountToReturn];
		for (int i = 0; i < classificationsCountToReturn; i++)
		{
			var classId = (ushort)classIds[i];
			var confidence = confidences[i];
			var classification = new Classification(classId, confidence);
			classifications[i] = classification;
		}
		return classifications;
	}

	private static void FillWithIndexes(Span<int> span)
	{
		for (int i = 0; i < span.Length; i++)
			span[i] = i;
	}
}