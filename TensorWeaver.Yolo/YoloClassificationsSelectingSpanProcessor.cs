using System.Numerics.Tensors;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloClassificationsSelectingSpanProcessor : OutputSpanProcessor<Classification>
{
	public int Process(RawOutput output, Span<Classification> target)
	{
		var span = output.Tensors[0].Buffer.Span;
		var classificationsCount = span.Length;
		Span<float> confidences = stackalloc float[classificationsCount];
		Span<int> classIds = stackalloc int[target.Length];
		span.CopyTo(confidences);
		Sort(confidences, classIds);
		confidences = confidences[..target.Length];
		classIds = classIds[..target.Length];
		for (int i = 0; i < target.Length; i++)
		{
			var classId = (ushort)classIds[i];
			var confidence = confidences[i];
			var classification = new Classification(classId, confidence);
			target[i] = classification;
		}
		return target.Length;
	}

	private static void Sort(Span<float> confidences, Span<int> classIds)
	{
		Span<int> allClassIds = stackalloc int[confidences.Length];
		for (int i = 0; i < classIds.Length; i++)
			allClassIds[i] = i;
		allClassIds[classIds.Length..].Fill(-1);
		for (int round = 0; round < classIds.Length; round++)
			SelectionSortOnce(confidences, allClassIds, round);
		allClassIds[..classIds.Length].CopyTo(classIds);
	}

	private static void SelectionSortOnce(Span<float> confidences, Span<int> classIds, int round)
	{
		var indexOfMax = TensorPrimitives.IndexOfMax(confidences[round..]);
		(confidences[round], confidences[indexOfMax]) = (confidences[indexOfMax], confidences[round]);
		if (classIds[indexOfMax] == -1)
			classIds[indexOfMax] = indexOfMax;
		(classIds[round], classIds[indexOfMax]) = (classIds[indexOfMax], classIds[round]);
	}
}