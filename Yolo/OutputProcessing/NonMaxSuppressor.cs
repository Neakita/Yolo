using System.Collections.Immutable;
using Collections.Pooled;

namespace Yolo.OutputProcessing;

internal static class NonMaxSuppressor
{
	public static ImmutableArray<Detection> SuppressAndCombine(
		PooledList<Detection> detections,
		float maximumIoU)
	{
		if (detections.Count == 0)
			return ImmutableArray<Detection>.Empty;
		var resultBuilder = ImmutableArray.CreateBuilder<Detection>(3);
		detections.Sort(ReverseDetectionClassificationConfidenceComparer.Instance);
		resultBuilder.Add(detections[0]);
		for (int i = 1; i < detections.Count; i++)
		{
			var detectionToAdd = detections[i];
			var addToResult = true;
			foreach (var alreadyAddedDetection in resultBuilder)
			{
				if (detectionToAdd.Classification.ClassId != alreadyAddedDetection.Classification.ClassId)
					continue;
				var intersectionOverUnion = CalculateIoU(detectionToAdd.Bounding, alreadyAddedDetection.Bounding);
				if (intersectionOverUnion > maximumIoU)
				{
					addToResult = false;
					break;
				}
			}
			if (addToResult)
				resultBuilder.Add(detectionToAdd);
		}
		return resultBuilder.DrainToImmutable();
	}

	private static float CalculateIoU(in Bounding first, in Bounding second)
	{
		if (first.Area == 0)
			return 0;
		if (second.Area == 0)
			return 0;
		var intersection = GetIntersection(first, second);
		return intersection.Area / (first.Area + second.Area - intersection.Area);
	}

	private static Bounding GetIntersection(Bounding first, Bounding second)
	{
		float left = Math.Max(first.Left, second.Left);
		float right = Math.Min(first.Right, second.Right);
		float top = Math.Max(first.Top, second.Top);
		float bottom = Math.Min(first.Bottom, second.Bottom);
		if (right >= left && bottom >= top)
			return new Bounding(left, top, right, bottom);
		return Bounding.Empty;
	}
}