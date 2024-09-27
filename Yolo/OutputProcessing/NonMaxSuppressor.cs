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
		var area1 = first.Width * first.Height;

		if (area1 <= 0f)
		{
			return 0f;
		}

		var area2 = second.Width * second.Height;

		if (area2 <= 0f)
		{
			return 0f;
		}

		var intersection = GetIntersection(first, second);
		var intersectionArea = intersection.Width * intersection.Height;

		return intersectionArea / (area1 + area2 - intersectionArea);
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