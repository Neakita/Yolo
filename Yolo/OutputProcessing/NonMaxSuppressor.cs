using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

internal class NonMaxSuppressor
{
	public float MaximumIoU
	{
		get => _maximumIoU;
		set
		{
			Guard.IsBetween(value, 0, 1);
			_maximumIoU = value;
		}
	}

	public IEnumerable<Detection> Suppress(
		IEnumerable<Detection> detections)
	{
		using var enumerator = detections.GetEnumerator();
		if (!enumerator.MoveNext())
			yield break;
		var firstDetection = enumerator.Current;
		yield return firstDetection;
		using PooledList<Detection> passedDetections = [firstDetection];
		while (enumerator.MoveNext())
		{
			var detection = enumerator.Current;
			if (Intersects(detection, passedDetections))
				continue;
			passedDetections.Add(detection);
			yield return detection;
		}
	}

	private float _maximumIoU = 0.45f;

	private static IEnumerable<Detection> GuardOrder(IEnumerable<Detection> detections)
	{
		using var enumerator = detections.GetEnumerator();
		if (!enumerator.MoveNext())
			yield break;
		var previous = enumerator.Current;
		yield return previous;
		while (enumerator.MoveNext())
		{
			var current = enumerator.Current;
			Guard.IsLessThanOrEqualTo(current.Confidence, previous.Confidence);
			previous = current;
			yield return current;
		}
	}

	private bool Intersects(Detection subject, IEnumerable<Detection> passedSubjects)
	{
		foreach (var passedDetection in passedSubjects)
		{
			if (subject.ClassId != passedDetection.ClassId)
				continue;
			var intersection = CalculateIoU(subject.Bounding, passedDetection.Bounding);
			if (intersection > _maximumIoU)
				return true;
		}
		return false;
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