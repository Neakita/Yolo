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

	public PooledList<Detection> Suppress(PooledList<Detection> detections)
	{
		var firstDetection = detections[0];
		PooledList<Detection> buffer = [firstDetection];
		for (var i = 1; i < detections.Count; i++)
		{
			var detection = detections[i];
			if (Intersects(detection, buffer))
				continue;
			buffer.Add(detection);
		}
		return buffer;
	}

	private float _maximumIoU = 0.45f;

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