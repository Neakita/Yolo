using TensorWeaver.OutputData;

namespace TensorWeaver.Yolo;

public sealed class NonMaxSpanSuppressor
{
	public float MaximumIoU
	{
		get;
		set
		{
			if (value is < 0 or > 1)
				throw new ArgumentOutOfRangeException(nameof(MaximumIoU), value,
					$"Value for {MaximumIoU} should be inclusively between 0 and 1, but was {value}");
			field = value;
		}
	} = 0.5f;

	public int Suppress(ReadOnlySpan<Detection> source, Span<Detection> target)
	{
		var passedDetectionsCount = 0;
		var passedDetections = Span<Detection>.Empty;
		foreach (var detection in source)
		{
			if (Intersects(detection, passedDetections))
				continue;
			target[passedDetectionsCount] = detection;
			passedDetectionsCount++;
			passedDetections = target[..passedDetectionsCount];
		}
		return passedDetectionsCount;
	}

	private bool Intersects(Detection detection, ReadOnlySpan<Detection> passedDetections)
	{
		foreach (var passedDetection in passedDetections)
		{
			if (detection.ClassId != passedDetection.ClassId)
				continue;
			var intersection = CalculateIoU(detection.Bounding, passedDetection.Bounding);
			if (intersection > MaximumIoU)
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